"""
Agente 2: Percepto Emocional

Responsável pela classificação emocional multimodal usando:
- Deep Learning (ResNet50 + RoBERTa)
- Análise visual + textual (caption)
- Geração de captions (SAT - Show, Attend and Tell)

Este agente USA contexto linguístico através de captions.
"""

from typing import Dict, Union, Optional, List
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import RobertaTokenizer
import torchvision.transforms as T

# Import model architecture from deep-mind
import sys
sys.path.append(str(Path(__file__).parent.parent.parent / "deep-mind" / "v1_baseline"))
from multimodal_classifier import MultimodalEmotionClassifier


class PerceptoEmocional:
    """
    Agente 2: Percepto Emocional (Deep Learning)

    Usa modelo multimodal ResNet50 + RoBERTa treinado (v1 baseline)
    para classificar emoções em arte usando imagem + caption.
    """

    # ArtEmis emotion labels (9 classes)
    EMOTIONS = [
        'amusement', 'awe', 'contentment', 'excitement',
        'anger', 'disgust', 'fear', 'sadness', 'something else'
    ]

    def __init__(
        self,
        model_path: Optional[str] = None,
        sat_checkpoint: Optional[str] = None,
        device: Optional[str] = None
    ):
        """
        Inicializa o Percepto Emocional.

        Args:
            model_path: Caminho para checkpoint do modelo v1 treinado
                       (default: busca em /data/paloma/deep-mind-checkpoints/)
            sat_checkpoint: Caminho para checkpoint do SAT (caption generation)
            device: 'cuda' ou 'cpu' (default: auto-detect)
        """
        # Device setup
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        print(f"🧠 Inicializando Percepto Emocional no device: {self.device}")

        # Load v1 checkpoint
        if model_path is None:
            model_path = self._find_best_checkpoint()

        self.model_path = Path(model_path)
        print(f"📂 Carregando checkpoint: {self.model_path}")

        # Load model
        checkpoint = torch.load(self.model_path, map_location=self.device)
        self.model = MultimodalEmotionClassifier(
            num_emotions=9,
            freeze_image_encoder=True,
            freeze_text_encoder=False,
            dropout=0.3
        )
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        # Emotion labels mapping (nome → índice)
        self.emotion_labels = {
            'amusement': 0, 'awe': 1, 'contentment': 2, 'excitement': 3,
            'anger': 4, 'disgust': 5, 'fear': 6, 'sadness': 7, 'something else': 8
        }

        # Load tokenizer
        self.tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

        # Image transforms (same as training)
        self.transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(
                mean=[0.485, 0.456, 0.406],  # ImageNet statistics
                std=[0.229, 0.224, 0.225]
            )
        ])

        # SAT model for caption generation (lazy load)
        self.sat_model = None
        self.sat_checkpoint = sat_checkpoint

        # Training info
        self.train_acc = checkpoint.get('train_acc', 'N/A')
        self.val_acc = checkpoint.get('val_acc', 'N/A')
        self.epoch = checkpoint.get('epoch', 'N/A')

        print(f"✅ Modelo carregado! Epoch {self.epoch} | Train: {self.train_acc} | Val: {self.val_acc}")

    def _find_best_checkpoint(self) -> str:
        """Busca o melhor checkpoint V1 disponível (V3 precisa de classe diferente)."""
        base_dir = Path("/data/paloma/deep-mind-checkpoints")
        
        # Por enquanto usa V1 (V3 tem arquitetura diferente - MultimodalFuzzyClassifier)
        v1_dir = base_dir / "multimodal_20251120_220404"
        if not v1_dir.exists():
            raise FileNotFoundError(
                f"Nenhum checkpoint V1 encontrado em {base_dir}\n"
                "Execute o treinamento primeiro."
            )

        # Procura checkpoint_best.pt primeiro
        best_ckpt = v1_dir / "checkpoint_best.pt"
        if best_ckpt.exists():
            print(f"📦 Usando V1 baseline (epoch 2): {best_ckpt}")
            return str(best_ckpt)

        # Senão, pega o último epoch
        ckpts = sorted(v1_dir.glob("checkpoint_epoch*.pt"))
        if not ckpts:
            raise FileNotFoundError(f"Nenhum checkpoint encontrado em {v1_dir}")

        print(f"📦 Usando V1 baseline (epoch 8): {ckpts[-1]}")
        return str(ckpts[-1])

    def _load_sat_model(self):
        """Lazy loading do modelo SAT para geração de captions."""
        if self.sat_model is not None:
            return

        print("📝 Carregando modelo SAT para geração de captions...")

        # Add artemis M2 path to sys.path first (BEFORE importing sat_loader)
        artemis_m2_path = Path(__file__).parent.parent.parent / "artemis-v2" / "neural_speaker" / "m2"
        if str(artemis_m2_path) not in sys.path:
            sys.path.insert(0, str(artemis_m2_path))

        # Lazy import SAT loader (now models.transformer will be found)
        fuzzy_brain_path = Path(__file__).parent.parent.parent / "fuzzy-brain" / "fuzzy_brain"
        if str(fuzzy_brain_path) not in sys.path:
            sys.path.insert(0, str(fuzzy_brain_path))
        
        from sat_loader_classic import SATModelLoader

        # Find SAT checkpoint if not provided
        if self.sat_checkpoint is None:
            artemis_root = Path(__file__).parent.parent.parent / 'artemis-v2'
            self.sat_checkpoint = artemis_root / 'sat_logs' / 'sat_combined' / 'checkpoints' / 'best_model.pt'
            vocab_path = artemis_root / 'dataset' / 'combined' / 'train' / 'vocabulary.pkl'

        self.sat_model = SATModelLoader(
            checkpoint_path=str(self.sat_checkpoint),
            vocab_path=str(vocab_path),
            device=str(self.device)
        )
        print("✅ SAT carregado!")

    def _load_image(self, image: Union[str, Path, np.ndarray, Image.Image]) -> Image.Image:
        """Carrega imagem de diferentes formatos para PIL.Image."""
        if isinstance(image, (str, Path)):
            return Image.open(image).convert('RGB')
        elif isinstance(image, np.ndarray):
            return Image.fromarray(image).convert('RGB')
        elif isinstance(image, Image.Image):
            return image.convert('RGB')
        else:
            raise ValueError(f"Formato de imagem não suportado: {type(image)}")
    
    def _preprocess_image_for_sat(self, image: Union[str, Path, np.ndarray, Image.Image]) -> torch.Tensor:
        """
        Preprocessa imagem para o encoder SAT (que tem seu próprio ResNet).
        SAT espera imagem RGB bruta (3, 224, 224), não features pré-processadas.
        """
        from torchvision import transforms
        
        # Load image
        pil_image = self._load_image(image)
        
        # Transform padrão para SAT (sem normalização específica, SAT faz internamente)
        sat_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet norm
        ])
        
        image_tensor = sat_transform(pil_image)  # (3, 224, 224)
        return image_tensor.unsqueeze(0)  # (1, 3, 224, 224)

    def _preprocess_image(self, image: Image.Image) -> torch.Tensor:
        """Preprocessa imagem para o modelo."""
        img_tensor = self.transform(image)
        return img_tensor.unsqueeze(0).to(self.device)  # (1, 3, 224, 224)

    def _tokenize_text(self, text: str, max_length: int = 128) -> Dict[str, torch.Tensor]:
        """Tokeniza texto usando RoBERTa."""
        encoding = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=max_length,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].to(self.device),
            'attention_mask': encoding['attention_mask'].to(self.device)
        }

    def generate_caption(
        self, 
        image: Union[str, Path, np.ndarray, Image.Image],
        emotion: Optional[str] = None,
        max_len: int = 54,
        beam_size: int = 5
    ) -> str:
        """
        Gera caption afetiva usando modelo SAT (Show, Attend and Tell).

        Args:
            image: Caminho da imagem ou array numpy
            emotion: Emoção opcional para condicionar a geração (e.g., 'awe', 'sadness')
            max_len: Comprimento máximo da caption
            beam_size: Tamanho do beam search

        Returns:
            Caption gerada automaticamente
        """
        # Lazy load SAT model
        if self.sat_model is None:
            self._load_sat_model()
        
        # Preprocessa imagem para SAT (RGB tensor 1x3x224x224)
        image_tensor = self._preprocess_image_for_sat(image)
        
        # Convert emotion string to one-hot tensor if provided
        emotion_onehot = None
        if emotion is not None:
            # Map emotion name to index
            emotion_idx = self.emotion_labels.get(emotion.lower(), None)
            if emotion_idx is not None:
                emotion_onehot = torch.zeros(1, 9).to(self.device)
                emotion_onehot[0, emotion_idx] = 1.0
        
        # Generate caption usando SAT (SAT encoder processará a imagem)
        caption = self.sat_model.generate(
            image_features=image_tensor,
            emotion_onehot=emotion_onehot,
            max_len=max_len,
            beam_size=beam_size
        )
        
        return caption

    @torch.no_grad()
    def analyze(
        self,
        image: Union[str, Path, np.ndarray, Image.Image],
        caption: Optional[str] = None,
        return_probabilities: bool = True,
        auto_caption: bool = False
    ) -> Dict:
        """
        Analisa imagem + caption usando deep learning multimodal.

        Args:
            image: Caminho da imagem, array numpy ou PIL.Image
            caption: Caption opcional (se None, usa caption padrão ou gera com SAT)
            return_probabilities: Retornar distribuição de probabilidades
            auto_caption: Se True, gera caption automaticamente usando SAT

        Returns:
            Dict com:
                - emotion: emoção dominante (str)
                - confidence: confiança da predição (float 0-1)
                - probabilities: distribuição completa (Dict[str, float])
                - caption: caption usado (str)
                - caption_source: 'user', 'default' ou 'generated'
        """
        # Load image
        pil_image = self._load_image(image)

        # Get or generate caption
        caption_source = 'default'
        if caption is None and auto_caption:
            # Generate caption automatically using SAT
            caption = self.generate_caption(pil_image)
            caption_source = 'generated'
        elif caption is None:
            # Use default caption if no auto-generation
            caption = "A painting."
            caption_source = 'default'
        else:
            caption_source = 'user'

        # Preprocess
        img_tensor = self._preprocess_image(pil_image)
        text_encoding = self._tokenize_text(caption)

        # Forward pass
        logits = self.model(
            img_tensor,
            text_encoding['input_ids'],
            text_encoding['attention_mask']
        )

        # Get probabilities
        probs = F.softmax(logits, dim=-1)[0]  # (9,)
        probs_np = probs.cpu().numpy()

        # Get dominant emotion
        pred_idx = torch.argmax(probs).item()
        dominant_emotion = self.EMOTIONS[pred_idx]
        confidence = float(probs_np[pred_idx])

        # Build result
        result = {
            'emotion': dominant_emotion,
            'confidence': confidence,
            'caption': caption,
            'caption_source': caption_source,
            'model': 'v1_baseline',
            'train_acc': self.train_acc,
            'val_acc': self.val_acc
        }

        if return_probabilities:
            result['probabilities'] = {
                emotion: float(prob)
                for emotion, prob in zip(self.EMOTIONS, probs_np)
            }

        return result

    @torch.no_grad()
    def analyze_with_emotion_search(
        self,
        image: Union[str, Path, np.ndarray, Image.Image],
        beam_size: int = 5,
        return_all_results: bool = False
    ) -> Dict:
        """
        Analisa imagem gerando captions para TODAS as 9 emoções e escolhe a melhor.
        
        Este método implementa "emotion search": gera uma caption condicionada para cada
        emoção, classifica cada caption, e retorna a emoção com maior confiança.
        
        Args:
            image: Caminho da imagem, array numpy ou PIL.Image
            beam_size: Tamanho do beam search para geração de captions
            return_all_results: Se True, retorna resultados de todas as emoções
            
        Returns:
            Dict com:
                - best_emotion: emoção com maior confiança (str)
                - best_confidence: confiança da melhor emoção (float 0-1)
                - best_caption: caption que gerou a melhor predição (str)
                - best_probabilities: distribuição completa da melhor (Dict[str, float])
                - all_captions: captions geradas para cada emoção (Dict[str, str])
                - all_scores: score de cada emoção (Dict[str, float])
                - all_results: (opcional) resultados completos de todas emoções
        """
        print("🔍 Executando Emotion Search (testando todas as 9 emoções)...")
        
        # Load image once
        pil_image = self._load_image(image)
        
        # Gera captions para todas as 9 emoções
        all_captions = {}
        all_scores = {}
        all_results_data = {}
        
        for emotion in self.EMOTIONS:
            # Gera caption condicionada pela emoção
            caption = self.generate_caption(
                pil_image,
                emotion=emotion,
                beam_size=beam_size
            )
            all_captions[emotion] = caption
            
            # Classifica com essa caption
            result = self.analyze(
                pil_image,
                caption=caption,
                auto_caption=False,
                return_probabilities=True
            )
            
            # Score = probabilidade da própria emoção na predição
            score = result['probabilities'].get(emotion, 0.0)
            all_scores[emotion] = score
            all_results_data[emotion] = result
            
            print(f"   {emotion:>15}: score={score:.3f} | \"{caption}\"")
        
        # Encontra melhor emoção (maior score)
        best_emotion = max(all_scores.items(), key=lambda x: x[1])[0]
        best_result = all_results_data[best_emotion]
        
        print()
        print(f"✨ Melhor emoção encontrada: {best_emotion.upper()}")
        print(f"   Confiança: {all_scores[best_emotion]*100:.1f}%")
        print(f"   Caption: \"{all_captions[best_emotion]}\"")
        
        # Monta resultado
        result = {
            'best_emotion': best_emotion,
            'best_confidence': all_scores[best_emotion],
            'best_caption': all_captions[best_emotion],
            'best_probabilities': best_result['probabilities'],
            'all_captions': all_captions,
            'all_scores': all_scores,
        }
        
        if return_all_results:
            result['all_results'] = all_results_data
        
        return result

    @torch.no_grad()
    def analyze_batch(
        self,
        images: List[Union[str, Path, np.ndarray, Image.Image]],
        captions: Optional[List[str]] = None,
        auto_caption: bool = False
    ) -> List[Dict]:
        """
        Analisa múltiplas imagens em batch (mais eficiente).

        Args:
            images: Lista de imagens
            captions: Lista de captions (opcional)
            auto_caption: Gerar captions automaticamente se None

        Returns:
            Lista de resultados (um Dict por imagem)
        """
        if captions is None:
            if auto_caption:
                captions = [self.generate_caption(img) for img in images]
            else:
                captions = ["A painting."] * len(images)

        # Preprocess all
        img_tensors = torch.cat([
            self._preprocess_image(self._load_image(img))
            for img in images
        ])  # (B, 3, 224, 224)

        # Tokenize all (need to handle variable lengths)
        text_encodings = [self._tokenize_text(cap) for cap in captions]

        # Batch tokenization (simpler: process one by one for now)
        results = []
        for i, (img_path, caption) in enumerate(zip(images, captions)):
            result = self.analyze(
                img_path,
                caption=caption,
                auto_caption=False
            )
            results.append(result)

        return results

    def __repr__(self) -> str:
        return (
            f"PerceptoEmocional(model='v1_baseline', "
            f"device='{self.device}', "
            f"checkpoint='{self.model_path.name}')"
        )


if __name__ == "__main__":
    # Test
    print("🧪 Testing PerceptoEmocional...")

    # Initialize
    percepto = PerceptoEmocional()
    print(percepto)

    print("\n✅ Agente 2 - Percepto Emocional implementado com sucesso!")
