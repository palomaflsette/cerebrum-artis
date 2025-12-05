"""
Agente 2 - Percepto Emocional V3 (com Fuzzy Features)

Versão melhorada que integra:
- Agent 1 (Fuzzy Emotion) para features interpretáveis
- SAT (Show, Attend & Tell) para geração automática de captions
- Multimodal classification com ResNet50 + RoBERTa + Fuzzy Features
"""

import torch
import torch.nn.functional as F
from pathlib import Path
from typing import Union, Optional, Dict, List
from PIL import Image
import numpy as np
import sys

from cerebrum_artis.models.multimodal_fuzzy import MultimodalFuzzyClassifier

# Add fuzzy-brain to path
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[2] / 'fuzzy-brain'))

from fuzzy_brain.extractors.visual import VisualFeatureExtractor


class PerceptoEmocionalV3:
    """
    Agente 2 - Percepto Emocional V3 (com Fuzzy Features).
    
    Combina:
    - Visual encoder (ResNet50)
    - Text encoder (RoBERTa)
    - Fuzzy emotion features (Agent 1)
    - SAT caption generation
    
    Capabilities:
    - Classificação de emoção multimodal (imagem + texto)
    - Geração automática de captions emocionalmente condicionados
    - Emotion search (testa todas emoções e escolhe melhor)
    - Interpretabilidade via fuzzy features
    """
    
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
        Inicializa Percepto Emocional V3.
        
        Args:
            model_path: Caminho para checkpoint V3 (com fuzzy features)
            sat_checkpoint: Caminho para checkpoint do SAT
            device: 'cuda' ou 'cpu'
        """
        # Device setup
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        print(f"🧠 Inicializando Percepto Emocional V3 no device: {self.device}")

        # Load checkpoint
        if model_path is None:
            model_path = self._find_best_checkpoint()

        self.model_path = Path(model_path)
        print(f"📂 Carregando checkpoint V3: {self.model_path}")

        checkpoint = torch.load(self.model_path, map_location=self.device)
        
        # Initialize fuzzy feature extractor
        print(f"🔮 Inicializando Visual Feature Extractor...")
        self.visual_extractor = VisualFeatureExtractor()
        
        # Initialize model V3
        self.model = MultimodalFuzzyClassifier(
            num_classes=9,
            freeze_resnet=True,
            dropout=0.3
        )
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        # Emotion labels mapping
        self.emotion_labels = {
            'amusement': 0, 'awe': 1, 'contentment': 2, 'excitement': 3,
            'anger': 4, 'disgust': 5, 'fear': 6, 'sadness': 7, 'something else': 8
        }

        # Store metrics
        self.train_acc = checkpoint.get('train_acc', None)
        self.val_acc = checkpoint.get('val_acc', None)
        
        print(f"✅ Modelo V3 carregado! Epoch {checkpoint.get('epoch', 'N/A')} | "
              f"Train: {self.train_acc} | Val: {self.val_acc}")

        # Preprocessing
        from torchvision import transforms
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # Tokenizer
        from transformers import RobertaTokenizer
        self.tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

        # SAT model (lazy load)
        self.sat_model = None
        self.sat_checkpoint = sat_checkpoint

    def _find_best_checkpoint(self) -> str:
        """Busca o melhor checkpoint V3 disponível."""
        base_dir = Path("/data/paloma/deep-mind-checkpoints")
        
        # V3 com fuzzy features
        v3_dir = base_dir / "v2_fuzzy_features"
        if not v3_dir.exists():
            raise FileNotFoundError(
                f"Checkpoint V3 não encontrado em {v3_dir}\n"
                "Execute o treinamento V3 primeiro."
            )

        # Procura checkpoint_best.pt primeiro
        best_ckpt = v3_dir / "checkpoint_best.pt"
        if best_ckpt.exists():
            print(f"🔥 Usando V3 (best): {best_ckpt}")
            return str(best_ckpt)

        # Senão, pega o último epoch
        ckpts = sorted(v3_dir.glob("checkpoint_epoch*.pt"))
        if not ckpts:
            raise FileNotFoundError(f"Nenhum checkpoint encontrado em {v3_dir}")

        print(f"🔥 Usando V3 (last): {ckpts[-1]}")
        return str(ckpts[-1])

    def _load_sat_model(self):
        """Carrega modelo SAT para geração de captions."""
        print("📝 Carregando modelo SAT para geração de captions...")

        # Lazy import SAT loader
        fuzzy_brain_path = Path(__file__).parent.parent.parent / "fuzzy-brain" / "fuzzy_brain"
        if str(fuzzy_brain_path) not in sys.path:
            sys.path.insert(0, str(fuzzy_brain_path))
        
        from sat_loader_classic import SATModelLoader

        # Find SAT checkpoint
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
        """Preprocessa imagem para o encoder SAT."""
        from torchvision import transforms
        
        pil_image = self._load_image(image)
        
        sat_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        image_tensor = sat_transform(pil_image)
        return image_tensor.unsqueeze(0)

    def _preprocess_image(self, image: Image.Image) -> torch.Tensor:
        """Preprocessa imagem para o modelo."""
        img_tensor = self.transform(image)
        return img_tensor.unsqueeze(0).to(self.device)

    def _tokenize_text(self, text: str, max_length: int = 128) -> Dict[str, torch.Tensor]:
        """Tokeniza texto usando RoBERTa."""
        encoding = self.tokenizer(
            text,
            max_length=max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].to(self.device),
            'attention_mask': encoding['attention_mask'].to(self.device)
        }

    @torch.no_grad()
    def generate_caption(
        self,
        image: Union[str, Path, np.ndarray, Image.Image],
        emotion: Optional[str] = None,
        max_len: int = 54,
        beam_size: int = 5
    ) -> str:
        """
        Gera caption afetiva usando modelo SAT.

        Args:
            image: Caminho da imagem ou array numpy
            emotion: Emoção opcional para condicionar geração
            max_len: Comprimento máximo da caption
            beam_size: Tamanho do beam search

        Returns:
            Caption gerada automaticamente
        """
        # Lazy load SAT model
        if self.sat_model is None:
            self._load_sat_model()
        
        # Preprocessa imagem para SAT
        image_tensor = self._preprocess_image_for_sat(image)
        
        # Convert emotion string to one-hot tensor if provided
        emotion_onehot = None
        if emotion is not None:
            emotion_idx = self.emotion_labels.get(emotion.lower(), None)
            if emotion_idx is not None:
                emotion_onehot = torch.zeros(1, 9).to(self.device)
                emotion_onehot[0, emotion_idx] = 1.0
        
        # Generate caption
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
        auto_caption: bool = False,
        return_probabilities: bool = False
    ) -> Dict:
        """
        Analisa imagem + caption e retorna emoção dominante.

        Args:
            image: Caminho da imagem, array numpy ou PIL.Image
            caption: Caption textual (opcional)
            auto_caption: Gerar caption automaticamente se None
            return_probabilities: Retornar distribuição completa de probabilidades

        Returns:
            Dict com:
                - emotion: emoção dominante (str)
                - confidence: confiança da predição (float 0-1)
                - probabilities: distribuição completa (Dict[str, float]) [opcional]
                - caption: caption usado (str)
                - caption_source: 'user', 'default' ou 'generated'
                - fuzzy_features: features fuzzy do Agent 1 (Dict)
        """
        # Load image
        pil_image = self._load_image(image)

        # Get or generate caption
        caption_source = 'default'
        if caption is None and auto_caption:
            caption = self.generate_caption(pil_image)
            caption_source = 'generated'
        elif caption is None:
            caption = "A painting."
            caption_source = 'default'
        else:
            caption_source = 'user'

        # Extract fuzzy visual features using VisualFeatureExtractor
        # (extractor requires file path, so save temp image)
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
            pil_image.save(tmp.name)
            tmp_path = tmp.name
        
        try:
            features = self.visual_extractor.extract_all(tmp_path)
        finally:
            import os
            os.unlink(tmp_path)
        
        # Build fuzzy feature vector in the SAME ORDER as training (precompute_fuzzy_features.py)
        fuzzy_features = torch.tensor([
            features['brightness'],
            features['color_temperature'],
            features['saturation'],
            features['color_harmony'],
            features['complexity'],
            features['symmetry'],
            features['texture_roughness']
        ], dtype=torch.float32).unsqueeze(0).to(self.device)

        # Preprocess image and text
        img_tensor = self._preprocess_image(pil_image)
        text_encoding = self._tokenize_text(caption)

        # Forward pass with fuzzy features
        logits = self.model(
            img_tensor,
            text_encoding['input_ids'],
            text_encoding['attention_mask'],
            fuzzy_features
        )

        # Get probabilities
        probs = F.softmax(logits, dim=-1)[0]
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
            'fuzzy_features': features,  # Raw feature dict from VisualFeatureExtractor
            'model': 'v3_fuzzy',
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
        
        Args:
            image: Caminho da imagem, array numpy ou PIL.Image
            beam_size: Tamanho do beam search
            return_all_results: Retornar resultados de todas emoções
            
        Returns:
            Dict com melhor emoção, caption, scores de todas emoções, etc.
        """
        print("🔍 Executando Emotion Search V3 (testando todas as 9 emoções)...")
        
        pil_image = self._load_image(image)
        
        all_captions = {}
        all_scores = {}
        all_results_data = {}
        
        for emotion in self.EMOTIONS:
            # Gera caption condicionada
            caption = self.generate_caption(pil_image, emotion=emotion, beam_size=beam_size)
            all_captions[emotion] = caption
            
            # Classifica
            result = self.analyze(
                pil_image,
                caption=caption,
                auto_caption=False,
                return_probabilities=True
            )
            
            # Score = probabilidade da própria emoção
            score = result['probabilities'].get(emotion, 0.0)
            all_scores[emotion] = score
            all_results_data[emotion] = result
            
            print(f"   {emotion:>15}: score={score:.3f} | \"{caption}\"")
        
        # Melhor emoção
        best_emotion = max(all_scores.items(), key=lambda x: x[1])[0]
        best_result = all_results_data[best_emotion]
        
        print()
        print(f"✨ Melhor emoção encontrada: {best_emotion.upper()}")
        print(f"   Confiança: {all_scores[best_emotion]*100:.1f}%")
        print(f"   Caption: \"{all_captions[best_emotion]}\"")
        
        result = {
            'best_emotion': best_emotion,
            'best_confidence': all_scores[best_emotion],
            'best_caption': all_captions[best_emotion],
            'best_probabilities': best_result['probabilities'],
            'best_fuzzy_features': best_result['fuzzy_features'],
            'all_captions': all_captions,
            'all_scores': all_scores,
        }
        
        if return_all_results:
            result['all_results'] = all_results_data
        
        return result

    def __repr__(self) -> str:
        return (
            f"PerceptoEmocionalV3(model='v3_fuzzy', "
            f"device='{self.device}', "
            f"checkpoint='{self.model_path.name}')"
        )
