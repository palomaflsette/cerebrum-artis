"""
SAT Model Loader - Carrega modelo Show, Attend and Tell (ResNet + LSTM)
"""
import torch
import pickle
import sys
from pathlib import Path

# Adiciona paths necessários para o SAT real
artemis_sat_path = Path(__file__).parent.parent.parent.parent / 'garbage' / 'old_artemis-v2' / 'neural_speaker' / 'sat'
sys.path.insert(0, str(artemis_sat_path))

from artemis.neural_models.show_attend_tell import describe_model
from artemis.neural_models.resnet_encoder import ResnetEncoder
from artemis.neural_models.attentive_decoder import AttentiveDecoder


class SimpleVocab:
    """Vocabulário simplificado compatível com SAT checkpoint."""
    
    def __init__(self, word2idx, idx2word):
        self.word2idx = word2idx
        self.idx2word = idx2word
        
        # Special tokens (padrão do SAT)
        self.pad = word2idx.get('<pad>', 0)
        self.sos = word2idx.get('<sos>', 1)  # Start of sentence
        self.eos = word2idx.get('<eos>', 2)  # End of sentence
        self.unk = word2idx.get('<unk>', 3)  # Unknown
    
    def __len__(self):
        return len(self.word2idx)
    
    def __repr__(self):
        return f"SimpleVocab(size={len(self)}, pad={self.pad}, sos={self.sos}, eos={self.eos})"


class SATModelLoader:
    """
    Carrega o modelo SAT (Show, Attend and Tell) treinado no ArtEmis.
    Usa ResNet encoder + LSTM decoder com attention.
    
    Attributes:
        model: SAT model (ResnetEncoder + AttentiveDecoder)
        vocab: Vocabulário de palavras
        device: 'cuda' ou 'cpu'
    """
    
    def __init__(
        self,
        checkpoint_path: str,
        vocab_path: str,
        use_emotion_labels: bool = True,
        device: str = 'cpu'
    ):
        """
        Inicializa e carrega o modelo SAT.
        
        Args:
            checkpoint_path: Caminho para checkpoint (.pt)
            vocab_path: Caminho para vocabulário (.pkl)
            use_emotion_labels: Se True, usa emotion grounding
            device: 'cuda' ou 'cpu'
        """
        self.device = device
        self.use_emotion_labels = use_emotion_labels
        
        # 1. Carrega vocabulário
        print(f"📚 Carregando vocabulário de {vocab_path}...")
        
        try:
            with open(vocab_path, 'rb') as f:
                self.vocab = pickle.load(f)
            print(f"  ✅ Vocabulário carregado: {len(self.vocab)} tokens")
        except Exception as e:
            print(f"  ⚠️  Erro ao carregar vocab: {e}")
            # Fallback: cria vocabulário mínimo
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            decoder_emb = checkpoint['model'].get('decoder.word_embedding.weight')
            if decoder_emb is not None:
                vocab_size = decoder_emb.shape[0]
                word2idx = {f'<pad>': 0, '<sos>': 1, '<eos>': 2, '<unk>': 3}
                for i in range(4, vocab_size):
                    word2idx[f'token_{i}'] = i
                idx2word = {v: k for k, v in word2idx.items()}
                self.vocab = SimpleVocab(word2idx, idx2word)
                print(f"  ✅ Vocabulário simplificado: {vocab_size} tokens")
        
        # 2. Carrega checkpoint para inspecionar arquitetura
        print(f"⚙️  Carregando checkpoint de {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # 3. Cria argumentos para describe_model (necessário para SAT)
    
    @torch.no_grad()
    def generate(
        self,
        image_features: torch.Tensor,
        emotion: str = None,
        max_len: int = 54,
        beam_size: int = 5,
        out_size: int = 1
    ) -> str:
        """
        Gera caption afetiva para a imagem.
        
        Args:
            image_features: Features visuais extraídas (1, num_regions, 2048)
            emotion: Nome da emoção (opcional, e.g., 'awe', 'sadness')
            max_len: Comprimento máximo da caption
            beam_size: Tamanho do beam search
            out_size: Número de captions a retornar
        
        Returns:
            Caption gerada como string
        """
        # Map emotion name to one-hot vector
        emotion_idx_map = {
            'amusement': 0, 'awe': 1, 'contentment': 2, 'excitement': 3,
            'anger': 4, 'disgust': 5, 'fear': 6, 'sadness': 7, 'something_else': 8
        }
        
        # Prepare emotion embedding
        if self.emotion_encoder is not None:
            if emotion is None:
                # Default: neutral/something_else
                emotion_onehot = torch.zeros(1, 9, device=self.device)
                emotion_onehot[0, 8] = 1.0  # something_else
            else:
                emotion_idx = emotion_idx_map.get(emotion.lower(), 8)
                emotion_onehot = torch.zeros(1, 9, device=self.device)
                emotion_onehot[0, emotion_idx] = 1.0
            
            # Encode emotion
            emotion_emb = self.emotion_encoder(emotion_onehot)  # (1, emotion_dim)
            
            # Concatenate with image features
            # image_features: (1, num_regions, 2048)
            # emotion_emb: (1, emotion_dim) → expand to (1, num_regions, emotion_dim)
            num_regions = image_features.size(1)
            emotion_emb_expanded = emotion_emb.unsqueeze(1).expand(1, num_regions, -1)
            visual_input = torch.cat([image_features, emotion_emb_expanded], dim=-1)
        else:
            visual_input = image_features
        
        # Move to device
        visual_input = visual_input.to(self.device)
        
        # Generate caption using beam search
        eos_idx = self.vocab.eos
        
        with torch.no_grad():
            outputs, _ = self.model.beam_search(
                visual_input,
                max_len=max_len,
                eos_idx=eos_idx,
                beam_size=beam_size,
                out_size=out_size
            )
        
        # Decode tokens to text
        caption_tokens = outputs[0].cpu().numpy()
        
        # Convert tokens to words
        words = []
        for token_id in caption_tokens:
            if token_id == eos_idx or token_id == self.vocab.sos:
                continue
            if token_id == self.vocab.pad:
                break
            word = self.vocab.idx2word.get(token_id, '<unk>')
            words.append(word)
        
        caption = ' '.join(words)
        return caption
    
    @torch.no_grad()
    def extract_emotion_logits(self, image_features, emotion_label):
        """
        Extrai logits de emoções usando o decoder.
        
        NOTA: O modelo M2 é um CAPTIONING model, não classificador.
        Para obter probabilidades de emoções, precisamos de outra abordagem.
        
        Args:
            image_features: Features visuais (B, num_regions, 2048)
            emotion_label: One-hot emotion (B, 9)
        
        Returns:
            Logits sobre vocabulário (não sobre emoções)
        """
        raise NotImplementedError(
            "O modelo M2 é um IMAGE CAPTIONING model, não classificador de emoções!\n"
            "Ele gera TEXTO descrevendo emoções, não prediz probabilidades.\n\n"
            "Para classificação de emoções, você precisa:\n"
            "1. Usar apenas o ENCODER + um classificador linear no topo, OU\n"
            "2. Treinar um modelo separado para classificação (ResNet50 + FC)"
        )
    
    def __repr__(self):
        return (
            f"SATModelLoader(\n"
            f"  vocab_size={len(self.vocab)},\n"
            f"  emotion_encoder={'Yes' if self.emotion_encoder else 'No'},\n"
            f"  device={self.device}\n"
            f")"
        )


# ============================================================================
# TESTE RÁPIDO
# ============================================================================

if __name__ == "__main__":
    import os
    
    # Paths
    artemis_root = Path(__file__).parent.parent.parent / 'artemis-v2'
    checkpoint_path = artemis_root / 'sat_logs' / 'sat_combined' / 'checkpoints' / 'best_model.pt'
    vocab_path = artemis_root / 'dataset' / 'combined' / 'train' / 'vocabulary.pkl'
    
    print("="*70)
    print("TESTE: Carregamento do modelo SAT")
    print("="*70)
    
    if not checkpoint_path.exists():
        print(f"❌ Checkpoint não encontrado: {checkpoint_path}")
        exit(1)
    
    if not vocab_path.exists():
        print(f"❌ Vocabulário não encontrado: {vocab_path}")
        exit(1)
    
    # Carrega modelo
    loader = SATModelLoader(
        checkpoint_path=str(checkpoint_path),
        vocab_path=str(vocab_path),
        use_emotion_labels=True,
        device='cpu'
    )
    
    print("\n" + "="*70)
    print(loader)
    print("="*70)
    print("\n✅ Modelo SAT carregado com sucesso!")
    print("\nNOTA: Este é um modelo de CAPTIONING, não classificação.")
    print("Para classificação de emoções, precisaremos de abordagem diferente.")
    print("="*70 + "\n")
