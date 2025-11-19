"""
SAT Model Loader - Carrega modelo M2 Transformer treinado
"""
import torch
import pickle
import sys
from pathlib import Path

# Adiciona paths necessários
artemis_path = Path(__file__).parent.parent.parent / 'artemis' / 'neural_speaker' / 'm2'
sys.path.insert(0, str(artemis_path))

from models.transformer import (
    MemoryAugmentedEncoder, 
    MeshedDecoder,
    ScaledDotProductAttentionMemory, 
    Transformer
)


class SATModelLoader:
    """
    Carrega o modelo SAT (M2 Transformer) treinado no ArtEmis.
    
    Attributes:
        model: Transformer model (encoder + decoder)
        vocab: Vocabulário de palavras
        emotion_encoder: Encoder de emoções (se usado)
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
            use_emotion_labels: Se True, carrega emotion encoder
            device: 'cuda' ou 'cpu'
        """
        self.device = device
        self.use_emotion_labels = use_emotion_labels
        
        # 1. Carrega vocabulário
        print(f"📚 Carregando vocabulário de {vocab_path}...")
        with open(vocab_path, 'rb') as f:
            self.vocab = pickle.load(f)
        
        vocab_size = len(self.vocab)
        pad_idx = self.vocab.pad
        bos_idx = self.vocab.sos  # SOS = Start Of Sequence = BOS
        
        print(f"  ✅ Vocabulário: {vocab_size} tokens")
        
        # 2. Constrói modelo (mesma arquitetura do treinamento)
        print("🧠 Construindo modelo M2 Transformer...")
        
        emotion_dim = 0
        self.emotion_encoder = None
        
        if use_emotion_labels:
            emotion_dim = 10
            self.emotion_encoder = torch.nn.Sequential(
                torch.nn.Linear(9, emotion_dim)
            )
        
        # Encoder: MemoryAugmentedEncoder (M=40 memory slots)
        encoder = MemoryAugmentedEncoder(
            N=3,  # 3 layers
            padding_idx=0,
            attention_module=ScaledDotProductAttentionMemory,
            attention_module_kwargs={'m': 40},  # 40 memory slots
            d_in=2048 + emotion_dim  # ResNet50 features + emotion
        )
        
        # Decoder: MeshedDecoder
        decoder = MeshedDecoder(
            vocab_size=vocab_size,
            max_len=54,
            N_dec=3,  # 3 decoder layers
            padding_idx=pad_idx
        )
        
        # Transformer completo
        self.model = Transformer(bos_idx, encoder, decoder)
        
        # 3. Carrega pesos do checkpoint
        print(f"⚙️  Carregando checkpoint de {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Carrega state_dict
        self.model.load_state_dict(checkpoint['model'])
        
        if use_emotion_labels and 'emotion_encoder' in checkpoint:
            self.emotion_encoder.load_state_dict(checkpoint['emotion_encoder'])
        
        # Move para device
        self.model.to(device)
        if self.emotion_encoder is not None:
            self.emotion_encoder.to(device)
        
        self.model.eval()
        if self.emotion_encoder is not None:
            self.emotion_encoder.eval()
        
        print(f"  ✅ Modelo carregado! (epoch {checkpoint.get('epoch', '?')})")
        print(f"  📊 Best CIDEr: {checkpoint.get('best_cider', 'N/A')}")
    
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
    artemis_root = Path(__file__).parent.parent.parent / 'artemis'
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
