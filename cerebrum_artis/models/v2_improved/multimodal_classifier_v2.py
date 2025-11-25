"""
Multimodal Emotion Classifier V2 - IMPROVED VERSION

Improvements over v1:
1. ✅ Fine-tuning ResNet layer4 (unfrozen last block)
2. ✅ Better fusion strategy (ready for attention if needed)
3. ✅ Configurable partial freezing

Expected Performance: 72-75% accuracy (vs v1: 69.38%)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import RobertaModel, RobertaTokenizer
import torchvision.models as models


class MultimodalEmotionClassifierV2(nn.Module):
    """
    Versão melhorada do classificador multimodal.
    
    MELHORIAS sobre v1:
    - Fine-tuning parcial do ResNet (layer4 unfrozen)
    - Dropout adaptativo
    - Fusion MLP mais profunda
    
    Args:
        num_emotions (int): Número de classes (9 para ArtEmis)
        freeze_image_encoder (str): 'full', 'partial', 'none'
            - 'full': Congela tudo (v1 baseline)
            - 'partial': Descongela layer4 (RECOMENDADO)
            - 'none': Treina tudo (cuidado com overfitting)
        freeze_text_encoder (bool): Congelar RoBERTa
        dropout (float): Dropout rate
    """
    
    def __init__(self, 
                 num_emotions=9,
                 freeze_image_encoder='partial',  # MUDANÇA PRINCIPAL!
                 freeze_text_encoder=False,
                 dropout=0.3):
        super().__init__()
        
        self.num_emotions = num_emotions
        self.freeze_strategy = freeze_image_encoder
        
        # === IMAGE ENCODER: ResNet50 ===
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        self.image_encoder = nn.Sequential(*list(resnet.children())[:-1])
        
        # ESTRATÉGIA DE CONGELAMENTO
        if freeze_image_encoder == 'full':
            # v1 baseline: congela tudo
            for param in self.image_encoder.parameters():
                param.requires_grad = False
            print("🧊 ResNet: FULLY FROZEN (v1 mode)")
            
        elif freeze_image_encoder == 'partial':
            # v2: congela tudo MENOS layer4 (últimas camadas)
            for name, child in self.image_encoder.named_children():
                # ResNet structure: conv1, bn1, relu, maxpool, layer1, layer2, layer3, layer4, avgpool
                # Queremos descongelar layer4 (índice 7)
                if int(name) < 7:  # Indices 0-6 = camadas iniciais
                    for param in child.parameters():
                        param.requires_grad = False
                else:  # layer4 (index 7) + avgpool (index 8)
                    for param in child.parameters():
                        param.requires_grad = True
            
            frozen_params = sum(1 for p in self.image_encoder.parameters() if not p.requires_grad)
            trainable_params = sum(1 for p in self.image_encoder.parameters() if p.requires_grad)
            print(f"🔥 ResNet: PARTIAL FREEZE (layer4 unfrozen)")
            print(f"   Frozen: {frozen_params} params | Trainable: {trainable_params} params")
            
        else:  # 'none'
            print("⚡ ResNet: FULLY TRAINABLE (risco de overfit!)")
        
        self.image_feat_dim = 2048
        
        # === TEXT ENCODER: RoBERTa-base ===
        self.text_encoder = RobertaModel.from_pretrained('roberta-base')
        
        if freeze_text_encoder:
            for param in self.text_encoder.parameters():
                param.requires_grad = False
            print("🧊 RoBERTa: FROZEN")
        else:
            print("🔥 RoBERTa: TRAINABLE")
            
        self.text_feat_dim = 768  # RoBERTa hidden size
        
        # === FUSION MLP (melhorado!) ===
        fusion_input_dim = self.image_feat_dim + self.text_feat_dim  # 2048 + 768 = 2816
        
        self.fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            # Extra layer pra mais capacidade
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),  # Menos dropout na última
            
            nn.Linear(256, num_emotions)
        )
        
        # Initialize fusion weights
        self._init_fusion_weights()
    
    def _init_fusion_weights(self):
        """Xavier initialization para fusion MLP."""
        for module in self.fusion.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def forward(self, image, input_ids, attention_mask):
        """
        Forward pass.
        
        Args:
            image: (B, 3, 224, 224) - Imagens normalizadas
            input_ids: (B, seq_len) - IDs dos tokens
            attention_mask: (B, seq_len) - Máscara de atenção
        
        Returns:
            logits: (B, num_emotions) - Logits das emoções
        """
        # Image features
        img_features = self.image_encoder(image)  # (B, 2048, 1, 1)
        img_features = img_features.squeeze(-1).squeeze(-1)  # (B, 2048)
        
        # Text features (CLS token)
        text_output = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        text_features = text_output.last_hidden_state[:, 0, :]  # (B, 768) - [CLS]
        
        # Fusion
        combined = torch.cat([img_features, text_features], dim=1)  # (B, 2816)
        logits = self.fusion(combined)  # (B, num_emotions)
        
        return logits


def load_multimodal_classifier_v2(checkpoint_path, device='cuda', freeze_strategy='partial'):
    """
    Carrega modelo v2 de um checkpoint.
    
    Args:
        checkpoint_path: Caminho pro .pt
        device: 'cuda' ou 'cpu'
        freeze_strategy: 'full', 'partial', 'none'
    
    Returns:
        model: Modelo carregado
        metadata: Dict com info do checkpoint
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    model = MultimodalEmotionClassifierV2(
        num_emotions=checkpoint.get('num_emotions', 9),
        freeze_image_encoder=freeze_strategy,
        dropout=checkpoint.get('dropout', 0.3)
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    metadata = {
        'epoch': checkpoint.get('epoch', -1),
        'val_acc': checkpoint.get('val_acc', -1),
        'num_emotions': checkpoint.get('num_emotions', 9),
    }
    
    print(f"✅ Loaded v2 model from epoch {metadata['epoch']}")
    print(f"   Val accuracy: {metadata['val_acc']:.4f}")
    
    return model, metadata


# Compatibilidade: alias pro nome antigo
MultimodalEmotionClassifier = MultimodalEmotionClassifierV2
