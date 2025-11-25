"""
MultimodalFuzzyClassifier - Classificador multimodal com fuzzy features

Combina:
- ResNet50 visual features (2048)
- RoBERTa text features (768)
- Fuzzy visual features (7): dominance, valence, arousal, color_temp, color_sat, brightness, complexity
Total: 2823 → MLP → 9 emotion classes
"""

import torch
import torch.nn as nn
from torchvision import models
from transformers import RobertaModel


class MultimodalFuzzyClassifier(nn.Module):
    """
    Combines:
    - ResNet50 visual features (2048)
    - RoBERTa text features (768)
    - Fuzzy visual features (7)
    Total: 2823 → MLP → 9 classes
    """
    
    def __init__(self, num_classes=9, dropout=0.3, freeze_resnet=True):
        super().__init__()
        
        # Vision: ResNet50
        resnet = models.resnet50(pretrained=True)
        self.visual_encoder = nn.Sequential(*list(resnet.children())[:-1])
        
        if freeze_resnet:
            for param in self.visual_encoder.parameters():
                param.requires_grad = False
        
        # Text: RoBERTa
        self.text_encoder = RobertaModel.from_pretrained('roberta-base')
        
        # Fusion MLP
        # 2048 (ResNet) + 768 (RoBERTa) + 7 (Fuzzy) = 2823
        self.fusion = nn.Sequential(
            nn.Linear(2048 + 768 + 7, 1024),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, image, input_ids, attention_mask, fuzzy_features):
        # Visual features: [B, 2048]
        visual_feats = self.visual_encoder(image)
        visual_feats = visual_feats.view(visual_feats.size(0), -1)
        
        # Text features: [B, 768]
        text_output = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        text_feats = text_output.last_hidden_state[:, 0, :]  # CLS token
        
        # Fuzzy features: [B, 7] (already provided)
        
        # Concatenate all features
        combined = torch.cat([visual_feats, text_feats, fuzzy_features], dim=1)
        
        # Classify
        logits = self.fusion(combined)
        return logits
