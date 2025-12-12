"""
Model Definitions for V2 and V3
================================
Extracted from training scripts for ensemble usage
"""

import torch
import torch.nn as nn
from torchvision import models
from transformers import RobertaModel


class MultimodalFuzzyClassifier(nn.Module):
    """V2: Simple concatenation of all features"""
    
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
        
        # Fusion: Concatenate visual (2048) + text (768) + fuzzy (7) = 2823
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
        """Forward pass with simple concatenation"""
        # Visual: [B, 2048]
        visual_feats = self.visual_encoder(image).view(image.size(0), -1)
        
        # Text: [B, 768]
        text_output = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        text_feats = text_output.last_hidden_state[:, 0, :]
        
        # Concatenate all features: visual + text + fuzzy
        combined = torch.cat([visual_feats, text_feats, fuzzy_features], dim=1)  # [B, 2823]
        
        # MLP classification
        logits = self.fusion(combined)
        
        return logits


class FuzzyGatingClassifier(nn.Module):
    """V3: Neural + Fuzzy with EXTERNAL gating"""
    
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
        
        # Neural classifier
        self.classifier = nn.Sequential(
            nn.Linear(2048 + 768, 1024),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, image, input_ids, attention_mask, fuzzy_features=None):
        """Forward pass - returns neural logits (fuzzy inference happens outside)"""
        # Visual: [B, 2048]
        visual_feats = self.visual_encoder(image).view(image.size(0), -1)
        
        # Text: [B, 768]
        text_output = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        text_feats = text_output.last_hidden_state[:, 0, :]
        
        # Concat & classify
        combined = torch.cat([visual_feats, text_feats], dim=1)
        logits = self.classifier(combined)
        
        return logits
