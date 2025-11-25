"""
Multimodal Emotion Classifier: Image + Text → Emotion Distribution

Combina features visuais (ResNet50) com features textuais (RoBERTa) para
classificar emoções em arte, seguindo a abordagem do ArtEmis dataset.

Architecture:
    IMAGE → ResNet50 (frozen) → features_img (2048-dim)
    TEXT  → RoBERTa-base → features_text (768-dim)
           ↓
    [Fusion MLP: 2816 → 1024 → 512 → 9]
           ↓
    Softmax(9 emotions)

Expected Performance: 75-82% accuracy on ArtEmis validation set
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import RobertaModel, RobertaTokenizer
import torchvision.models as models


class MultimodalEmotionClassifier(nn.Module):
    """
    Classificador multimodal de emoções para arte.
    
    Args:
        num_emotions (int): Número de classes de emoção (padrão: 9 do ArtEmis)
        freeze_image_encoder (bool): Congelar pesos do ResNet50
        freeze_text_encoder (bool): Congelar pesos do RoBERTa
        dropout (float): Taxa de dropout nas camadas de fusão
    """
    
    def __init__(self, 
                 num_emotions=9,
                 freeze_image_encoder=True,
                 freeze_text_encoder=False,  # Fine-tune RoBERTa é melhor
                 dropout=0.3):
        super().__init__()
        
        self.num_emotions = num_emotions
        
        # === IMAGE ENCODER: ResNet50 ===
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        # Remove FC layer, keep features (B, 2048, 7, 7) → (B, 2048) após pool
        self.image_encoder = nn.Sequential(*list(resnet.children())[:-1])
        
        if freeze_image_encoder:
            for param in self.image_encoder.parameters():
                param.requires_grad = False
        
        self.img_feature_dim = 2048
        
        # === TEXT ENCODER: RoBERTa-base ===
        self.text_encoder = RobertaModel.from_pretrained('roberta-base')
        
        if freeze_text_encoder:
            for param in self.text_encoder.parameters():
                param.requires_grad = False
        
        self.text_feature_dim = 768  # RoBERTa hidden size
        
        # === FUSION MLP ===
        fusion_input_dim = self.img_feature_dim + self.text_feature_dim  # 2816

        self.fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, 1024),  # 0
            nn.BatchNorm1d(1024),                # 1
            nn.ReLU(),                           # 2
            nn.Dropout(dropout),                 # 3

            nn.Linear(1024, 512),                # 4
            nn.BatchNorm1d(512),                 # 5
            nn.ReLU(),                           # 6
            nn.Dropout(dropout),                 # 7

            nn.Linear(512, 256),                 # 8
            nn.BatchNorm1d(256),                 # 9
            nn.ReLU(),                           # 10
            nn.Dropout(dropout),                 # 11

            nn.Linear(256, num_emotions)        # 12
        )
        
        # Initialize fusion layers
        self._init_fusion_weights()
    
    def _init_fusion_weights(self):
        """Xavier initialization para MLP fusion"""
        for module in self.fusion.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, image, input_ids, attention_mask):
        """
        Forward pass.
        
        Args:
            image: Tensor (B, 3, H, W) - imagens normalizadas
            input_ids: Tensor (B, seq_len) - tokens do texto
            attention_mask: Tensor (B, seq_len) - máscara de atenção
        
        Returns:
            logits: Tensor (B, num_emotions) - logits não normalizados
        """
        # Extract image features
        img_features = self.image_encoder(image)  # (B, 2048, 1, 1)
        img_features = img_features.flatten(1)     # (B, 2048)
        
        # Extract text features (use [CLS] token)
        text_output = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        text_features = text_output.last_hidden_state[:, 0, :]  # (B, 768) [CLS]
        
        # Fusion
        combined = torch.cat([img_features, text_features], dim=1)  # (B, 2816)
        logits = self.fusion(combined)  # (B, 9)
        
        return logits
    
    def predict_proba(self, image, input_ids, attention_mask):
        """
        Predict emotion probabilities.
        
        Returns:
            probs: Tensor (B, num_emotions) - probabilidades [0, 1]
        """
        logits = self.forward(image, input_ids, attention_mask)
        probs = F.softmax(logits, dim=-1)
        return probs
    
    def predict(self, image, input_ids, attention_mask):
        """
        Predict emotion class.
        
        Returns:
            predictions: Tensor (B,) - índices da classe predita
        """
        logits = self.forward(image, input_ids, attention_mask)
        predictions = torch.argmax(logits, dim=-1)
        return predictions


class ImageOnlyEmotionClassifier(nn.Module):
    """
    Classificador simplificado: apenas imagem → emoção.
    Útil como baseline ou quando não há texto disponível.
    """
    
    def __init__(self, num_emotions=9, freeze_encoder=True, dropout=0.3):
        super().__init__()
        
        # ResNet50
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        self.encoder = nn.Sequential(*list(resnet.children())[:-1])
        
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.BatchNorm1d(512),
            nn.Linear(512, num_emotions)
        )
    
    def forward(self, image):
        features = self.encoder(image).flatten(1)
        logits = self.classifier(features)
        return logits
    
    def predict_proba(self, image):
        logits = self.forward(image)
        return F.softmax(logits, dim=-1)


def load_multimodal_classifier(checkpoint_path, device='cpu'):
    """
    Carrega modelo treinado.
    
    Args:
        checkpoint_path: Caminho para .pt/.pth
        device: 'cpu' ou 'cuda'
    
    Returns:
        model: MultimodalEmotionClassifier carregado
        tokenizer: RobertaTokenizer
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Reconstruir modelo
    model = MultimodalEmotionClassifier(
        num_emotions=checkpoint.get('num_emotions', 9),
        freeze_image_encoder=True,
        freeze_text_encoder=False,
        dropout=checkpoint.get('dropout', 0.3)
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    # Tokenizer
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    
    return model, tokenizer


if __name__ == "__main__":
    # Test
    print("🧪 Testing MultimodalEmotionClassifier...")
    
    model = MultimodalEmotionClassifier()
    print(f"✅ Model created: {sum(p.numel() for p in model.parameters())/1e6:.2f}M params")
    
    # Dummy inputs
    batch_size = 4
    image = torch.randn(batch_size, 3, 224, 224)
    input_ids = torch.randint(0, 50265, (batch_size, 32))  # RoBERTa vocab
    attention_mask = torch.ones(batch_size, 32)
    
    # Forward
    logits = model(image, input_ids, attention_mask)
    probs = model.predict_proba(image, input_ids, attention_mask)
    preds = model.predict(image, input_ids, attention_mask)
    
    print(f"✅ Logits shape: {logits.shape}")
    print(f"✅ Probs shape: {probs.shape}, sum={probs[0].sum():.4f}")
    print(f"✅ Predictions shape: {preds.shape}")
    print(f"✅ Sample prediction: emotion {preds[0].item()} (prob={probs[0, preds[0]].item():.4f})")
