#!/usr/bin/env python3
"""
Deep-Mind V3: Intelligent Fuzzy-Neural Gating
==============================================

Fuzzy REALMENTE trabalha aqui!
- Executa 18 regras fuzzy
- Calcula confiança do fuzzy
- Adapta peso dinamicamente

Quando fuzzy está CERTO (alta confiança) → mais peso
Quando fuzzy está INCERTO (baixa confiança) → menos peso
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from transformers import RobertaTokenizer, RobertaModel
from PIL import Image
import pandas as pd
from tqdm import tqdm
import numpy as np
from datetime import datetime
from pathlib import Path
import pickle

# Fix para DecompressionBombWarning - permite imagens grandes
Image.MAX_IMAGE_PIXELS = None

# Fuzzy system
sys.path.insert(0, '/home/paloma/cerebrum-artis/fuzzy-brain')
from fuzzy_brain.fuzzy.system import FuzzyInferenceSystem

# ============================================================================
# CONSTANTS
# ============================================================================

EMOTIONS = [
    'amusement', 'awe', 'contentment', 'excitement', 
    'anger', 'disgust', 'fear', 'sadness', 'something else'
]

EMOTION_TO_IDX = {emotion: idx for idx, emotion in enumerate(EMOTIONS)}
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ============================================================================
# DATASET with CACHED Fuzzy Features
# ============================================================================

class ArtEmisFuzzyGatingDataset(Dataset):
    """Dataset that loads cached fuzzy features"""
    
    def __init__(self, csv_path, image_dir, fuzzy_cache, split='train', 
                 tokenizer=None, transform=None, max_length=128):
        
        self.image_dir = Path(image_dir)
        self.tokenizer = tokenizer
        self.transform = transform
        self.max_length = max_length
        self.split = split
        self.fuzzy_cache = fuzzy_cache
        
        # Load CSV
        print(f"📂 Loading {split} split...")
        df = pd.read_csv(csv_path)
        self.data = df[df['split'] == split].reset_index(drop=True)
        print(f"✅ {len(self.data)} examples")
        
        # Index images
        print(f"🔍 Indexing images...")
        self.image_paths = {}
        for img_file in self.image_dir.rglob('*.jpg'):
            self.image_paths[img_file.stem] = img_file
        print(f"✅ {len(self.image_paths)} images")
        
        # Filter valid
        print(f"🔍 Filtering...")
        valid_indices = []
        for idx in tqdm(range(len(self.data)), desc="Filtering"):
            painting = self.data.loc[idx, 'painting']
            if painting in self.image_paths and painting in self.fuzzy_cache:
                valid_indices.append(idx)
        
        self.data = self.data.iloc[valid_indices].reset_index(drop=True)
        print(f"✅ {len(self.data)} valid")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        painting = row['painting']
        
        # Image
        img_path = self.image_paths[painting]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        
        # Cached fuzzy features (7 values)
        fuzzy_features = torch.from_numpy(self.fuzzy_cache[painting])
        
        # Text
        utterance = row['utterance']
        tokens = self.tokenizer(
            utterance,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Label
        label = EMOTION_TO_IDX[row['emotion']]
        
        return {
            'image': image,
            'input_ids': tokens['input_ids'].squeeze(0),
            'attention_mask': tokens['attention_mask'].squeeze(0),
            'fuzzy_features': fuzzy_features,
            'label': torch.tensor(label, dtype=torch.long)
        }


# ============================================================================
# MODEL v4: Neural + Fuzzy with Intelligent Gating
# ============================================================================

class FuzzyGatingClassifier(nn.Module):
    """
    Neural network for multimodal classification
    + Fuzzy system with adaptive gating
    """
    
    def __init__(self, num_classes=9, dropout=0.3, freeze_resnet=True):
        super().__init__()
        
        # Vision: ResNet50
        resnet = models.resnet50(weights='IMAGENET1K_V1')
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
        
        # Fuzzy system (initialized externally)
        self.fuzzy_system = None
    
    def forward(self, image, input_ids, attention_mask, fuzzy_features=None):
        """
        Forward pass - returns neural logits
        Fuzzy inference happens OUTSIDE during training/eval
        """
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


# ============================================================================
# FUZZY INFERENCE (batch processing)
# ============================================================================

def batch_fuzzy_inference(fuzzy_system, fuzzy_features_batch):
    """
    Run fuzzy inference on a batch of feature vectors
    
    Args:
        fuzzy_system: FuzzyInferenceSystem instance
        fuzzy_features_batch: Tensor [B, 7]
    
    Returns:
        fuzzy_probs: Tensor [B, 9] - fuzzy emotion probabilities
    """
    batch_size = fuzzy_features_batch.size(0)
    device = fuzzy_features_batch.device
    
    fuzzy_probs_list = []
    
    for i in range(batch_size):
        # Convert to dict for fuzzy system
        features_dict = {
            'brightness': fuzzy_features_batch[i, 0].item(),
            'color_temperature': fuzzy_features_batch[i, 1].item(),
            'saturation': fuzzy_features_batch[i, 2].item(),
            'color_harmony': fuzzy_features_batch[i, 3].item(),
            'complexity': fuzzy_features_batch[i, 4].item(),
            'symmetry': fuzzy_features_batch[i, 5].item(),
            'texture_roughness': fuzzy_features_batch[i, 6].item()
        }
        
        # Run fuzzy inference (18 rules)
        fuzzy_dist = fuzzy_system.infer(features_dict)
        
        # Convert to tensor
        fuzzy_prob = torch.tensor(
            [fuzzy_dist.get(e, 0.0) for e in EMOTIONS],
            device=device,
            dtype=torch.float32
        )
        
        fuzzy_probs_list.append(fuzzy_prob)
    
    fuzzy_probs = torch.stack(fuzzy_probs_list)  # [B, 9]
    
    return fuzzy_probs


def adaptive_fusion(neural_logits, fuzzy_probs, 
                    base_alpha=0.85, min_alpha=0.6, max_alpha=0.95):
    """
    Adaptive fusion based on AGREEMENT between neural and fuzzy
    
    When neural and fuzzy AGREE → give fuzzy more weight (reinforcement)
    When they DISAGREE → trust neural more (it sees text context)
    
    Args:
        neural_logits: [B, 9] - neural network outputs
        fuzzy_probs: [B, 9] - fuzzy probabilities
        base_alpha: default neural weight (0.85 = 85% neural)
        min_alpha: minimum neural weight when they agree strongly
        max_alpha: maximum neural weight when they disagree
    
    Returns:
        final_logits: [B, 9] - fused predictions
        agreement: [B] - cosine similarity (0-1)
    """
    # Convert neural logits to probabilities
    neural_probs = torch.softmax(neural_logits, dim=1)
    
    # Compute agreement using cosine similarity
    # High similarity → both models agree on emotion distribution
    # Low similarity → models disagree
    agreement = torch.nn.functional.cosine_similarity(
        neural_probs, fuzzy_probs, dim=1
    )
    # Normalize to [0, 1] (cosine can be [-1, 1])
    agreement = (agreement + 1) / 2
    
    # Adaptive alpha based on agreement
    # High agreement → lower alpha (give fuzzy more weight, they reinforce each other)
    # Low agreement → higher alpha (trust neural, it has text context)
    adaptive_alpha = max_alpha - (max_alpha - min_alpha) * agreement
    adaptive_alpha = adaptive_alpha.unsqueeze(1)  # [B, 1]
    
    # Weighted fusion
    final_probs = adaptive_alpha * neural_probs + (1 - adaptive_alpha) * fuzzy_probs
    
    # Convert back to logits for loss calculation
    final_logits = torch.log(final_probs + 1e-8)
    
    return final_logits, agreement


# ============================================================================
# TRAINING
# ============================================================================

def train_epoch(model, fuzzy_system, dataloader, criterion, optimizer, device):
    model.train()
    
    total_loss = 0
    correct = 0
    total = 0
    
    avg_agreement = 0
    
    pbar = tqdm(dataloader, desc="Training")
    for batch in pbar:
        images = batch['image'].to(device)
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        fuzzy_features = batch['fuzzy_features'].to(device)
        labels = batch['label'].to(device)
        
        optimizer.zero_grad()
        
        # Neural forward
        neural_logits = model(images, input_ids, attention_mask)
        
        # Fuzzy inference
        fuzzy_probs = batch_fuzzy_inference(fuzzy_system, fuzzy_features)
        
        # Adaptive fusion based on agreement
        final_logits, agreement = adaptive_fusion(neural_logits, fuzzy_probs)
        
        # Loss on fused output
        loss = criterion(final_logits, labels)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        _, predicted = final_logits.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        avg_agreement += agreement.mean().item()
        
        pbar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'acc': f"{100.*correct/total:.2f}%",
            'agree': f"{agreement.mean().item():.3f}"
        })
    
    n_batches = len(dataloader)
    return (total_loss / n_batches, 
            100. * correct / total,
            avg_agreement / n_batches)


def validate(model, fuzzy_system, dataloader, criterion, device):
    model.eval()
    
    total_loss = 0
    correct = 0
    total = 0
    
    avg_agreement = 0
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Validation")
        for batch in pbar:
            images = batch['image'].to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            fuzzy_features = batch['fuzzy_features'].to(device)
            labels = batch['label'].to(device)
            
            # Neural
            neural_logits = model(images, input_ids, attention_mask)
            
            # Fuzzy
            fuzzy_probs = batch_fuzzy_inference(fuzzy_system, fuzzy_features)
            
            # Fusion
            final_logits, agreement = adaptive_fusion(neural_logits, fuzzy_probs)
            
            loss = criterion(final_logits, labels)
            
            total_loss += loss.item()
            
            _, predicted = final_logits.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            avg_agreement += agreement.mean().item()
            
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'acc': f"{100.*correct/total:.2f}%",
                'agree': f"{agreement.mean().item():.3f}"
            })
    
    n_batches = len(dataloader)
    return (total_loss / n_batches, 
            100. * correct / total,
            avg_agreement / n_batches)


# ============================================================================
# MAIN
# ============================================================================

def main():
    # Paths
    csv_path = '/home/paloma/cerebrum-artis/artemis-v2/dataset/official_data/combined_artemis_with_splits.csv'
    image_dir = '/data/paloma/data/paintings/wikiart'
    fuzzy_cache_file = '/data/paloma/fuzzy_features_cache.pkl'
    checkpoint_dir = Path('/data/paloma/deep-mind-checkpoints/v3_adaptive_gating')
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Hyperparameters
    batch_size = 32  # Adjusted for CUDA memory constraints with RoBERTa
    num_epochs = 20  # Full training with early stopping
    learning_rate = 2e-5
    num_workers = 8
    early_stop_patience = 5  # Stop if no improvement for 5 epochs
    
    print("="*80)
    print("🚀 DEEP-MIND v4: Intelligent Fuzzy-Neural Gating")
    print("="*80)
    print(f"📦 Batch size: {batch_size}")
    print(f"📚 Epochs: {num_epochs}")
    print(f"🎯 LR: {learning_rate}")
    print(f"🖥️  Device: {device}")
    print()
    
    # Load fuzzy cache
    print(f"🧠 Loading fuzzy cache...")
    with open(fuzzy_cache_file, 'rb') as f:
        fuzzy_cache = pickle.load(f)
    print(f"✅ {len(fuzzy_cache)} cached features")
    
    # Initialize fuzzy system
    print("🧠 Initializing fuzzy inference system...")
    fuzzy_system = FuzzyInferenceSystem()
    print(f"✅ Fuzzy system ready")
    print()
    
    # Transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Tokenizer
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    
    # Datasets
    print("📂 Creating datasets...")
    train_dataset = ArtEmisFuzzyGatingDataset(
        csv_path, image_dir, fuzzy_cache, split='train',
        tokenizer=tokenizer, transform=transform
    )
    
    val_dataset = ArtEmisFuzzyGatingDataset(
        csv_path, image_dir, fuzzy_cache, split='val',
        tokenizer=tokenizer, transform=transform
    )
    
    # DataLoaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size,
        shuffle=True, num_workers=num_workers, pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size,
        shuffle=False, num_workers=num_workers, pin_memory=True
    )
    
    print(f"✅ Train: {len(train_dataset)} | {len(train_loader)} batches")
    print(f"✅ Val: {len(val_dataset)} | {len(val_loader)} batches")
    print()
    
    # Model
    print("🤖 Creating model...")
    model = FuzzyGatingClassifier(num_classes=9, dropout=0.3).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"📊 Total: {total_params:,} | Trainable: {trainable_params:,}")
    print()
    
    # Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=learning_rate,
        weight_decay=0.01
    )
    
    # Resume from checkpoint if exists
    start_epoch = 1
    best_val_acc = 0.0
    epochs_no_improve = 0
    
    checkpoint_path = checkpoint_dir / 'checkpoint_best.pt'
    if checkpoint_path.exists():
        print(f"🔄 Resuming from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1  # Start from NEXT epoch
        best_val_acc = checkpoint['val_acc']  # Use val_acc from checkpoint as best
        print(f"✅ Resumed from epoch {checkpoint['epoch']}")
        print(f"✅ Best val acc so far: {best_val_acc:.2f}%")
        print()

    for epoch in range(start_epoch, num_epochs + 1):
        print(f"\n{'='*80}")
        print(f"📅 Epoch {epoch}/{num_epochs}")
        print(f"{'='*80}")

        train_loss, train_acc, train_agreement = train_epoch(
            model, fuzzy_system, train_loader, criterion, optimizer, device
        )

        val_loss, val_acc, val_agreement = validate(
            model, fuzzy_system, val_loader, criterion, device
        )

        print(f"\n📊 Epoch {epoch}:")
        print(f"  Train - Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%, Agreement: {train_agreement:.3f}")
        print(f"  Val   - Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%, Agreement: {val_agreement:.3f}")

        # Save checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'best_val_acc': best_val_acc,
        }

        # Save last checkpoint (cleanup old ones to save space)
        torch.save(checkpoint, checkpoint_dir / f'checkpoint_epoch{epoch}_last.pt')
        
        # Delete checkpoints older than 2 epochs
        if epoch > 2:
            old_checkpoint = checkpoint_dir / f'checkpoint_epoch{epoch-2}_last.pt'
            if old_checkpoint.exists():
                old_checkpoint.unlink()

        # Early stopping check
        is_best = val_acc > best_val_acc
        if is_best:
            best_val_acc = val_acc
            epochs_no_improve = 0
            torch.save(checkpoint, checkpoint_dir / 'checkpoint_best.pt')
            print(f"  🎯 NEW BEST! Val Acc: {val_acc:.2f}%")
        else:
            epochs_no_improve += 1
            print(f"  ⏳ No improvement for {epochs_no_improve}/{early_stop_patience} epochs")

        # Early stopping
        if epochs_no_improve >= early_stop_patience:
            print(f"\n🛑 Early stopping triggered! No improvement for {early_stop_patience} epochs.")
            print(f"   Best val acc: {best_val_acc:.2f}%")
            break
    
    print("\n" + "="*80)
    print(f"🎉 Training done! Best: {best_val_acc:.2f}%")
    print("="*80)


if __name__ == '__main__':
    main()
