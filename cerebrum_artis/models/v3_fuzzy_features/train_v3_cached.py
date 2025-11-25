#!/usr/bin/env python3
"""
Deep-Mind v3: Multimodal + CACHED Fuzzy Features
=================================================
Usa features fuzzy pré-computadas = RÁPIDO!
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

class ArtEmisCachedFuzzyDataset(Dataset):
    """ArtEmis dataset loading PRE-COMPUTED fuzzy features"""
    
    def __init__(self, csv_path, image_dir, fuzzy_cache, split='train', 
                 tokenizer=None, transform=None, max_length=128):
        
        self.image_dir = Path(image_dir)
        self.tokenizer = tokenizer
        self.transform = transform
        self.max_length = max_length
        self.split = split
        self.fuzzy_cache = fuzzy_cache  # Dict[painting_name, np.array(7)]
        
        # Load CSV
        print(f"📂 Loading {split} split from {csv_path}...")
        df = pd.read_csv(csv_path)
        self.data = df[df['split'] == split].reset_index(drop=True)
        print(f"✅ Loaded {len(self.data)} examples")
        
        # Index images
        print(f"🔍 Indexing images...")
        self.image_paths = {}
        for img_file in self.image_dir.rglob('*.jpg'):
            self.image_paths[img_file.stem] = img_file
        print(f"✅ {len(self.image_paths)} images indexed")
        
        # Filter valid examples (have image AND fuzzy features)
        print(f"🔍 Filtering valid examples...")
        valid_indices = []
        for idx in tqdm(range(len(self.data)), desc="Filtering"):
            painting = self.data.loc[idx, 'painting']
            
            if painting in self.image_paths and painting in self.fuzzy_cache:
                valid_indices.append(idx)
        
        self.data = self.data.iloc[valid_indices].reset_index(drop=True)
        print(f"✅ {len(self.data)} valid examples with fuzzy features")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        # Get image
        painting = row['painting']
        img_path = self.image_paths[painting]
        
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        
        # Load CACHED fuzzy features (instantaneous!)
        fuzzy_vector = torch.from_numpy(self.fuzzy_cache[painting])
        
        # Tokenize text
        utterance = row['utterance']
        tokens = self.tokenizer(
            utterance,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Get label
        emotion = row['emotion']
        label = EMOTION_TO_IDX[emotion]
        
        return {
            'image': image,
            'input_ids': tokens['input_ids'].squeeze(0),
            'attention_mask': tokens['attention_mask'].squeeze(0),
            'fuzzy_features': fuzzy_vector,
            'label': torch.tensor(label, dtype=torch.long)
        }


# ============================================================================
# MODEL v3
# ============================================================================

class MultimodalFuzzyClassifier(nn.Module):
    """ResNet50 (2048) + RoBERTa (768) + Fuzzy (7) → 9 emotions"""
    
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
        
        # Fusion MLP: 2048 + 768 + 7 = 2823 → 9
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
        # Visual: [B, 2048]
        visual_feats = self.visual_encoder(image).view(image.size(0), -1)
        
        # Text: [B, 768]
        text_output = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        text_feats = text_output.last_hidden_state[:, 0, :]
        
        # Concat: [B, 2823]
        combined = torch.cat([visual_feats, text_feats, fuzzy_features], dim=1)
        
        # Classify
        logits = self.fusion(combined)
        return logits


# ============================================================================
# TRAINING
# ============================================================================

def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    
    total_loss = 0
    correct = 0
    total = 0
    
    pbar = tqdm(dataloader, desc="Training")
    for batch in pbar:
        images = batch['image'].to(device)
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        fuzzy_features = batch['fuzzy_features'].to(device)
        labels = batch['label'].to(device)
        
        optimizer.zero_grad()
        
        outputs = model(images, input_ids, attention_mask, fuzzy_features)
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        pbar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'acc': f"{100.*correct/total:.2f}%"
        })
    
    return total_loss / len(dataloader), 100. * correct / total


def validate(model, dataloader, criterion, device):
    model.eval()
    
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Validation")
        for batch in pbar:
            images = batch['image'].to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            fuzzy_features = batch['fuzzy_features'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(images, input_ids, attention_mask, fuzzy_features)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'acc': f"{100.*correct/total:.2f}%"
            })
    
    return total_loss / len(dataloader), 100. * correct / total


# ============================================================================
# MAIN
# ============================================================================

def main():
    # Paths
    csv_path = '/home/paloma/cerebrum-artis/artemis-v2/dataset/official_data/combined_artemis_with_splits.csv'
    image_dir = '/data/paloma/data/paintings/wikiart'
    fuzzy_cache_file = '/data/paloma/fuzzy_features_cache.pkl'
    checkpoint_dir = Path('/data/paloma/deep-mind-checkpoints/v3_fuzzy_features')
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Hyperparameters
    batch_size = 32  # Adjusted for CUDA memory constraints with RoBERTa
    num_epochs = 20  # Full training with early stopping
    learning_rate = 2e-5
    num_workers = 8
    early_stop_patience = 5  # Stop if no improvement for 5 epochs
    
    print("="*80)
    print("🚀 DEEP-MIND v3: Multimodal + CACHED Fuzzy Features")
    print("="*80)
    print(f"📦 Batch size: {batch_size}")
    print(f"📚 Epochs: {num_epochs}")
    print(f"🎯 Learning rate: {learning_rate}")
    print(f"🖥️  Device: {device}")
    print()
    
    # Load fuzzy cache
    print(f"🧠 Loading fuzzy features cache from {fuzzy_cache_file}...")
    with open(fuzzy_cache_file, 'rb') as f:
        fuzzy_cache = pickle.load(f)
    print(f"✅ Loaded {len(fuzzy_cache)} cached fuzzy feature vectors")
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
    train_dataset = ArtEmisCachedFuzzyDataset(
        csv_path, image_dir, fuzzy_cache, split='train',
        tokenizer=tokenizer, transform=transform
    )
    
    val_dataset = ArtEmisCachedFuzzyDataset(
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
    
    print(f"✅ Train: {len(train_dataset)} examples, {len(train_loader)} batches")
    print(f"✅ Val: {len(val_dataset)} examples, {len(val_loader)} batches")
    print()
    
    # Model
    print("🤖 Creating model...")
    model = MultimodalFuzzyClassifier(num_classes=9, dropout=0.3).to(device)
    
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
    
    # Training loop with early stopping
    best_val_acc = 0.0
    epochs_no_improve = 0

    for epoch in range(1, num_epochs + 1):
        print(f"\n{'='*80}")
        print(f"📅 Epoch {epoch}/{num_epochs}")
        print(f"{'='*80}")

        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        print(f"\n📊 Epoch {epoch}:")
        print(f"  Train - Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%")
        print(f"  Val   - Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%")

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

        torch.save(checkpoint, checkpoint_dir / f'checkpoint_epoch{epoch}_last.pt')

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
    print(f"🎉 Training completed! Best: {best_val_acc:.2f}%")
    print("="*80)


if __name__ == '__main__':
    main()
