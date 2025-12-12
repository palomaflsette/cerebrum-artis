#!/usr/bin/env python3
"""
Deep-Mind V2: Multimodal + Fuzzy Features (IMPROVED)
=====================================================

MELHORIAS:
- ✅ F1 Score, Precision, Recall por classe e macro
- ✅ Salvamento SOMENTE do best checkpoint em /data/paloma/deep-mind-checkpoints
- ✅ Estratificação e validação anti-vazamento
- ✅ Monitoramento de métricas em tempo real
- ✅ Early Stop com LR Scheduler
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
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report
import warnings
warnings.filterwarnings('ignore')

# Fix para DecompressionBombWarning
Image.MAX_IMAGE_PIXELS = None

# Fuzzy system
sys.path.insert(0, '/home/paloma/cerebrum-artis/cerebrum_artis/fuzzy')
from fuzzy_brain.fuzzy.system import FuzzyInferenceSystem

# ============================================================================
# CONSTANTS
# ============================================================================

EMOTIONS = [
    'amusement', 'awe', 'contentment', 'excitement', 
    'anger', 'disgust', 'fear', 'sadness', 'something else'
]

EMOTION_TO_IDX = {emotion: idx for idx, emotion in enumerate(EMOTIONS)}

# ============================================================================
# DATASET with CACHED Fuzzy Features + Stratification Validation
# ============================================================================

class ArtEmisFuzzyGatingDataset(Dataset):
    """Dataset with cached fuzzy features and split validation"""
    
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
        
        # ✅ VALIDAÇÃO ANTI-VAZAMENTO: Garantir que split está correto
        self.data = df[df['split'] == split].reset_index(drop=True)
        
        # Verificar estratificação
        print(f"📊 Emotion distribution in {split}:")
        emotion_counts = self.data['emotion'].value_counts()
        for emotion in EMOTIONS:
            count = emotion_counts.get(emotion, 0)
            pct = 100 * count / len(self.data) if len(self.data) > 0 else 0
            print(f"  {emotion:15s}: {count:6d} ({pct:5.2f}%)")
        
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
# MODEL V2: Simple Concatenation (Visual + Text + Fuzzy)
# ============================================================================

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


# ============================================================================
# TRAINING WITH METRICS (V2: No external fuzzy inference)
# ============================================================================

def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    
    total_loss = 0
    all_preds = []
    all_labels = []
    
    pbar = tqdm(dataloader, desc="Training", ncols=100)
    for batch in pbar:
        images = batch['image'].to(device)
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        fuzzy_features = batch['fuzzy_features'].to(device)
        labels = batch['label'].to(device)
        
        optimizer.zero_grad()
        
        # Forward pass (fuzzy features concatenated inside model)
        logits = model(images, input_ids, attention_mask, fuzzy_features)
        
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        _, predicted = logits.max(1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
        # Calcular métricas parciais
        current_acc = 100 * np.mean(np.array(all_preds) == np.array(all_labels))
        
        pbar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'acc': f"{current_acc:.2f}%"
        })
    
    n_batches = len(dataloader)
    
    # ✅ MÉTRICAS COMPLETAS
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    acc = 100 * np.mean(all_preds == all_labels)
    f1_macro = 100 * f1_score(all_labels, all_preds, average='macro', zero_division=0)
    precision_macro = 100 * precision_score(all_labels, all_preds, average='macro', zero_division=0)
    recall_macro = 100 * recall_score(all_labels, all_preds, average='macro', zero_division=0)
    
    return {
        'loss': total_loss / n_batches,
        'acc': acc,
        'f1': f1_macro,
        'precision': precision_macro,
        'recall': recall_macro
    }


def validate(model, dataloader, criterion, device, verbose=False):
    model.eval()
    
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validating", ncols=100):
            images = batch['image'].to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            fuzzy_features = batch['fuzzy_features'].to(device)
            labels = batch['label'].to(device)
            
            # Forward
            logits = model(images, input_ids, attention_mask, fuzzy_features)
            
            loss = criterion(logits, labels)
            total_loss += loss.item()
            
            _, predicted = logits.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    n_batches = len(dataloader)
    
    # ✅ MÉTRICAS COMPLETAS
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    acc = 100 * np.mean(all_preds == all_labels)
    f1_macro = 100 * f1_score(all_labels, all_preds, average='macro', zero_division=0)
    precision_macro = 100 * precision_score(all_labels, all_preds, average='macro', zero_division=0)
    recall_macro = 100 * recall_score(all_labels, all_preds, average='macro', zero_division=0)
    
    # Per-class metrics (verbose)
    if verbose:
        print("\n📊 Per-Class Metrics:")
        f1_per_class = f1_score(all_labels, all_preds, average=None, zero_division=0)
        precision_per_class = precision_score(all_labels, all_preds, average=None, zero_division=0)
        recall_per_class = recall_score(all_labels, all_preds, average=None, zero_division=0)
        
        for i, emotion in enumerate(EMOTIONS):
            print(f"  {emotion:15s}: F1={f1_per_class[i]:.4f} | P={precision_per_class[i]:.4f} | R={recall_per_class[i]:.4f}")
    
    return {
        'loss': total_loss / n_batches,
        'acc': acc,
        'f1': f1_macro,
        'precision': precision_macro,
        'recall': recall_macro
    }


# ============================================================================
# MAIN TRAINING LOOP
# ============================================================================

def main():
    print("=" * 80)
    print("🧠 DEEP-MIND V2 (IMPROVED): MULTIMODAL + FUZZY FEATURES")
    print("=" * 80)
    
    # GPU (will be mapped via CUDA_VISIBLE_DEVICES)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print()
    
    # Paths
    csv_path = '/home/paloma/cerebrum-artis/data/artemis/dataset/official_data/combined_artemis_with_splits.csv'
    image_dir = '/data/paloma/data/paintings/wikiart'
    fuzzy_cache_path = '/data/paloma/fuzzy_features_cache.pkl'
    checkpoint_dir = '/data/paloma/deep-mind-checkpoints/v2_fuzzy_features'
    
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Load fuzzy cache
    print("📦 Loading fuzzy features cache...")
    with open(fuzzy_cache_path, 'rb') as f:
        fuzzy_cache = pickle.load(f)
    print(f"✅ {len(fuzzy_cache)} paintings in cache\n")
    
    # Transforms
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Tokenizer
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    
    # Datasets
    print("=" * 80)
    print("LOADING DATASETS")
    print("=" * 80)
    
    train_dataset = ArtEmisFuzzyGatingDataset(
        csv_path, image_dir, fuzzy_cache, 
        split='train', tokenizer=tokenizer, transform=train_transform
    )
    
    val_dataset = ArtEmisFuzzyGatingDataset(
        csv_path, image_dir, fuzzy_cache,
        split='val', tokenizer=tokenizer, transform=val_transform
    )
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)
    
    print(f"\n✅ Train: {len(train_dataset)} | Val: {len(val_dataset)}\n")
    
    # Model
    print("=" * 80)
    print("INITIALIZING MODEL V2")
    print("=" * 80)
    
    model = MultimodalFuzzyClassifier(
        num_classes=9,
        dropout=0.3,
        freeze_resnet=True
    ).to(device)
    
    print(f"✅ V2 model created\n")
    
    # Optimizer
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=2e-5,
        weight_decay=0.01
    )
    
    # ✅ LR SCHEDULER: Reduz LR quando validation F1 para de melhorar
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3, verbose=True
    )
    
    criterion = nn.CrossEntropyLoss()
    
    # Training config
    num_epochs = 20
    best_val_f1 = 0  # Usar F1 como métrica principal
    epochs_no_improve = 0
    early_stop_patience = 5  # ✅ EARLY STOP
    
    log_file = os.path.join(checkpoint_dir, 'training_log.txt')
    
    print("=" * 80)
    print("STARTING TRAINING")
    print("=" * 80)
    print(f"Epochs: {num_epochs}")
    print(f"Learning rate: 2e-5 (adaptive with ReduceLROnPlateau)")
    print(f"Early Stop: {early_stop_patience} epochs")
    print(f"Checkpoints: {checkpoint_dir}")
    print(f"Log: {log_file}")
    print("=" * 80)
    print()
    
    with open(log_file, 'a') as f:
        f.write(f"\n{'='*80}\n")
        f.write(f"V3 Training Started: {datetime.now()}\n")
        f.write(f"{'='*80}\n\n")
    
    for epoch in range(1, num_epochs + 1):
        print(f"\n{'='*80}")
        print(f"EPOCH {epoch}/{num_epochs}")
        print(f"{'='*80}")
        
        # Train
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_metrics = validate(model, val_loader, criterion, device, verbose=(epoch % 5 == 0))
        
        # ✅ SCHEDULER step (reduz LR se F1 não melhorar)
        scheduler.step(val_metrics['f1'])
        
        # Log
        log_msg = (
            f"Epoch {epoch:02d} | "
            f"Train: Loss={train_metrics['loss']:.4f} Acc={train_metrics['acc']:.2f}% "
            f"F1={train_metrics['f1']:.2f}% P={train_metrics['precision']:.2f}% R={train_metrics['recall']:.2f}% | "
            f"Val: Loss={val_metrics['loss']:.4f} Acc={val_metrics['acc']:.2f}% "
            f"F1={val_metrics['f1']:.2f}% P={val_metrics['precision']:.2f}% R={val_metrics['recall']:.2f}%"
        )
        
        print(f"\n📊 {log_msg}")
        
        with open(log_file, 'a') as f:
            f.write(log_msg + '\n')
        
        # ✅ SALVAR SOMENTE BEST CHECKPOINT
        if val_metrics['f1'] > best_val_f1:
            best_val_f1 = val_metrics['f1']
            epochs_no_improve = 0
            
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_metrics': train_metrics,
                'val_metrics': val_metrics
            }
            
            best_path = os.path.join(checkpoint_dir, 'checkpoint_best.pt')
            torch.save(checkpoint, best_path)
            print(f"🏆 NEW BEST F1! Saved: {best_path}")
        else:
            epochs_no_improve += 1
            print(f"⏳ No improvement for {epochs_no_improve}/{early_stop_patience} epochs")
        
        # ✅ EARLY STOPPING
        if epochs_no_improve >= early_stop_patience:
            print(f"\n🛑 EARLY STOPPING! No improvement for {early_stop_patience} epochs")
            print(f"   Best val F1: {best_val_f1:.2f}%")
            break
    
    print(f"\n{'='*80}")
    print(f"✅ TRAINING COMPLETE!")
    print(f"Best validation F1: {best_val_f1:.2f}%")
    print(f"{'='*80}")


if __name__ == '__main__':
    main()
