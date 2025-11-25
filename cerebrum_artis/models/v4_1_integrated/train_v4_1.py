#!/usr/bin/env python3
"""
Deep-Mind v4.1: Integrated Fuzzy-Neural Gating
===============================================

REFATORAÇÃO do V4:
- Fuzzy system DENTRO do modelo
- Agreement calculation DENTRO do forward()
- Adaptive alpha calculation DENTRO do forward()
- Tudo encapsulado, pronto para produção

Diferenças vs V4:
- V4: Fuzzy inference FORA do modelo (external)
- V4.1: Fuzzy inference DENTRO do modelo (integrated)
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
# DATASET with CACHED Fuzzy Features (same as V4)
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
# MODEL v4.1: INTEGRATED Fuzzy-Neural Gating
# ============================================================================

class IntegratedFuzzyGatingClassifier(nn.Module):
    """
    V4.1: Neural + Fuzzy with INTEGRATED gating
    
    Everything encapsulated:
    - Fuzzy system is a model component
    - Forward pass returns fused output
    - Agreement & alpha calculated inside
    - Proper production-ready architecture
    """
    
    def __init__(self, num_classes=9, dropout=0.3, freeze_resnet=True,
                 base_alpha=0.85, min_alpha=0.6, max_alpha=0.95):
        super().__init__()
        
        self.num_classes = num_classes
        self.base_alpha = base_alpha
        self.min_alpha = min_alpha
        self.max_alpha = max_alpha
        
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
        
        # 🔥 INTEGRATED: Fuzzy system as model component
        self.fuzzy_system = FuzzyInferenceSystem()
        
        # Emotion mapping for fuzzy output
        self.emotion_names = EMOTIONS
    
    def _batch_fuzzy_inference(self, fuzzy_features_batch):
        """
        INTEGRATED: Fuzzy inference within model
        
        Args:
            fuzzy_features_batch: Tensor [B, 7]
        
        Returns:
            fuzzy_probs: Tensor [B, 9]
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
            fuzzy_dist = self.fuzzy_system.infer(features_dict)
            
            # Convert to tensor
            fuzzy_prob = torch.tensor(
                [fuzzy_dist.get(e, 0.0) for e in self.emotion_names],
                device=device,
                dtype=torch.float32
            )
            
            fuzzy_probs_list.append(fuzzy_prob)
        
        fuzzy_probs = torch.stack(fuzzy_probs_list)  # [B, 9]
        
        return fuzzy_probs
    
    def _adaptive_fusion(self, neural_logits, fuzzy_probs):
        """
        INTEGRATED: Adaptive fusion with agreement-based gating
        
        Args:
            neural_logits: [B, 9]
            fuzzy_probs: [B, 9]
        
        Returns:
            final_logits: [B, 9]
            agreement: [B]
            adaptive_alpha: [B]
        """
        # Convert neural logits to probabilities
        neural_probs = torch.softmax(neural_logits, dim=1)
        
        # Compute agreement using cosine similarity
        agreement = torch.nn.functional.cosine_similarity(
            neural_probs, fuzzy_probs, dim=1
        )
        # Normalize to [0, 1]
        agreement = (agreement + 1) / 2
        
        # Adaptive alpha based on agreement
        # High agreement → lower alpha (give fuzzy more weight)
        # Low agreement → higher alpha (trust neural)
        adaptive_alpha = self.max_alpha - (self.max_alpha - self.min_alpha) * agreement
        adaptive_alpha = adaptive_alpha.unsqueeze(1)  # [B, 1]
        
        # Weighted fusion
        final_probs = adaptive_alpha * neural_probs + (1 - adaptive_alpha) * fuzzy_probs
        
        # Convert back to logits for loss calculation
        final_logits = torch.log(final_probs + 1e-8)
        
        return final_logits, agreement, adaptive_alpha.squeeze(1)
    
    def forward(self, image, input_ids, attention_mask, fuzzy_features=None, 
                return_components=False):
        """
        INTEGRATED forward pass
        
        Args:
            image: [B, 3, 224, 224]
            input_ids: [B, seq_len]
            attention_mask: [B, seq_len]
            fuzzy_features: [B, 7] - REQUIRED for V4.1
            return_components: If True, returns (logits, agreement, alpha, neural_logits, fuzzy_probs)
        
        Returns:
            If return_components=False: final_logits [B, 9]
            If return_components=True: tuple(final_logits, agreement, alpha, neural_logits, fuzzy_probs)
        """
        # Visual: [B, 2048]
        visual_feats = self.visual_encoder(image).view(image.size(0), -1)
        
        # Text: [B, 768]
        text_output = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        text_feats = text_output.last_hidden_state[:, 0, :]
        
        # Neural branch
        combined = torch.cat([visual_feats, text_feats], dim=1)
        neural_logits = self.classifier(combined)
        
        # Fuzzy branch (INTEGRATED)
        if fuzzy_features is None:
            # Fallback: neural only (no fuzzy features provided)
            if return_components:
                return neural_logits, None, None, neural_logits, None
            return neural_logits
        
        fuzzy_probs = self._batch_fuzzy_inference(fuzzy_features)
        
        # Adaptive fusion (INTEGRATED)
        final_logits, agreement, alpha = self._adaptive_fusion(neural_logits, fuzzy_probs)
        
        if return_components:
            return final_logits, agreement, alpha, neural_logits, fuzzy_probs
        
        return final_logits


# ============================================================================
# TRAINING (Simplified - no external fuzzy calls)
# ============================================================================

def train_epoch(model, dataloader, criterion, optimizer, device):
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
        
        # 🔥 INTEGRATED: Everything in one forward pass
        final_logits, agreement, alpha, _, _ = model(
            images, input_ids, attention_mask, fuzzy_features,
            return_components=True
        )
        
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


def validate(model, dataloader, criterion, device):
    model.eval()
    
    total_loss = 0
    correct = 0
    total = 0
    avg_agreement = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validating"):
            images = batch['image'].to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            fuzzy_features = batch['fuzzy_features'].to(device)
            labels = batch['label'].to(device)
            
            # 🔥 INTEGRATED: Everything in one forward pass
            final_logits, agreement, alpha, _, _ = model(
                images, input_ids, attention_mask, fuzzy_features,
                return_components=True
            )
            
            loss = criterion(final_logits, labels)
            
            total_loss += loss.item()
            
            _, predicted = final_logits.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            avg_agreement += agreement.mean().item()
    
    n_batches = len(dataloader)
    return (total_loss / n_batches,
            100. * correct / total,
            avg_agreement / n_batches)


# ============================================================================
# MAIN TRAINING LOOP
# ============================================================================

def main():
    print("=" * 80)
    print("🧠 DEEP-MIND V4.1: INTEGRATED FUZZY-NEURAL GATING")
    print("=" * 80)
    print(f"Device: {device}")
    print()
    
    # Paths
    csv_path = '/home/paloma/cerebrum-artis/artemis-v2/dataset/official_data/combined_artemis_with_splits.csv'
    image_dir = '/data/paloma/data/paintings/wikiart'
    fuzzy_cache_path = '/data/paloma/fuzzy_features_cache.pkl'
    checkpoint_dir = '/data/paloma/deep-mind-checkpoints/v4.1_integrated_gating'
    
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Load fuzzy cache
    print("📦 Loading fuzzy features cache...")
    with open(fuzzy_cache_path, 'rb') as f:
        fuzzy_cache = pickle.load(f)
    print(f"✅ {len(fuzzy_cache)} paintings in cache")
    print()
    
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
    
    print(f"\n✅ Train: {len(train_dataset)} | Val: {len(val_dataset)}")
    print()
    
    # Model V4.1
    print("=" * 80)
    print("INITIALIZING MODEL V4.1")
    print("=" * 80)
    
    model = IntegratedFuzzyGatingClassifier(
        num_classes=9,
        dropout=0.3,
        freeze_resnet=True,
        base_alpha=0.85,
        min_alpha=0.6,
        max_alpha=0.95
    ).to(device)
    
    print(f"✅ V4.1 model created")
    
    # Load V4 weights (strict=False to skip fuzzy_system)
    v4_checkpoint_path = '/data/paloma/deep-mind-checkpoints/v4_fuzzy_gating/checkpoint_best.pt'
    if os.path.exists(v4_checkpoint_path):
        print(f"\n🔄 Loading V4 weights from: {v4_checkpoint_path}")
        v4_checkpoint = torch.load(v4_checkpoint_path, map_location=device)
        
        # Load with strict=False (V4.1 has fuzzy_system, V4 doesn't)
        missing_keys, unexpected_keys = model.load_state_dict(
            v4_checkpoint['model_state_dict'], 
            strict=False
        )
        
        print(f"✅ V4 weights loaded!")
        print(f"   📝 Missing keys (expected): {len(missing_keys)}")
        print(f"   📝 Unexpected keys: {len(unexpected_keys)}")
        print(f"   📊 V4 checkpoint: epoch {v4_checkpoint['epoch']}, val_acc={v4_checkpoint['val_acc']:.2f}%")
        start_epoch = v4_checkpoint['epoch'] + 1
    else:
        print("\n⚠️  No V4 checkpoint found, training from scratch")
        start_epoch = 1
    
    print()
    
    # Optimizer & Loss
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-5,  # Lower LR for fine-tuning
        weight_decay=0.01
    )
    
    criterion = nn.CrossEntropyLoss()
    
    # Training config
    num_epochs = 20
    best_val_acc = 0
    epochs_no_improve = 0
    early_stop_patience = 5  # Stop if no improvement for 5 epochs
    
    log_file = os.path.join(checkpoint_dir, 'training_log.txt')
    
    print("=" * 80)
    print("STARTING TRAINING")
    print("=" * 80)
    print(f"Epochs: {start_epoch} → {num_epochs}")
    print(f"Learning rate: 1e-5 (fine-tuning)")
    print(f"Checkpoints: {checkpoint_dir}")
    print(f"Log: {log_file}")
    print("=" * 80)
    print()
    
    with open(log_file, 'a') as f:
        f.write(f"\n{'='*80}\n")
        f.write(f"V4.1 Training Started: {datetime.now()}\n")
        f.write(f"Start epoch: {start_epoch}\n")
        f.write(f"{'='*80}\n\n")
    
    for epoch in range(start_epoch, num_epochs + 1):
        print(f"\n{'='*80}")
        print(f"EPOCH {epoch}/{num_epochs}")
        print(f"{'='*80}")
        
        # Train
        train_loss, train_acc, train_agreement = train_epoch(
            model, train_loader, criterion, optimizer, device
        )
        
        # Validate
        val_loss, val_acc, val_agreement = validate(
            model, val_loader, criterion, device
        )
        
        # Log
        log_msg = (
            f"Epoch {epoch:02d} | "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% Agreement: {train_agreement:.3f} | "
            f"Val Loss: {val_loss:.4f} Acc: {val_acc:.2f}% Agreement: {val_agreement:.3f}"
        )
        
        print(f"\n📊 {log_msg}")
        
        with open(log_file, 'a') as f:
            f.write(log_msg + '\n')
        
        # Save checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'train_acc': train_acc,
            'train_agreement': train_agreement,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'val_agreement': val_agreement
        }
        
        # Always save last
        last_path = os.path.join(checkpoint_dir, f'checkpoint_epoch{epoch}_last.pt')
        torch.save(checkpoint, last_path)
        print(f"💾 Saved: {last_path}")
        
        # Save best
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_no_improve = 0
            best_path = os.path.join(checkpoint_dir, 'checkpoint_best.pt')
            torch.save(checkpoint, best_path)
            print(f"🏆 NEW BEST! Saved: {best_path}")
        else:
            epochs_no_improve += 1
            print(f"⏳ No improvement for {epochs_no_improve}/{early_stop_patience} epochs")
        
        # Early stopping
        if epochs_no_improve >= early_stop_patience:
            print(f"\n🛑 EARLY STOPPING! No improvement for {early_stop_patience} epochs")
            print(f"   Best val acc: {best_val_acc:.2f}%")
            break
        
        # Cleanup old checkpoints (keep only last 2 + best)
        cleanup_old_checkpoints(checkpoint_dir, current_epoch=epoch, keep_last=2)
    
    print(f"\n{'='*80}")
    print(f"✅ TRAINING COMPLETE!")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    print(f"{'='*80}")


def cleanup_old_checkpoints(checkpoint_dir, current_epoch, keep_last=2):
    """Delete checkpoints older than keep_last epochs"""
    checkpoint_files = list(Path(checkpoint_dir).glob('checkpoint_epoch*_last.pt'))
    
    for ckpt_file in checkpoint_files:
        # Extract epoch number
        epoch_num = int(ckpt_file.stem.split('_')[1].replace('epoch', ''))
        
        # Delete if too old
        if current_epoch - epoch_num > keep_last:
            ckpt_file.unlink()
            print(f"🗑️  Deleted old checkpoint: {ckpt_file.name}")


if __name__ == '__main__':
    main()
