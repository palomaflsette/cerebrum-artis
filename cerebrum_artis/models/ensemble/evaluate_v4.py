#!/usr/bin/env python3
"""
Evaluate V4 Ensemble on validation/test set.

Compares:
- V2 alone
- V3 alone  
- V4 Ensemble (50/50)
- V4 Ensemble (optimized weights)
"""

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from transformers import RobertaTokenizer
from PIL import Image
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    classification_report, confusion_matrix
)
import argparse
import pickle

# Add project root
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from cerebrum_artis.models.ensemble.ensemble_v4 import load_ensemble

# Fix for DecompressionBombWarning
Image.MAX_IMAGE_PIXELS = None

# Emotions
EMOTIONS = [
    'amusement', 'awe', 'contentment', 'excitement',
    'anger', 'disgust', 'fear', 'sadness', 'something else'
]
EMOTION_TO_IDX = {emotion: idx for idx, emotion in enumerate(EMOTIONS)}


# ============================================================================
# Dataset (copied from training script)
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
        
        # VALIDAÇÃO ANTI-VAZAMENTO: Garantir que split está correto
        self.data = df[df['split'] == split].reset_index(drop=True)
        
        # Index images
        print(f"🔍 Indexing images...")
        self.image_paths = {}
        for img_file in self.image_dir.rglob('*.jpg'):
            self.image_paths[img_file.stem] = img_file
        print(f"✅ {len(self.image_paths)} images")
        
        # Filter valid
        print(f"🔍 Filtering...")
        valid_indices = []
        for idx in tqdm(range(len(self.data)), desc="Filtering", disable=True):
            painting = self.data.loc[idx, 'painting']
            if painting in self.image_paths and painting in self.fuzzy_cache:
                valid_indices.append(idx)
        
        self.data = self.data.iloc[valid_indices].reset_index(drop=True)
        print(f"✅ {len(self.data)} valid examples")
    
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
# Evaluation
# ============================================================================

def evaluate_ensemble(ensemble, dataloader, device='cuda'):
    """Evaluate ensemble and individual models."""
    
    ensemble.eval()
    
    all_labels = []
    ensemble_preds = []
    v2_preds = []
    v3_preds = []
    
    print("\n🔍 Evaluating on dataset...")
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            image = batch['image'].to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            fuzzy = batch['fuzzy_features'].to(device)
            labels = batch['label']
            
            # Get predictions from ensemble and individual models
            ens_logits, v2_logits, v3_logits = ensemble(
                image, input_ids, attention_mask, fuzzy
            )
            
            # Convert to predictions
            ens_pred = torch.argmax(ens_logits, dim=1).cpu().numpy()
            v2_pred = torch.argmax(v2_logits, dim=1).cpu().numpy()
            v3_pred = torch.argmax(v3_logits, dim=1).cpu().numpy()
            
            ensemble_preds.extend(ens_pred)
            v2_preds.extend(v2_pred)
            v3_preds.extend(v3_pred)
            all_labels.extend(labels.numpy())
    
    # Convert to numpy
    all_labels = np.array(all_labels)
    ensemble_preds = np.array(ensemble_preds)
    v2_preds = np.array(v2_preds)
    v3_preds = np.array(v3_preds)
    
    # Compute metrics
    results = {}
    
    for name, preds in [('V2', v2_preds), ('V3', v3_preds), ('V4_Ensemble', ensemble_preds)]:
        acc = accuracy_score(all_labels, preds)
        f1 = f1_score(all_labels, preds, average='macro')
        precision = precision_score(all_labels, preds, average='macro')
        recall = recall_score(all_labels, preds, average='macro')
        
        results[name] = {
            'accuracy': acc,
            'f1': f1,
            'precision': precision,
            'recall': recall
        }
    
    return results, all_labels, ensemble_preds, v2_preds, v3_preds


def print_results(results):
    """Print comparison table."""
    
    print("\n" + "="*80)
    print("📊 MODEL COMPARISON")
    print("="*80)
    print(f"{'Model':<15} | {'Accuracy':<10} | {'F1 Score':<10} | {'Precision':<10} | {'Recall':<10}")
    print("-"*80)
    
    for model_name in ['V2', 'V3', 'V4_Ensemble']:
        r = results[model_name]
        print(f"{model_name:<15} | {r['accuracy']:<10.4f} | {r['f1']:<10.4f} | "
              f"{r['precision']:<10.4f} | {r['recall']:<10.4f}")
    
    print("="*80)
    
    # Highlight improvement
    v2_f1 = results['V2']['f1']
    v3_f1 = results['V3']['f1']
    ens_f1 = results['V4_Ensemble']['f1']
    
    improvement_vs_v2 = (ens_f1 - v2_f1) * 100
    improvement_vs_v3 = (ens_f1 - v3_f1) * 100
    
    print(f"\n💡 Ensemble Improvement:")
    print(f"   vs V2: {improvement_vs_v2:+.2f}% F1")
    print(f"   vs V3: {improvement_vs_v3:+.2f}% F1")
    
    if ens_f1 > max(v2_f1, v3_f1):
        print(f"   🎉 ENSEMBLE WINS! (+{max(improvement_vs_v2, improvement_vs_v3):.2f}%)")
    else:
        print(f"   ⚠️  Individual model better (ensemble may need weight tuning)")


def print_per_class_metrics(labels, predictions):
    """Print per-class performance."""
    
    print("\n" + "="*80)
    print("📊 PER-CLASS METRICS (V4 Ensemble)")
    print("="*80)
    
    report = classification_report(
        labels, predictions, 
        target_names=EMOTIONS,
        digits=4
    )
    print(report)


def main():
    parser = argparse.ArgumentParser(description='Evaluate V4 Ensemble')
    parser.add_argument('--split', choices=['val', 'test'], default='val',
                        help='Dataset split to evaluate')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--optimize-weights', action='store_true',
                        help='Optimize ensemble weights on validation set')
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    
    device = torch.device(args.device)
    
    # Paths
    CSV_PATH = '/home/paloma/cerebrum-artis/data/artemis/dataset/official_data/combined_artemis_with_splits.csv'
    IMAGE_DIR = '/data/paloma/data/paintings/wikiart'
    FUZZY_CACHE_PATH = '/data/paloma/fuzzy_features_cache.pkl'
    
    # Load dataset
    print(f"\n📂 Loading {args.split} dataset...")
    
    # Load fuzzy cache
    print("📦 Loading fuzzy cache...")
    with open(FUZZY_CACHE_PATH, 'rb') as f:
        fuzzy_cache = pickle.load(f)
    print(f"✅ Loaded {len(fuzzy_cache)} cached fuzzy features")
    
    # Image transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Tokenizer
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    
    # Dataset
    dataset = ArtEmisFuzzyGatingDataset(
        csv_path=CSV_PATH,
        image_dir=IMAGE_DIR,
        fuzzy_cache=fuzzy_cache,
        split=args.split,
        tokenizer=tokenizer,
        transform=transform,
        max_length=128
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    print(f"✅ Loaded {len(dataset)} examples")
    
    # Load ensemble
    print("\n🔧 Loading V4 Ensemble...")
    
    ensemble = load_ensemble(
        device=device,
        optimize=args.optimize_weights,
        val_loader=dataloader if args.optimize_weights else None
    )
    
    # Evaluate
    results, labels, ens_preds, v2_preds, v3_preds = evaluate_ensemble(
        ensemble, dataloader, device
    )
    
    # Print results
    print_results(results)
    print_per_class_metrics(labels, ens_preds)
    
    # Save results
    output_dir = PROJECT_ROOT / 'outputs' / 'ensemble_evaluation'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    np.savez(
        output_dir / f'v4_ensemble_{args.split}_predictions.npz',
        labels=labels,
        ensemble_preds=ens_preds,
        v2_preds=v2_preds,
        v3_preds=v3_preds,
        v2_weight=ensemble.v2_weight,
        v3_weight=ensemble.v3_weight
    )
    
    print(f"\n💾 Results saved to {output_dir}")


if __name__ == '__main__':
    main()
