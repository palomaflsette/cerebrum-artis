#!/usr/bin/env python3
"""
Quick evaluation script to get precision/recall/F1 for presentation.
Estimates ~10-15min on validation set.
"""
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from transformers import RobertaTokenizer
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.metrics import classification_report, accuracy_score
import json

# Import models  
from cerebrum_artis.models.multimodal_fuzzy import (
    MultimodalEmotionClassifierV2FuzzyFeatures,
    MultimodalEmotionClassifierV3AdaptiveGating,
    MultimodalEmotionClassifierV31Integrated
)
from cerebrum_artis.fuzzy.system import FuzzyEmotionSystem

# Dataset
import torch
from torch.utils.data import Dataset
from PIL import Image

class QuickDataset(Dataset):
    def __init__(self, csv_path, img_dir, split='val', transform=None):
        df = pd.read_csv(csv_path, low_memory=False)
        self.data = df[df['split'] == split].reset_index(drop=True)
        self.img_dir = Path(img_dir)
        self.transform = transform
        
        # Emotion mapping
        self.emotions = ['amusement', 'awe', 'contentment', 'excitement', 
                         'anger', 'disgust', 'fear', 'sadness', 'something_else']
        self.emotion_to_idx = {e: i for i, e in enumerate(self.emotions)}
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        # Load image
        img_path = self.img_dir / f"{row['art_style']}" / f"{row['painting']}.jpg"
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        
        # Get label
        label = self.emotion_to_idx[row['emotion']]
        
        # Get utterance
        utterance = str(row['utterance'])
        
        return image, utterance, label


def evaluate_model(model_class, checkpoint_path, model_name, device, val_loader, tokenizer):
    """Evaluate single model."""
    print(f"\n{'='*70}")
    print(f"📊 Evaluating {model_name}")
    print(f"{'='*70}")
    
    # Load model
    if 'V2' in model_name:
        model = model_class(num_classes=9)
    else:  # V3 or V3.1 need fuzzy system
        fuzzy_system = FuzzyEmotionSystem()
        model = model_class(num_classes=9, fuzzy_system=fuzzy_system)
    
    # Load checkpoint
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    # Evaluate
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, utterances, labels in tqdm(val_loader, desc=f"{model_name}"):
            images = images.to(device)
            labels = labels.to(device)
            
            # Tokenize
            encoding = tokenizer(utterances, padding=True, truncation=True, 
                                max_length=128, return_tensors='pt')
            input_ids = encoding['input_ids'].to(device)
            attention_mask = encoding['attention_mask'].to(device)
            
            # Forward
            if 'V2' in model_name:
                outputs = model(images, input_ids, attention_mask)
            else:  # V3/V3.1 need fuzzy features (pass image again)
                outputs = model(images, input_ids, attention_mask, images)
            
            preds = outputs.argmax(dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate metrics
    emotions = ['amusement', 'awe', 'contentment', 'excitement', 
                'anger', 'disgust', 'fear', 'sadness', 'something_else']
    
    report = classification_report(all_labels, all_preds, target_names=emotions, 
                                   output_dict=True, zero_division=0)
    
    accuracy = accuracy_score(all_labels, all_preds)
    
    results = {
        'model_name': model_name,
        'accuracy': float(accuracy),
        'macro_precision': float(report['macro avg']['precision']),
        'macro_recall': float(report['macro avg']['recall']),
        'macro_f1': float(report['macro avg']['f1-score']),
        'per_class': {
            emotion: {
                'precision': float(report[emotion]['precision']),
                'recall': float(report[emotion]['recall']),
                'f1': float(report[emotion]['f1-score'])
            }
            for emotion in emotions
        }
    }
    
    print(f"✅ Accuracy: {accuracy*100:.2f}%")
    print(f"   Macro Precision: {results['macro_precision']*100:.2f}%")
    print(f"   Macro Recall: {results['macro_recall']*100:.2f}%")
    print(f"   Macro F1: {results['macro_f1']*100:.2f}%")
    
    return results


def main():
    print("🚀 QUICK EVALUATION FOR PRESENTATION")
    print("Estimativa: ~10-15 min para 3 modelos\n")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")
    
    # Paths
    csv_path = '/home/paloma/cerebrum-artis/data/artemis/dataset/combined_artemis.csv'
    img_dir = '/data/paloma/datasets/artemis/images'
    
    # Transform
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Dataset
    print("📦 Loading validation data...")
    val_dataset = QuickDataset(csv_path, img_dir, split='val', transform=transform)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, 
                           num_workers=4, pin_memory=True)
    print(f"✅ Loaded {len(val_dataset)} validation samples\n")
    
    # Tokenizer
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    
    # Models to evaluate
    models_config = [
        {
            'name': 'V2 Fuzzy Features',
            'class': MultimodalEmotionClassifierV2FuzzyFeatures,
            'checkpoint': '/data/paloma/deep-mind-checkpoints/v2_fuzzy_features/checkpoint_best.pt'
        },
        {
            'name': 'V3 Adaptive Gating',
            'class': MultimodalEmotionClassifierV3AdaptiveGating,
            'checkpoint': '/data/paloma/deep-mind-checkpoints/v3_adaptive_gating/checkpoint_best.pt'
        },
        {
            'name': 'V3.1 Integrated',
            'class': MultimodalEmotionClassifierV31Integrated,
            'checkpoint': '/data/paloma/deep-mind-checkpoints/v3_1_integrated/checkpoint_best.pt'
        }
    ]
    
    # Evaluate all
    all_results = {}
    for config in models_config:
        try:
            results = evaluate_model(
                config['class'], 
                config['checkpoint'], 
                config['name'],
                device,
                val_loader,
                tokenizer
            )
            all_results[config['name']] = results
        except Exception as e:
            print(f"❌ Error evaluating {config['name']}: {e}")
            import traceback
            traceback.print_exc()
    
    # Save results
    output_path = '/home/paloma/cerebrum-artis/outputs/metrics/complete_evaluation.json'
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n{'='*70}")
    print("📊 FINAL RESULTS")
    print(f"{'='*70}\n")
    
    print(f"{'Model':<25} {'Accuracy':>10} {'Macro-F1':>10} {'Precision':>10} {'Recall':>10}")
    print("-"*70)
    for name, results in all_results.items():
        print(f"{name:<25} {results['accuracy']*100:>9.2f}% "
              f"{results['macro_f1']*100:>9.2f}% "
              f"{results['macro_precision']*100:>9.2f}% "
              f"{results['macro_recall']*100:>9.2f}%")
    
    print(f"\n✅ Results saved to: {output_path}")


if __name__ == '__main__':
    main()
