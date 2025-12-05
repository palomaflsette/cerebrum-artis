"""
Ensemble Testing Script
Tests different ensemble strategies combining V2, V3, and V3.1
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import sys
import os

# Add paths
sys.path.append('/home/paloma/cerebrum-artis/deep-mind/v2_fuzzy_features')
sys.path.append('/home/paloma/cerebrum-artis/deep-mind/v3_adaptive_gating')
sys.path.append('/home/paloma/cerebrum-artis/deep-mind/v3_1_integrated')

from train_v3 import MultimodalFuzzyClassifier as V3Model, ArtEmisWithFuzzyDataset
from train_v4 import FuzzyGatingClassifier as V4Model
from train_v4_1 import IntegratedFuzzyGatingClassifier as V4_1Model

from torchvision import transforms
from transformers import RobertaTokenizer

def load_model_checkpoint(model_class, checkpoint_path, device):
    """Load model from checkpoint"""
    print(f"Loading {checkpoint_path}...")
    model = model_class().to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model

def get_predictions(model, dataloader, device):
    """Get all predictions and labels from a model"""
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Getting predictions"):
            # Move batch to device
            images = batch['image'].to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            fuzzy_features = batch['fuzzy_features'].to(device)
            labels = batch['label']
            
            # Forward pass (model decides what inputs it needs)
            try:
                # Try full multimodal call
                outputs = model(images, input_ids, attention_mask, fuzzy_features)
            except TypeError:
                # Fallback to image-only if model doesn't accept all args
                try:
                    outputs = model(images, input_ids, attention_mask)
                except TypeError:
                    outputs = model(images)
            
            probs = torch.softmax(outputs, dim=1)
            
            all_probs.append(probs.cpu().numpy())
            all_labels.append(labels.numpy())
    
    all_probs = np.concatenate(all_probs, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    
    return all_probs, all_labels

def calculate_accuracy(probs, labels):
    """Calculate accuracy from probabilities"""
    predictions = np.argmax(probs, axis=1)
    accuracy = np.mean(predictions == labels)
    return accuracy * 100

def ensemble_average(probs_list, weights=None):
    """Simple or weighted average ensemble"""
    if weights is None:
        weights = [1.0 / len(probs_list)] * len(probs_list)
    
    ensemble_probs = np.zeros_like(probs_list[0])
    for prob, weight in zip(probs_list, weights):
        ensemble_probs += weight * prob
    
    return ensemble_probs

def ensemble_voting(probs_list):
    """Hard voting ensemble"""
    votes = np.array([np.argmax(prob, axis=1) for prob in probs_list])
    # Get most common vote for each sample
    from scipy import stats
    ensemble_votes = stats.mode(votes, axis=0, keepdims=False)[0]
    
    # Convert back to probabilities (one-hot)
    num_classes = probs_list[0].shape[1]
    ensemble_probs = np.zeros_like(probs_list[0])
    ensemble_probs[np.arange(len(ensemble_votes)), ensemble_votes] = 1.0
    
    return ensemble_probs

def optimize_weights(probs_list, labels, step=0.05):
    """Find optimal weights using grid search"""
    print("\n🔍 Optimizing weights...")
    best_acc = 0
    best_weights = None
    
    # Grid search (coarse first)
    weights_range = np.arange(0, 1.0 + step, step)
    total_combinations = len(weights_range) ** 2
    
    print(f"Testing {total_combinations} weight combinations...")
    
    for w1 in tqdm(weights_range):
        for w2 in weights_range:
            w3 = 1.0 - w1 - w2
            if w3 < 0 or w3 > 1.0:
                continue
            
            weights = [w1, w2, w3]
            ensemble_probs = ensemble_average(probs_list, weights)
            acc = calculate_accuracy(ensemble_probs, labels)
            
            if acc > best_acc:
                best_acc = acc
                best_weights = weights
    
    return best_weights, best_acc

def main():
    print("=" * 80)
    print("🎯 ENSEMBLE TESTING: V2 + V3 + V3.1")
    print("=" * 80)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")
    
    # Checkpoint paths
    checkpoints = {
        'V2': '/data/paloma/deep-mind-checkpoints/v2_fuzzy_features/checkpoint_best.pt',
        'V3': '/data/paloma/deep-mind-checkpoints/v3_adaptive_gating/checkpoint_best.pt',
        'V3.1': '/data/paloma/deep-mind-checkpoints/v3_1_integrated/checkpoint_best.pt'
    }
    
    # Load models
    print("📦 Loading models...")
    models = {}
    models['V3'] = load_model_checkpoint(V3Model, checkpoints['V3'], device)
    models['V4'] = load_model_checkpoint(V4Model, checkpoints['V4'], device)
    models['V4.1'] = load_model_checkpoint(V4_1Model, checkpoints['V4.1'], device)
    
    # Get dataloader
    print("\n📊 Loading validation data...")
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Use V3's dataset (has fuzzy features)
    val_dataset = ArtEmisWithFuzzyDataset(
        csv_path='/home/paloma/cerebrum-artis/artemis-v2/dataset/official_data/combined_artemis_with_splits.csv',
        image_dir='/data/paloma/data/paintings/wikiart',
        split='val',
        tokenizer=tokenizer,
        transform=transform,
        max_length=128,
        fuzzy_cache_path='/data/paloma/fuzzy_features_cache.pkl'
    )
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)
    
    # Get predictions from each model
    print("\n🔮 Getting predictions from each model...")
    predictions = {}
    labels = None
    
    for name, model in models.items():
        probs, lbls = get_predictions(model, val_loader, device)
        predictions[name] = probs
        if labels is None:
            labels = lbls
        
        # Individual accuracy
        acc = calculate_accuracy(probs, labels)
        print(f"  {name}: {acc:.2f}%")
    
    # Prepare for ensemble
    probs_list = [predictions['V3'], predictions['V4'], predictions['V4.1']]
    
    print("\n" + "=" * 80)
    print("🎲 ENSEMBLE STRATEGIES")
    print("=" * 80)
    
    # 1. Simple Average
    print("\n1️⃣  Simple Average (equal weights)")
    ensemble_probs = ensemble_average(probs_list)
    acc = calculate_accuracy(ensemble_probs, labels)
    print(f"   Accuracy: {acc:.2f}%")
    
    # 2. Voting
    print("\n2️⃣  Hard Voting")
    ensemble_probs = ensemble_voting(probs_list)
    acc = calculate_accuracy(ensemble_probs, labels)
    print(f"   Accuracy: {acc:.2f}%")
    
    # 3. Weighted by individual performance
    print("\n3️⃣  Performance-based Weighted Average")
    v3_acc = calculate_accuracy(predictions['V3'], labels)
    v4_acc = calculate_accuracy(predictions['V4'], labels)
    v4_1_acc = calculate_accuracy(predictions['V4.1'], labels)
    
    # Normalize to sum to 1
    total = v3_acc + v4_acc + v4_1_acc
    perf_weights = [v3_acc/total, v4_acc/total, v4_1_acc/total]
    
    print(f"   Weights: V3={perf_weights[0]:.3f}, V4={perf_weights[1]:.3f}, V4.1={perf_weights[2]:.3f}")
    ensemble_probs = ensemble_average(probs_list, perf_weights)
    acc = calculate_accuracy(ensemble_probs, labels)
    print(f"   Accuracy: {acc:.2f}%")
    
    # 4. Optimized weights
    print("\n4️⃣  Optimized Weighted Average")
    best_weights, best_acc = optimize_weights(probs_list, labels, step=0.05)
    print(f"   Best Weights: V3={best_weights[0]:.3f}, V4={best_weights[1]:.3f}, V4.1={best_weights[2]:.3f}")
    print(f"   Accuracy: {best_acc:.2f}%")
    
    # 5. Only V3 + V4 (without V4.1)
    print("\n5️⃣  V3 + V4 Only (no V4.1)")
    probs_v3_v4 = [predictions['V3'], predictions['V4']]
    
    # Simple average
    ensemble_probs = ensemble_average(probs_v3_v4)
    acc_simple = calculate_accuracy(ensemble_probs, labels)
    print(f"   Simple Average: {acc_simple:.2f}%")
    
    # Optimized weights
    best_weights_v3_v4, best_acc_v3_v4 = optimize_weights(probs_v3_v4, labels, step=0.05)
    print(f"   Optimized Weights: V3={best_weights_v3_v4[0]:.3f}, V4={best_weights_v3_v4[1]:.3f}")
    print(f"   Optimized Accuracy: {best_acc_v3_v4:.2f}%")
    
    # Summary
    print("\n" + "=" * 80)
    print("📊 SUMMARY")
    print("=" * 80)
    print(f"Individual Models:")
    print(f"  V3:   {calculate_accuracy(predictions['V3'], labels):.2f}%")
    print(f"  V3:   {calculate_accuracy(predictions['V3'], labels):.2f}%")
    print(f"  V3.1: {calculate_accuracy(predictions['V3.1'], labels):.2f}%")
    print(f"\nBest V4 Ensemble (3 models):")
    print(f"  Optimized: {best_acc:.2f}%")
    print(f"  Weights: V2={best_weights[0]:.3f}, V3={best_weights[1]:.3f}, V3.1={best_weights[2]:.3f}")
    print(f"\nBest Ensemble (V2 + V3 only):")
    print(f"  Optimized: {best_acc_v3_v4:.2f}%")
    print(f"  Weights: V2={best_weights_v3_v4[0]:.3f}, V3={best_weights_v3_v4[1]:.3f}")
    
    gain_3models = best_acc - max(v3_acc, v4_acc, v4_1_acc)
    gain_2models = best_acc_v3_v4 - max(v3_acc, v4_acc)
    
    print(f"\n💡 Improvement over best single model:")
    print(f"  V4 ensemble (3 models): +{gain_3models:.2f}%")
    print(f"  2-model ensemble: +{gain_2models:.2f}%")
    print("=" * 80)

if __name__ == "__main__":
    main()
