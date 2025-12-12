#!/usr/bin/env python3
"""
V4 Ensemble Model: Combines V2 (Fuzzy Features) + V3 (Adaptive Gating)

Strategy: Weighted average of predictions
- V2: Best F1 = 65.77% (strong on fuzzy-interpretable emotions)
- V3: Best F1 = 65.66% (strong on complex multimodal fusion)
- Ensemble: Expected F1 > 66%
"""

import torch
import torch.nn as nn
from pathlib import Path
import sys

# Add paths for V2 and V3 models
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from cerebrum_artis.models.ensemble.model_definitions import (
    MultimodalFuzzyClassifier,
    FuzzyGatingClassifier
)


class EnsembleV4(nn.Module):
    """
    Ensemble of V2 and V3 models using weighted average.
    
    Args:
        v2_checkpoint: Path to V2 best checkpoint
        v3_checkpoint: Path to V3 best checkpoint
        v2_weight: Weight for V2 predictions (0-1)
        device: torch device
    """
    
    def __init__(
        self, 
        v2_checkpoint,
        v3_checkpoint,
        v2_weight=0.5,
        device='cuda'
    ):
        super().__init__()
        
        self.device = device
        self.v2_weight = v2_weight
        self.v3_weight = 1.0 - v2_weight
        
        # Load V2 model
        print(f"📦 Loading V2 from {v2_checkpoint}")
        self.v2_model = MultimodalFuzzyClassifier(num_classes=9)
        v2_state = torch.load(v2_checkpoint, map_location=device)
        self.v2_model.load_state_dict(v2_state['model_state_dict'])
        self.v2_model.to(device)
        self.v2_model.eval()
        v2_f1 = v2_state.get('val_f1', None)
        v2_f1_str = f"{v2_f1:.4f}" if v2_f1 is not None else "N/A"
        print(f"   ✅ V2 loaded (F1: {v2_f1_str})")
        
        # Load V3 model
        print(f"📦 Loading V3 from {v3_checkpoint}")
        self.v3_model = FuzzyGatingClassifier(num_classes=9)
        v3_state = torch.load(v3_checkpoint, map_location=device)
        self.v3_model.load_state_dict(v3_state['model_state_dict'])
        self.v3_model.to(device)
        self.v3_model.eval()
        v3_f1 = v3_state.get('val_f1', None)
        v3_f1_str = f"{v3_f1:.4f}" if v3_f1 is not None else "N/A"
        print(f"   ✅ V3 loaded (F1: {v3_f1_str})")
        
        print(f"\n⚖️  Ensemble weights: V2={v2_weight:.2f}, V3={self.v3_weight:.2f}")
    
    def forward(self, image, input_ids, attention_mask, fuzzy_features):
        """
        Forward pass through ensemble.
        
        Returns:
            final_logits: Weighted average of V2 and V3 predictions
            v2_logits: V2 predictions (for analysis)
            v3_logits: V3 predictions (for analysis)
        """
        with torch.no_grad():
            # V2 prediction
            v2_logits = self.v2_model(image, input_ids, attention_mask, fuzzy_features)
            
            # V3 prediction (note: V3 doesn't use fuzzy features directly)
            v3_logits = self.v3_model(image, input_ids, attention_mask)
        
        # Weighted average (in probability space for better calibration)
        v2_probs = torch.softmax(v2_logits, dim=1)
        v3_probs = torch.softmax(v3_logits, dim=1)
        
        ensemble_probs = self.v2_weight * v2_probs + self.v3_weight * v3_probs
        ensemble_logits = torch.log(ensemble_probs + 1e-8)  # Back to logits
        
        return ensemble_logits, v2_logits, v3_logits
    
    def optimize_weights(self, val_loader, emotions):
        """
        Find optimal V2/V3 weights using validation set.
        
        Grid search over [0.0, 0.1, ..., 1.0] to maximize F1.
        """
        from sklearn.metrics import f1_score
        import numpy as np
        
        print("\n🔍 Optimizing ensemble weights on validation set...")
        
        best_f1 = 0.0
        best_weight = 0.5
        
        for w in np.arange(0.0, 1.1, 0.1):
            self.v2_weight = w
            self.v3_weight = 1.0 - w
            
            all_preds = []
            all_labels = []
            
            for batch in val_loader:
                image = batch['image'].to(self.device)
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                fuzzy = batch['fuzzy_features'].to(self.device)
                labels = batch['label']
                
                logits, _, _ = self.forward(image, input_ids, attention_mask, fuzzy)
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                
                all_preds.extend(preds)
                all_labels.extend(labels.numpy())
            
            f1 = f1_score(all_labels, all_preds, average='macro')
            print(f"   V2 weight={w:.1f}: F1={f1:.4f}")
            
            if f1 > best_f1:
                best_f1 = f1
                best_weight = w
        
        self.v2_weight = best_weight
        self.v3_weight = 1.0 - best_weight
        
        print(f"\n✅ Optimal weights: V2={best_weight:.2f}, V3={1-best_weight:.2f}")
        print(f"   Best F1: {best_f1:.4f}")
        
        return best_weight, best_f1


def load_ensemble(
    v2_checkpoint=None,
    v3_checkpoint=None,
    v2_weight=0.5,
    device='cuda',
    optimize=False,
    val_loader=None
):
    """
    Convenience function to load V4 ensemble.
    
    Args:
        v2_checkpoint: Path to V2 checkpoint (defaults to best)
        v3_checkpoint: Path to V3 checkpoint (defaults to best)
        v2_weight: Initial V2 weight (ignored if optimize=True)
        device: torch device
        optimize: Whether to optimize weights on validation set
        val_loader: DataLoader for optimization (required if optimize=True)
    
    Returns:
        ensemble: EnsembleV4 model
    """
    # Default checkpoint paths
    if v2_checkpoint is None:
        v2_checkpoint = '/data/paloma/deep-mind-checkpoints/v2_fuzzy_features/checkpoint_best.pt'
    
    if v3_checkpoint is None:
        v3_checkpoint = '/data/paloma/deep-mind-checkpoints/v3_adaptive_gating/checkpoint_best.pt'
    
    # Load ensemble
    ensemble = EnsembleV4(
        v2_checkpoint=v2_checkpoint,
        v3_checkpoint=v3_checkpoint,
        v2_weight=v2_weight,
        device=device
    )
    
    # Optimize weights if requested
    if optimize:
        if val_loader is None:
            raise ValueError("val_loader required when optimize=True")
        ensemble.optimize_weights(val_loader, emotions=[
            'amusement', 'awe', 'contentment', 'excitement',
            'anger', 'disgust', 'fear', 'sadness', 'something else'
        ])
    
    return ensemble
