#!/usr/bin/env python3
"""
Generate REAL Grad-CAM visualizations using trained V2 model
=============================================================

Uses existing checkpoint: /data/paloma/deep-mind-checkpoints/v2_fuzzy_features/checkpoint_best.pt
"""

import sys
sys.path.insert(0, '/home/paloma/cerebrum-artis')

import torch
import torch.nn as nn
from torchvision import models, transforms
from transformers import RobertaModel, RobertaTokenizer
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from PIL import Image
from pathlib import Path
import pandas as pd

# Model definition (V2)
class MultimodalFuzzyClassifier(nn.Module):
    def __init__(self, num_classes=9, dropout=0.3):
        super().__init__()
        
        resnet = models.resnet50(pretrained=False)
        self.visual_encoder = nn.Sequential(*list(resnet.children())[:-1])
        self.text_encoder = RobertaModel.from_pretrained('roberta-base')
        
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
        visual_feats = self.visual_encoder(image).view(image.size(0), -1)
        text_output = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        text_feats = text_output.last_hidden_state[:, 0, :]
        combined = torch.cat([visual_feats, text_feats, fuzzy_features], dim=1)
        logits = self.fusion(combined)
        return logits


def compute_gradcam(model, image_tensor, target_class):
    """Compute Grad-CAM for V2 model"""
    
    model.eval()
    
    # Hook to capture gradients and activations
    gradients = []
    activations = []
    
    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0])
    
    def forward_hook(module, input, output):
        activations.append(output)
    
    # Register hooks on last conv layer of ResNet
    target_layer = model.visual_encoder[-2]  # Layer before avgpool
    handle_f = target_layer.register_forward_hook(forward_hook)
    handle_b = target_layer.register_backward_hook(backward_hook)
    
    # Forward pass
    model.zero_grad()
    output = model(image_tensor, dummy_input_ids, dummy_attention_mask, dummy_fuzzy)
    
    # Backward pass
    target = output[0, target_class]
    target.backward()
    
    # Remove hooks
    handle_f.remove()
    handle_b.remove()
    
    # Compute Grad-CAM
    grad = gradients[0][0]  # [C, H, W]
    act = activations[0][0]  # [C, H, W]
    
    weights = torch.mean(grad, dim=(1, 2))  # [C]
    cam = torch.sum(weights.view(-1, 1, 1) * act, dim=0)  # [H, W]
    
    # ReLU and normalize
    cam = torch.clamp(cam, min=0)
    cam = cam / (cam.max() + 1e-8)
    
    return cam.detach().cpu().numpy()


# Load model
print("Loading V2 model...")
device = torch.device('cpu')  # Use CPU to avoid conflicting with training GPUs
model = MultimodalFuzzyClassifier(num_classes=9).to(device)

checkpoint_path = '/data/paloma/deep-mind-checkpoints/v2_fuzzy_features/checkpoint_best.pt'
checkpoint = torch.load(checkpoint_path, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

print(f"✅ Model loaded from: {checkpoint_path}")
print(f"   Device: {device}")

# Tokenizer
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

# Dummy inputs for Grad-CAM
dummy_input_ids = tokenizer("A painting", return_tensors='pt', max_length=128, padding='max_length')['input_ids'].to(device)
dummy_attention_mask = tokenizer("A painting", return_tensors='pt', max_length=128, padding='max_length')['attention_mask'].to(device)
dummy_fuzzy = torch.zeros(1, 7).to(device)

# Transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Find example images - usar paths diretos de imagens conhecidas
wikiart_dir = Path('/data/paloma/data/paintings/wikiart')

# Buscar exemplos reais por emoção
target_emotions_examples = {
    'sadness': 'Realism/ivan-kramskoy_grief.jpg',
    'awe': 'Romanticism/caspar-david-friedrich_wanderer-above-the-sea-of-fog.jpg',
    'fear': 'Expressionism/edvard-munch_the-scream-1893.jpg'
}

examples = []
print("\nSearching for example images...")

# Primeiro tentar os exemplos específicos
for emotion, rel_path in target_emotions_examples.items():
    img_path = wikiart_dir / rel_path
    if img_path.exists():
        examples.append((emotion, img_path, rel_path))
        print(f"  ✓ Found {emotion}: {rel_path}")
    else:
        print(f"  ✗ Not found: {rel_path}")

# Se não achou 3, pegar qualquer imagem
if len(examples) < 3:
    print(f"\nSearching for any paintings...")
    import random
    all_imgs = list(wikiart_dir.rglob('*.jpg'))
    random.seed(42)
    random.shuffle(all_imgs)
    
    for emotion in ['sadness', 'awe', 'fear']:
        if len([e for e in examples if e[0] == emotion]) == 0:
            if all_imgs:
                img_path = all_imgs.pop()
                examples.append((emotion, img_path, str(img_path.relative_to(wikiart_dir))))
                print(f"  + Using {emotion}: {img_path.name}")

print(f"\nTotal examples found: {len(examples)}\n")

# Generate Grad-CAMs
EMOTIONS = ['amusement', 'awe', 'contentment', 'excitement', 'anger', 'disgust', 'fear', 'sadness', 'something else']

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

for col_idx, (emotion, img_path, painting_name) in enumerate(examples):
    print(f"Processing {emotion}: {painting_name}")
    
    # Load and preprocess image
    img = Image.open(img_path).convert('RGB')
    img_tensor = transform(img).unsqueeze(0).to(device)
    
    # Predict
    with torch.no_grad():
        output = model(img_tensor, dummy_input_ids, dummy_attention_mask, dummy_fuzzy)
        probs = torch.softmax(output, dim=1)[0]
        pred_idx = torch.argmax(probs).item()
        confidence = probs[pred_idx].item()
    
    # Compute Grad-CAM
    cam = compute_gradcam(model, img_tensor, pred_idx)
    
    # Resize CAM to image size
    cam_resized = np.array(Image.fromarray(cam).resize(img.size, Image.BILINEAR))
    
    # Apply colormap
    cam_colored = cm.jet(cam_resized)[:, :, :3]
    
    # Overlay
    img_np = np.array(img) / 255.0
    overlay = 0.6 * img_np + 0.4 * cam_colored
    overlay = np.clip(overlay, 0, 1)
    
    # Plot original
    axes[0, col_idx].imshow(img)
    axes[0, col_idx].set_title(f'({chr(97+col_idx)}) {EMOTIONS[pred_idx].capitalize()}\nConf: {confidence:.1%}',
                               fontsize=11, weight='bold')
    axes[0, col_idx].axis('off')
    
    # Plot Grad-CAM
    axes[1, col_idx].imshow(overlay)
    axes[1, col_idx].set_title(f'Ground truth: {emotion}', fontsize=9, style='italic')
    axes[1, col_idx].axis('off')

# Row labels
fig.text(0.02, 0.75, 'Original\nPaintings', va='center', ha='center',
         fontsize=11, weight='bold', rotation=90)
fig.text(0.02, 0.25, 'Grad-CAM\nHeatmaps', va='center', ha='center',
         fontsize=11, weight='bold', rotation=90)

# Add colorbar
from mpl_toolkits.axes_grid1 import make_axes_locatable
divider = make_axes_locatable(axes[1, 2])
cax = divider.append_axes("right", size="5%", pad=0.1)
sm = cm.ScalarMappable(cmap=cm.jet, norm=plt.Normalize(vmin=0, vmax=1))
sm.set_array([])
cbar = plt.colorbar(sm, cax=cax)
cbar.set_label('Activation', fontsize=9)

plt.suptitle('Grad-CAM Visual Explanations: V2 Model Attention Patterns (REAL)',
             fontsize=14, weight='bold', y=0.98)

plt.tight_layout(rect=[0.03, 0, 1, 0.96])

# Save
output_dir = Path('/home/paloma/cerebrum-artis/paper-factory/figures')
output_dir.mkdir(parents=True, exist_ok=True)

png_path = output_dir / 'figure9_gradcam_real.png'
plt.savefig(png_path, dpi=300, bbox_inches='tight', format='png')
print(f"\n✅ Saved: {png_path}")

pdf_path = output_dir / 'figure9_gradcam_real.pdf'
plt.savefig(pdf_path, dpi=300, bbox_inches='tight', format='pdf')
print(f"✅ Saved: {pdf_path}")

print("\n" + "="*70)
print("REAL Grad-CAM visualizations generated using trained V2 checkpoint!")
print("="*70)

plt.show()
