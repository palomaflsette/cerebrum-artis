#!/usr/bin/env python3
"""
Diagnóstico de viés sistemático nas predições do modelo
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms
from transformers import RobertaTokenizer, RobertaModel
from sklearn.cluster import KMeans
import cv2
from dotenv import load_dotenv

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'cerebrum_artis' / 'models' / 'v2_fuzzy_features'))

from train_v3 import MultimodalFuzzyClassifier as V3Model

class FuzzyGatingClassifier(nn.Module):
    def __init__(self, num_classes=9, fuzzy_dim=7, dropout=0.3, freeze_resnet=True):
        super().__init__()
        resnet = models.resnet50(weights='IMAGENET1K_V1')
        self.visual_encoder = nn.Sequential(*list(resnet.children())[:-1])
        if freeze_resnet:
            for param in self.visual_encoder.parameters():
                param.requires_grad = False
        self.text_encoder = RobertaModel.from_pretrained('roberta-base')
        self.classifier = nn.Sequential(
            nn.Linear(2048 + 768, 1024),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, image, text_input_ids, text_attention_mask, fuzzy_features=None):
        visual_feats = self.visual_encoder(image).view(image.size(0), -1)
        text_output = self.text_encoder(input_ids=text_input_ids, attention_mask=text_attention_mask)
        text_feats = text_output.last_hidden_state[:, 0, :]
        combined = torch.cat([visual_feats, text_feats], dim=1)
        return self.classifier(combined)

EMOTION_LABELS = [
    'amusement', 'awe', 'contentment', 'excitement',
    'anger', 'disgust', 'fear', 'sadness', 'something else'
]

def extract_color_palette_lab(image_pil, n_colors=6):
    img_array = np.array(image_pil.resize((150, 150)))
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    
    lab[:,:,0] = lab[:,:,0] * (100.0 / 255.0)
    lab[:,:,1] = lab[:,:,1] - 128.0
    lab[:,:,2] = lab[:,:,2] - 128.0
    
    pixels_lab = lab.reshape(-1, 3)
    kmeans = KMeans(n_clusters=n_colors, random_state=42, n_init=10)
    kmeans.fit(pixels_lab)
    
    colors_lab = kmeans.cluster_centers_
    labels = kmeans.labels_
    proportions = np.bincount(labels) / len(labels)
    
    sorted_indices = np.argsort(proportions)[::-1]
    colors_lab = colors_lab[sorted_indices]
    proportions = proportions[sorted_indices]
    
    return colors_lab, proportions

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Load models
    load_dotenv()
    CHECKPOINT_BASE = Path(os.getenv('CHECKPOINT_BASE'))
    
    print("\nCarregando modelos...")
    checkpoint_v3 = torch.load(CHECKPOINT_BASE / 'v2_fuzzy_features' / 'checkpoint_best.pt', map_location=device)
    model_v3 = V3Model(num_classes=9)
    model_v3.load_state_dict(checkpoint_v3['model_state_dict'])
    model_v3 = model_v3.to(device).eval()
    
    checkpoint_v4 = torch.load(CHECKPOINT_BASE / 'v3_adaptive_gating' / 'checkpoint_best.pt', map_location=device)
    model_v4 = FuzzyGatingClassifier(num_classes=9, fuzzy_dim=7)
    model_v4.load_state_dict(checkpoint_v4['model_state_dict'])
    model_v4 = model_v4.to(device).eval()
    
    checkpoint_v41 = torch.load(CHECKPOINT_BASE / 'v3_1_integrated' / 'checkpoint_best.pt', map_location=device)
    model_v41 = FuzzyGatingClassifier(num_classes=9, fuzzy_dim=7)
    model_v41.load_state_dict(checkpoint_v41['model_state_dict'])
    model_v41 = model_v41.to(device).eval()
    
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Test images
    wikiart_test_dir = Path('/data/paloma/data/paintings/wikiart/')
    styles_to_test = ['Fauvism', 'Impressionism', 'Baroque', 'Romanticism', 'Abstract_Expressionism']
    
    results = []
    
    print("\nProcessando imagens...")
    for style in styles_to_test:
        style_dir = wikiart_test_dir / style
        if not style_dir.exists():
            print(f"  Skipping {style} (not found)")
            continue
        
        images = list(style_dir.rglob('*.jpg'))[:10]
        print(f"  {style}: {len(images)} imagens")
        
        for img_path in images:
            try:
                img = Image.open(img_path).convert('RGB')
                img_tensor = transform(img).unsqueeze(0).to(device)
                
                colors_lab, props = extract_color_palette_lab(img, n_colors=6)
                avg_L = np.average(colors_lab[:, 0], weights=props)
                avg_a = np.average(colors_lab[:, 1], weights=props)
                avg_b = np.average(colors_lab[:, 2], weights=props)
                chroma = np.sqrt(avg_a**2 + avg_b**2)
                
                brightness = avg_L / 100.0
                color_temp = (avg_a + 127) / 254.0
                saturation = chroma / 100.0
                color_harmony = np.std(colors_lab[:, 1:3]) / 50.0
                complexity = len(np.unique(colors_lab.astype(int), axis=0)) / 10.0
                symmetry = 0.5
                texture = np.mean(np.std(colors_lab, axis=0)) / 50.0
                
                fuzzy = torch.tensor([[
                    brightness, color_temp, saturation, color_harmony,
                    complexity, symmetry, texture
                ]], dtype=torch.float32).to(device)
                
                caption_neutral = "A painting"
                tokens = tokenizer(caption_neutral, max_length=128, padding='max_length',
                                  truncation=True, return_tensors='pt')
                input_ids = tokens['input_ids'].to(device)
                attention_mask = tokens['attention_mask'].to(device)
                
                with torch.no_grad():
                    logits_v3 = model_v3(img_tensor, input_ids, attention_mask, fuzzy)
                    logits_v4 = model_v4(img_tensor, input_ids, attention_mask, fuzzy)
                    logits_v41 = model_v41(img_tensor, input_ids, attention_mask, fuzzy)
                    
                    probs_v3 = torch.softmax(logits_v3, dim=1)[0]
                    probs_v4 = torch.softmax(logits_v4, dim=1)[0]
                    probs_v41 = torch.softmax(logits_v41, dim=1)[0]
                    
                    ensemble_probs = (probs_v3 + probs_v4 + probs_v41) / 3.0
                    pred_idx = torch.argmax(ensemble_probs).item()
                    confidence = ensemble_probs[pred_idx].item()
                
                results.append({
                    'style': style,
                    'filename': img_path.name,
                    'emotion': EMOTION_LABELS[pred_idx],
                    'confidence': confidence,
                    'brightness': brightness,
                    'saturation': saturation,
                    'complexity': complexity
                })
                
            except Exception as e:
                print(f"    Error {img_path.name}: {e}")
                continue
    
    df = pd.DataFrame(results)
    
    # Analysis
    print("\n" + "="*80)
    print("DIAGNÓSTICO DE VIÉS SISTEMÁTICO")
    print("="*80)
    
    print("\n1. DISTRIBUIÇÃO DE PREDIÇÕES:")
    print("-"*80)
    emotion_counts = df['emotion'].value_counts()
    for emotion, count in emotion_counts.items():
        percentage = (count / len(df)) * 100
        print(f"  {emotion:20s} {count:3d} ({percentage:5.1f}%)")
    
    print("\n2. ESTATÍSTICAS DE CONFIANÇA:")
    print("-"*80)
    print(f"  Média:     {df['confidence'].mean():.3f}")
    print(f"  Mediana:   {df['confidence'].median():.3f}")
    print(f"  Desvio:    {df['confidence'].std():.3f}")
    print(f"  Mínimo:    {df['confidence'].min():.3f}")
    print(f"  Máximo:    {df['confidence'].max():.3f}")
    
    print("\n3. CONFIANÇA POR EMOÇÃO:")
    print("-"*80)
    conf_by_emotion = df.groupby('emotion')['confidence'].agg(['mean', 'std', 'count'])
    for emotion in conf_by_emotion.index:
        stats = conf_by_emotion.loc[emotion]
        print(f"  {emotion:20s} μ={stats['mean']:.3f}  σ={stats['std']:.3f}  n={int(stats['count'])}")
    
    print("\n4. PREDIÇÕES COM CONFIANÇA > 50%:")
    print("-"*80)
    high_conf = df[df['confidence'] > 0.5]
    print(f"  Total: {len(high_conf)} / {len(df)} ({len(high_conf)/len(df)*100:.1f}%)")
    if len(high_conf) > 0:
        print(f"\n  Exemplos:")
        for _, row in high_conf.head(5).iterrows():
            print(f"    {row['style']:25s} {row['emotion']:15s} {row['confidence']:.1%}")
    
    print("\n5. ANÁLISE POR ESTILO:")
    print("-"*80)
    for style in df['style'].unique():
        style_data = df[df['style'] == style]
        top_emotion = style_data['emotion'].value_counts().iloc[0]
        top_count = style_data['emotion'].value_counts().values[0]
        avg_conf = style_data['confidence'].mean()
        print(f"  {style:30s} n={len(style_data):2d}  "
              f"Predominante: {style_data['emotion'].value_counts().index[0]:15s} ({top_count}/{len(style_data)})  "
              f"Conf média: {avg_conf:.3f}")
    
    print("\n" + "="*80)
    print("CONCLUSÕES:")
    print("="*80)
    
    if df['confidence'].mean() < 0.3:
        print("⚠️  CONFIANÇA MÉDIA MUITO BAIXA (<30%)")
        print("    → Modelo está INCERTO na maioria das predições")
        print("    → Predições são estatisticamente questionáveis")
    
    if len(emotion_counts) <= 3:
        print("\n⚠️  VIÉS FORTE: Apenas 3 emoções sendo preditas")
        print("    → Modelo colapsou para poucas categorias")
        print("    → Não está utilizando toda a taxonomia")
    
    if emotion_counts.iloc[0] > len(df) * 0.4:
        print(f"\n⚠️  VIÉS EXTREMO: '{emotion_counts.index[0]}' representa {emotion_counts.iloc[0]/len(df)*100:.1f}%")
        print("    → Modelo tem forte preferência por uma categoria")
        print("    → Possível overfitting ou dataset bias")
    
    print("\n" + "="*80)
    
    # Save results
    output_path = project_root / 'results' / 'diagnostic_bias.csv'
    output_path.parent.mkdir(exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"\nResultados salvos em: {output_path}")

if __name__ == '__main__':
    main()
