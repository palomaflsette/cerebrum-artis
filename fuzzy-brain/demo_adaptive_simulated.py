"""
Demonstração: Fusão Adaptativa com predições SIMULADAS
(Para mostrar a diferença quando temos modelo neural)
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
from fuzzy_brain.extractors.visual import VisualFeatureExtractor
from fuzzy_brain.fuzzy.system import FuzzyInferenceSystem

print("\n" + "="*70)
print("DEMONSTRAÇÃO: EFEITO DA FUSÃO ADAPTATIVA")
print("="*70)
print("\nSimulando predições neurais para mostrar como α muda o resultado\n")
print("="*70 + "\n")

# Inicializa componentes
extractor = VisualFeatureExtractor()
fuzzy_system = FuzzyInferenceSystem(use_guided=False)

emotion_names = [
    'amusement', 'awe', 'contentment', 'excitement',
    'anger', 'disgust', 'fear', 'sadness', 'something_else'
]

def simulate_neural_prediction(image_type):
    """Simula predição neural baseada no tipo de imagem."""
    if image_type == 'war':
        # Neural vê pessoas sofrendo → FEAR, SADNESS
        return {
            'fear': 0.35,
            'sadness': 0.30,
            'anger': 0.15,
            'disgust': 0.10,
            'awe': 0.05,
            'contentment': 0.02,
            'excitement': 0.02,
            'amusement': 0.01,
            'something_else': 0.00
        }
    elif image_type == 'death':
        # Neural vê morte, corpos → SADNESS, FEAR
        return {
            'sadness': 0.40,
            'fear': 0.30,
            'disgust': 0.15,
            'anger': 0.05,
            'something_else': 0.05,
            'awe': 0.03,
            'contentment': 0.01,
            'excitement': 0.01,
            'amusement': 0.00
        }
    else:
        # Default genérico
        return {e: 1.0/9 for e in emotion_names}

def compute_adaptive_weight(features):
    """Calcula peso adaptativo (igual ao integration.py)."""
    complexity = features['complexity']
    saturation = features['saturation']
    symmetry = features['symmetry']
    roughness = features['texture_roughness']
    
    semantic_score = (
        0.3 * complexity +
        0.3 * saturation +
        0.2 * (1 - symmetry) +
        0.2 * roughness
    )
    
    alpha_min = 0.5
    alpha_max = 0.9
    alpha = alpha_min + (alpha_max - alpha_min) * semantic_score
    alpha = max(alpha_min, min(alpha_max, alpha))
    
    return alpha, semantic_score

def fuse_predictions(p_neural, p_fuzzy, alpha):
    """Fusão weighted sum."""
    p_final = {}
    for emotion in emotion_names:
        p_final[emotion] = (
            alpha * p_neural[emotion] + 
            (1 - alpha) * p_fuzzy[emotion]
        )
    
    # Normaliza
    total = sum(p_final.values())
    p_final = {e: p / total for e, p in p_final.items()}
    
    return p_final

# Testa com Otto Dix (guerra)
image_path = '/data/paloma/data/paintings/wikiart/Expressionism/otto-dix_dying-warrior.jpg'

print("🎨 IMAGEM: Otto Dix - Dying Warrior (Guerra)")
print("="*70)

# 1. Extrai features
features = extractor.extract_all(image_path)

print(f"\n📊 FEATURES:")
print(f"  Complexity : {features['complexity']:.3f}")
print(f"  Saturation : {features['saturation']:.3f}")
print(f"  Symmetry   : {features['symmetry']:.3f}")
print(f"  Roughness  : {features['texture_roughness']:.3f}")

# 2. Predição fuzzy
p_fuzzy = fuzzy_system.infer(features)

print(f"\n🧠 FUZZY (regras estéticas - vê composição formal):")
fuzzy_sorted = sorted(p_fuzzy.items(), key=lambda x: x[1], reverse=True)
for emotion, prob in fuzzy_sorted[:3]:
    bar = "█" * int(prob * 30)
    print(f"  {emotion:15s}: {prob:.3f} {bar}")

# 3. Predição neural SIMULADA
p_neural = simulate_neural_prediction('war')

print(f"\n🤖 NEURAL SIMULADO (CNN - vê conteúdo semântico):")
neural_sorted = sorted(p_neural.items(), key=lambda x: x[1], reverse=True)
for emotion, prob in neural_sorted[:3]:
    bar = "█" * int(prob * 30)
    print(f"  {emotion:15s}: {prob:.3f} {bar}")

# 4. Calcula alpha adaptativo
alpha, semantic_score = compute_adaptive_weight(features)

print(f"\n🎯 FUSÃO ADAPTATIVA:")
print(f"  Semantic Score: {semantic_score:.3f}")
print(f"  Alpha (peso neural): {alpha:.3f}")
print(f"  → {alpha*100:.0f}% neural + {(1-alpha)*100:.0f}% fuzzy")

# 5. Fusão
p_final = fuse_predictions(p_neural, p_fuzzy, alpha)

print(f"\n✨ RESULTADO FINAL (fusão adaptativa):")
final_sorted = sorted(p_final.items(), key=lambda x: x[1], reverse=True)
for emotion, prob in final_sorted[:3]:
    bar = "█" * int(prob * 30)
    print(f"  {emotion:15s}: {prob:.3f} {bar}")

# COMPARAÇÃO: E se usássemos peso FIXO?
print("\n" + "="*70)
print("📊 COMPARAÇÃO: Peso FIXO vs ADAPTATIVO")
print("="*70)

alpha_fixed = 0.7
p_fixed = fuse_predictions(p_neural, p_fuzzy, alpha_fixed)

print(f"\nCom α FIXO = 0.7:")
fixed_sorted = sorted(p_fixed.items(), key=lambda x: x[1], reverse=True)
for emotion, prob in fixed_sorted[:3]:
    print(f"  {emotion:15s}: {prob:.3f}")

print(f"\nCom α ADAPTATIVO = {alpha:.3f}:")
for emotion, prob in final_sorted[:3]:
    print(f"  {emotion:15s}: {prob:.3f}")

print("\n" + "="*70)
print("💡 CONCLUSÃO:")
print("="*70)
print(f"\nPara esta imagem (semantic_score = {semantic_score:.3f}):")
print(f"  • Fuzzy detecta: {fuzzy_sorted[0][0].upper()} (composição equilibrada)")
print(f"  • Neural detecta: {neural_sorted[0][0].upper()} (conteúdo: guerra)")
print(f"\n  • α adaptativo ({alpha:.2f}) dá mais peso ao NEURAL")
print(f"    → Correto! Imagens com pessoas precisam de análise semântica")
print(f"\n  • Se fosse abstrato (α~0.5), fuzzy teria mais influência")
print(f"    → Também correto! Abstratos são sobre forma, não conteúdo")

print("\n" + "="*70 + "\n")
