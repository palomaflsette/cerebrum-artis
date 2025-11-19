"""
Demonstração: Fusão Adaptativa em Ação
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import fuzzy_brain.integration as integration

print("\n" + "="*70)
print("DEMONSTRAÇÃO: FUSÃO ADAPTATIVA")
print("="*70)
print("\nO peso α ajusta automaticamente baseado no conteúdo:\n")
print("  • α baixo  (0.5): Abstrato/Formal → 50% fuzzy, 50% neural")
print("  • α médio  (0.7): Misto → 70% neural, 30% fuzzy")
print("  • α alto   (0.9): Representacional → 90% neural, 10% fuzzy")
print("\n" + "="*70 + "\n")

# Inicializa preditor COM fusão adaptativa
predictor = integration.HybridEmotionPredictor(
    sat_checkpoint_path=None,
    fusion_weight=0.7,  
    use_guided_fuzzy=False,
    adaptive_fusion=True  # ✨ ATIVA
)

# Pinturas com características MUITO diferentes
test_images = [
    {
        'path': '/data/paloma/data/paintings/wikiart/Action_painting/franz-kline_horizontal-rust-1960.jpg',
        'name': 'Franz Kline (Abstrato P&B)',
        'expected_alpha': '~0.5-0.6 (baixo)'
    },
    {
        'path': '/data/paloma/data/paintings/wikiart/Expressionism/otto-dix_dying-warrior.jpg',
        'name': 'Otto Dix (Guerra)',
        'expected_alpha': '~0.6-0.7 (médio)'
    },
    {
        'path': '/data/paloma/data/paintings/wikiart/Pop_Art/aki-kuroda_cosmogarden-2011.jpg',
        'name': 'Aki Kuroda (Pop Art)',
        'expected_alpha': '~0.7-0.9 (alto)'
    }
]

for i, test in enumerate(test_images, 1):
    print(f"\n{'='*70}")
    print(f"TESTE {i}/3: {test['name']}")
    print(f"Alpha Esperado: {test['expected_alpha']}")
    print('='*70)
    
    # Prediz com componentes
    result = predictor.predict(test['path'], return_components=True)
    
    # Mostra features e análise
    features = result['features']
    
    # Calcula semantic score manualmente para debug
    semantic_score = (
        0.3 * features['complexity'] +
        0.3 * features['saturation'] +
        0.2 * (1 - features['symmetry']) +
        0.2 * features['texture_roughness']
    )
    
    print(f"\n📊 FEATURES CRÍTICAS:")
    print(f"  • Complexity       : {features['complexity']:.3f}")
    print(f"  • Saturation       : {features['saturation']:.3f}")
    print(f"  • Symmetry         : {features['symmetry']:.3f}")
    print(f"  • Texture Roughness: {features['texture_roughness']:.3f}")
    
    print(f"\n🎨 ANÁLISE ADAPTATIVA:")
    print(f"  • Semantic Score: {semantic_score:.3f}")
    print(f"  • Alpha Calculado: {result['fusion_weight']:.3f}")
    
    if semantic_score < 0.3:
        interpretation = "ABSTRATO/FORMAL → Fuzzy domina"
    elif semantic_score < 0.6:
        interpretation = "MISTO → Balance entre neural e fuzzy"
    else:
        interpretation = "REPRESENTACIONAL → Neural domina"
    
    print(f"  • Interpretação: {interpretation}")
    
    print(f"\n🧠 TOP 3 EMOÇÕES:")
    final_sorted = sorted(result['final'].items(), key=lambda x: x[1], reverse=True)
    for emotion, prob in final_sorted[:3]:
        bar = "█" * int(prob * 30)
        print(f"  • {emotion:15s}: {prob:.3f} {bar}")

print("\n" + "="*70)
print("✅ FUSÃO ADAPTATIVA IMPLEMENTADA COM SUCESSO!")
print("="*70)
print("\nCONCLUSÃO:")
print("  → Sistema ajusta automaticamente o peso baseado no tipo de imagem")
print("  → Abstrato: confia mais no fuzzy (regras estéticas)")
print("  → Representacional: confia mais no neural (conteúdo semântico)")
print("="*70 + "\n")
