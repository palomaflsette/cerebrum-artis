"""
Teste comparativo: Pinturas dramáticas vs minimalista
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

# Import do arquivo integration.py (não é pacote)
import fuzzy_brain.integration as integration

# Inicializa preditor COM fusão adaptativa
print("⏳ Inicializando sistema...")
predictor = integration.HybridEmotionPredictor(
    sat_checkpoint_path=None,
    fusion_weight=0.7,         # Peso padrão (se não usar adaptativo)
    use_guided_fuzzy=False,
    adaptive_fusion=True       # ✨ ATIVA FUSÃO ADAPTATIVA
)

# Testa 3 pinturas com características MUITO diferentes
test_images = [
    {
        'path': '/data/paloma/data/paintings/wikiart/Action_painting/franz-kline_horizontal-rust-1960.jpg',
        'name': 'Franz Kline - Horizontal Rust (1960)',
        'expected': 'Minimalista P&B → CONTENTMENT/AWE'
    },
    {
        'path': '/data/paloma/data/paintings/wikiart/Expressionism/otto-dix_dying-warrior.jpg',
        'name': 'Otto Dix - Dying Warrior',
        'expected': 'Guerra/Morte → SADNESS/FEAR/ANGER'
    },
    {
        'path': '/data/paloma/data/paintings/wikiart/Expressionism/egon-schiele_the-self-seers-death-and-man-1911.jpg',
        'name': 'Egon Schiele - Death and Man (1911)',
        'expected': 'Morte/Existencial → FEAR/SADNESS'
    }
]

for i, test in enumerate(test_images, 1):
    print("\n" + "="*70)
    print(f"TESTE {i}/3: {test['name']}")
    print(f"Expectativa: {test['expected']}")
    print("="*70)
    
    try:
        print(predictor.explain_prediction(test['path']))
    except Exception as e:
        print(f"❌ Erro: {e}")

print("\n" + "="*70)
print("🎯 COMPARAÇÃO COMPLETA!")
print("="*70)
