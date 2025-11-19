"""
Debug: Por que o sistema acha que Franz Kline = Contentment?
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from fuzzy_brain.extractors.visual import VisualFeatureExtractor
from fuzzy_brain.fuzzy.variables import FuzzyVariables, fuzzify_value
from fuzzy_brain.fuzzy.system import FuzzyInferenceSystem

# Imagem do Franz Kline
image_path = "/data/paloma/data/paintings/wikiart/Action_painting/franz-kline_horizontal-rust-1960.jpg"

print("="*70)
print("🔍 INVESTIGAÇÃO: Por que CONTENTMENT?")
print("="*70)

# 1. Extrai features
print("\n📊 FEATURES EXTRAÍDAS:")
extractor = VisualFeatureExtractor()
features = extractor.extract_all(image_path)

for feature, value in features.items():
    print(f"  {feature:20s}: {value:.3f}")

# 2. Fuzzifica cada feature para ver termos ativados
print("\n🧠 FUZZIFICAÇÃO (quais termos linguísticos estão ativos):")
fuzzy_vars = FuzzyVariables()

for feature_name, value in features.items():
    var = fuzzy_vars.get_input_variable(feature_name)
    terms = fuzzify_value(value, var)
    
    # Mostra termos com pertinência > 0
    active_terms = {term: pert for term, pert in terms.items() if pert > 0.01}
    
    if active_terms:
        print(f"\n  {feature_name} = {value:.3f}:")
        for term, pert in sorted(active_terms.items(), key=lambda x: -x[1]):
            bar = "█" * int(pert * 20)
            print(f"    • {term:20s}: {pert:5.1%} {bar}")

# 3. Quais regras de CONTENTMENT existem?
print("\n" + "="*70)
print("📋 REGRAS DE CONTENTMENT:")
print("="*70)

from fuzzy_brain.fuzzy.rules import FuzzyRules
fuzzy_rules = FuzzyRules(fuzzy_vars)

contentment_rules = fuzzy_rules.get_rules_by_emotion('contentment')
print(f"\nTotal: {len(contentment_rules)} regras\n")

for rule in contentment_rules:
    print(f"• {rule.label}:")
    print(f"  {rule}\n")

# 4. Análise manual: por que CONTENTMENT_1_balanced pode ter ativado?
print("="*70)
print("🤔 ANÁLISE:")
print("="*70)

print("\nREGRA: CONTENTMENT_1_balanced")
print("  Antecedente: brightness[medio] & saturation[medio_saturado] & color_temp[neutro]")
print("  Valores reais:")
print(f"    brightness = {features['brightness']:.3f}")
print(f"    saturation = {features['saturation']:.3f}")
print(f"    color_temperature = {features['color_temperature']:.3f}")

# Fuzzifica
bright_terms = fuzzify_value(features['brightness'], fuzzy_vars.get_input_variable('brightness'))
sat_terms = fuzzify_value(features['saturation'], fuzzy_vars.get_input_variable('saturation'))
temp_terms = fuzzify_value(features['color_temperature'], fuzzy_vars.get_input_variable('color_temperature'))

print("\n  Pertinências:")
print(f"    brightness[medio] = {bright_terms.get('medio', 0):.1%}")
print(f"    saturation[medio_saturado] = {sat_terms.get('medio_saturado', 0):.1%}")
print(f"    color_temp[neutro] = {temp_terms.get('neutro', 0):.1%}")

activation_1 = min(
    bright_terms.get('medio', 0),
    sat_terms.get('medio_saturado', 0),
    temp_terms.get('neutro', 0)
)
print(f"\n  ✅ Grau de ativação (min): {activation_1:.1%}")

print("\n" + "-"*70)

print("\nREGRA: CONTENTMENT_2_harmony_simple")
print("  Antecedente: color_harmony[harmonico] & complexity[simples]")
print("  Valores reais:")
print(f"    color_harmony = {features['color_harmony']:.3f}")
print(f"    complexity = {features['complexity']:.3f}")

# Fuzzifica
harmony_terms = fuzzify_value(features['color_harmony'], fuzzy_vars.get_input_variable('color_harmony'))
complex_terms = fuzzify_value(features['complexity'], fuzzy_vars.get_input_variable('complexity'))

print("\n  Pertinências:")
print(f"    color_harmony[harmonico] = {harmony_terms.get('harmonico', 0):.1%}")
print(f"    complexity[simples] = {complex_terms.get('simples', 0):.1%}")

activation_2 = min(
    harmony_terms.get('harmonico', 0),
    complex_terms.get('simples', 0)
)
print(f"\n  ✅ Grau de ativação (min): {activation_2:.1%}")

print("\n" + "="*70)
print("💡 CONCLUSÃO:")
print("="*70)

if activation_2 > activation_1:
    print(f"\nA regra CONTENTMENT_2_harmony_simple foi fortemente ativada ({activation_2:.1%})!")
    print("\nPor quê?")
    print(f"  • complexity = {features['complexity']:.3f} → MUITO SIMPLES (composição minimalista)")
    print(f"  • color_harmony = {features['color_harmony']:.3f} → HARMONIOSO (P&B é harmonioso!)")
    print("\n👉 Pinturas abstratas SIMPLES + HARMONIOSAS evocam SERENIDADE/CONTENTAMENTO")
    print("   segundo a teoria estética (Ramachandran & Hirstein, 1999)")
else:
    print(f"\nA regra CONTENTMENT_1_balanced foi ativada ({activation_1:.1%})!")
    print("Cores equilibradas, neutras, médias → contentamento.")

print("\n" + "="*70)
