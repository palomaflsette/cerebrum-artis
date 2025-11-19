#!/usr/bin/env python3
"""
Script rápido para testar o Visual Feature Extractor
"""
import sys
sys.path.insert(0, '/home/paloma/cerebrum-artis/fuzzy-brain')

from fuzzy_brain.extractors.visual import VisualFeatureExtractor
import json

def interpret_value(name, value):
    """Converte valores numéricos em termos linguísticos"""
    interpretations = {
        'brightness': [
            (0.0, 0.2, 'muito escuro 🌑'),
            (0.2, 0.4, 'escuro 🌒'),
            (0.4, 0.6, 'médio ☁️'),
            (0.6, 0.8, 'claro ☀️'),
            (0.8, 1.0, 'muito claro ⚡')
        ],
        'color_temperature': [
            (0.0, 0.3, 'muito frio ❄️ (azul)'),
            (0.3, 0.45, 'frio 🧊'),
            (0.45, 0.55, 'neutro ⚪'),
            (0.55, 0.7, 'quente 🔥'),
            (0.7, 1.0, 'muito quente 🌶️ (vermelho)')
        ],
        'saturation': [
            (0.0, 0.2, 'muito dessaturado ⚫ (cinza)'),
            (0.2, 0.4, 'dessaturado 🔘'),
            (0.4, 0.6, 'médio saturado 🎨'),
            (0.6, 0.8, 'saturado 🌈'),
            (0.8, 1.0, 'muito saturado 💥 (vibrante)')
        ],
        'color_harmony': [
            (0.0, 0.3, 'monocromático 🎭'),
            (0.3, 0.5, 'harmônico limitado 🎨'),
            (0.5, 0.7, 'harmônico diverso 🌈'),
            (0.7, 0.85, 'muito diverso 🎪'),
            (0.85, 1.0, 'caótico 🌀')
        ],
        'complexity': [
            (0.0, 0.2, 'muito simples ⬜'),
            (0.2, 0.4, 'simples 📊'),
            (0.4, 0.6, 'médio 🔲'),
            (0.6, 0.8, 'complexo 🧩'),
            (0.8, 1.0, 'muito complexo 🌀')
        ],
        'symmetry': [
            (0.0, 0.2, 'muito assimétrico 🌊'),
            (0.2, 0.4, 'assimétrico ⚡'),
            (0.4, 0.6, 'levemente simétrico 🔶'),
            (0.6, 0.8, 'simétrico 🦋'),
            (0.8, 1.0, 'muito simétrico 🪞')
        ],
        'roughness': [
            (0.0, 0.2, 'muito liso 🧊'),
            (0.2, 0.4, 'liso 📄'),
            (0.4, 0.6, 'textura média 🧱'),
            (0.6, 0.8, 'áspero 🏔️'),
            (0.8, 1.0, 'muito áspero 🌋')
        ]
    }
    
    ranges = interpretations.get(name, [])
    for min_val, max_val, desc in ranges:
        if min_val <= value < max_val or (max_val == 1.0 and value == 1.0):
            return desc
    return "indefinido"

def test_image(image_path):
    """Testa extração de features de uma imagem"""
    print(f"\n{'='*70}")
    print(f"🎨 TESTANDO: {image_path}")
    print(f"{'='*70}\n")
    
    try:
        # Criar extrator
        extractor = VisualFeatureExtractor()
        
        # Extrair features
        print("⏳ Extraindo features...")
        features = extractor.extract_all(image_path)
        
        # Mostrar resultados
        print("\n📊 FEATURES EXTRAÍDAS (valores numéricos [0,1]):")
        print("-" * 70)
        for name, value in features.items():
            interpretation = interpret_value(name, value)
            print(f"  {name:20s}: {value:5.3f}  →  {interpretation}")
        
        print("\n" + "="*70)
        print("✅ EXTRAÇÃO CONCLUÍDA COM SUCESSO!")
        print("="*70 + "\n")
        
        return features
        
    except Exception as e:
        print(f"\n❌ ERRO: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    # Testar com imagens do dataset oficial WikiArt
    
    print("\n" + "🎯" * 35)
    print("    TESTANDO VISUAL FEATURE EXTRACTOR")
    print("🎯" * 35)
    
    # Pinturas variadas do dataset oficial do ArtEmis (diferentes estilos/cores)
    test_images = [
        # 1. Abstrata P&B (Franz Kline - Action Painting)
        "/data/paloma/data/paintings/wikiart/Action_painting/franz-kline_black-and-white-png.jpg",
        # 2. Impressionismo noturno/escuro (Whistler)
        "/data/paloma/data/paintings/wikiart/Impressionism/james-mcneill-whistler_nocturne-battersea-bridge.jpg",
        # 3. Impressionismo claro/neve (Monet)
        "/data/paloma/data/paintings/wikiart/Impressionism/claude-monet_snow-at-argenteuil-02.jpg",
    ]
    
    results = {}
    for img_path in test_images:
        result = test_image(img_path)
        if result:
            results[img_path] = result
    
    print("\n" + "📈" * 35)
    print("    RESUMO DOS TESTES")
    print("📈" * 35 + "\n")
    print(f"Total de imagens testadas: {len(results)}/{len(test_images)}")
    print(f"Taxa de sucesso: {len(results)/len(test_images)*100:.1f}%\n")
