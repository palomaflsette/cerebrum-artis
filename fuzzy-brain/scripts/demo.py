#!/usr/bin/env python
"""
Script de demonstração do Visual Feature Extractor.

Este script mostra como usar o extrator de features visuais em uma imagem
e visualiza os resultados de forma interpretável.

Uso:
    python scripts/demo.py <caminho_para_imagem>
"""

import sys
import os
from pathlib import Path

# Adiciona o diretório raiz ao path
sys.path.insert(0, str(Path(__file__).parent.parent))

from fuzzy_brain.extractors.visual import VisualFeatureExtractor


def interpret_value(feature_name: str, value: float) -> str:
    """
    Converte valor numérico em interpretação linguística.
    
    Esta função faz uma "pré-fuzzificação" manual para facilitar entendimento.
    """
    interpretations = {
        'brightness': [
            (0.0, 0.2, "muito escuro 🌑"),
            (0.2, 0.4, "escuro 🌘"),
            (0.4, 0.6, "médio ☁️"),
            (0.6, 0.8, "claro ☀️"),
            (0.8, 1.0, "muito claro ✨"),
        ],
        'color_temperature': [
            (0.0, 0.3, "muito frio ❄️ (azul/verde)"),
            (0.3, 0.45, "frio 🧊"),
            (0.45, 0.55, "neutro ⚖️"),
            (0.55, 0.7, "quente 🔥"),
            (0.7, 1.0, "muito quente 🌶️ (vermelho/amarelo)"),
        ],
        'saturation': [
            (0.0, 0.2, "dessaturado (preto e branco) ⬛"),
            (0.2, 0.4, "pouco saturado 🌫️"),
            (0.4, 0.6, "moderadamente saturado 🎨"),
            (0.6, 0.8, "saturado 🌈"),
            (0.8, 1.0, "muito saturado (cores vibrantes) 💥"),
        ],
        'color_harmony': [
            (0.0, 0.3, "dissonante (muitas cores) 🎪"),
            (0.3, 0.5, "pouco harmônico 🌀"),
            (0.5, 0.7, "harmônico 🎼"),
            (0.7, 1.0, "muito harmônico (paleta unificada) 🎵"),
        ],
        'complexity': [
            (0.0, 0.2, "muito simples (minimalista) ➖"),
            (0.2, 0.4, "simples 📄"),
            (0.4, 0.6, "moderadamente complexo 📰"),
            (0.6, 0.8, "complexo 🗺️"),
            (0.8, 1.0, "muito complexo (muito detalhe) 🧩"),
        ],
        'symmetry': [
            (0.0, 0.3, "assimétrico ↗️"),
            (0.3, 0.5, "pouco simétrico ⚡"),
            (0.5, 0.7, "moderadamente simétrico ⚖️"),
            (0.7, 0.9, "simétrico 🦋"),
            (0.9, 1.0, "perfeitamente simétrico 🔲"),
        ],
        'texture_roughness': [
            (0.0, 0.2, "muito suave (sfumato) 🧈"),
            (0.2, 0.4, "suave 🌊"),
            (0.4, 0.6, "textura média 🪨"),
            (0.6, 0.8, "áspero (pinceladas visíveis) 🎨"),
            (0.8, 1.0, "muito áspero (impasto) 🖌️"),
        ],
    }
    
    ranges = interpretations.get(feature_name, [])
    for low, high, description in ranges:
        if low <= value < high:
            return description
    
    # Fallback para última categoria se value == 1.0
    if ranges:
        return ranges[-1][2]
    
    return "?"


def main():
    if len(sys.argv) < 2:
        print("❌ Erro: Forneça o caminho de uma imagem")
        print(f"\nUso: python {sys.argv[0]} <caminho_da_imagem>")
        print("\nExemplo:")
        print(f"  python {sys.argv[0]} ../artemis/dataset/sample_painting.jpg")
        sys.exit(1)
    
    image_path = sys.argv[1]
    
    if not os.path.exists(image_path):
        print(f"❌ Erro: Arquivo não encontrado: {image_path}")
        sys.exit(1)
    
    print("\n" + "=" * 70)
    print("🎨 ANÁLISE DE FEATURES VISUAIS - FUZZY-BRAIN")
    print("=" * 70)
    print(f"\n📂 Imagem: {image_path}\n")
    
    # Extrai features
    print("⏳ Extraindo features...")
    extractor = VisualFeatureExtractor()
    
    try:
        features = extractor.extract_all(image_path)
    except Exception as e:
        print(f"❌ Erro ao processar imagem: {e}")
        sys.exit(1)
    
    print("✅ Features extraídas com sucesso!\n")
    
    # Exibe resultados
    print("=" * 70)
    print("RESULTADOS")
    print("=" * 70)
    
    feature_order = [
        'brightness',
        'color_temperature',
        'saturation',
        'color_harmony',
        'complexity',
        'symmetry',
        'texture_roughness'
    ]
    
    feature_names = {
        'brightness': 'Brilho',
        'color_temperature': 'Temperatura de Cor',
        'saturation': 'Saturação',
        'color_harmony': 'Harmonia Cromática',
        'complexity': 'Complexidade Visual',
        'symmetry': 'Simetria',
        'texture_roughness': 'Aspereza de Textura'
    }
    
    for key in feature_order:
        value = features[key]
        name = feature_names[key]
        interpretation = interpret_value(key, value)
        
        # Cria barra visual
        bar_length = int(value * 40)
        bar = "█" * bar_length + "░" * (40 - bar_length)
        
        print(f"\n{name}:")
        print(f"  Valor: {value:.4f}")
        print(f"  [{bar}]")
        print(f"  → {interpretation}")
    
    print("\n" + "=" * 70)
    print("\n💡 INTERPRETAÇÃO PARA LÓGICA FUZZY:")
    print("=" * 70)
    
    # Análise simplificada
    print("\nBaseado nestas features, a imagem sugere:")
    
    # Regras simples de interpretação
    suggestions = []
    
    if features['brightness'] < 0.3:
        suggestions.append("• Emoções negativas (tristeza, medo) devido ao baixo brilho")
    elif features['brightness'] > 0.7:
        suggestions.append("• Emoções positivas (alegria, esperança) devido ao alto brilho")
    
    if features['color_temperature'] < 0.4:
        suggestions.append("• Cores frias podem evocar calma ou melancolia")
    elif features['color_temperature'] > 0.6:
        suggestions.append("• Cores quentes podem evocar energia ou paixão")
    
    if features['saturation'] > 0.7:
        suggestions.append("• Alta saturação sugere excitação ou vivacidade")
    elif features['saturation'] < 0.3:
        suggestions.append("• Baixa saturação pode evocar nostalgia ou sobriedade")
    
    if features['color_harmony'] > 0.7:
        suggestions.append("• Alta harmonia pode evocar contentamento ou admiração")
    
    if features['complexity'] > 0.7:
        suggestions.append("• Alta complexidade pode evocar admiração ou confusão")
    elif features['complexity'] < 0.3:
        suggestions.append("• Baixa complexidade pode evocar calma ou monotonia")
    
    if features['symmetry'] > 0.7:
        suggestions.append("• Alta simetria frequentemente evoca admiração")
    
    if suggestions:
        for s in suggestions:
            print(s)
    else:
        print("• Características visuais equilibradas/neutras")
    
    print("\n" + "=" * 70)
    print("\n✨ Próximo passo: Usar estas features no sistema fuzzy!")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
