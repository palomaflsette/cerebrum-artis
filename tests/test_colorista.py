#!/usr/bin/env python3
"""
Script de teste do Agente 1 - Colorista Quantitativo

Testa a extração de features LAB e inferência fuzzy.
"""

import sys
from pathlib import Path

# Adiciona cerebrum-artis ao path
sys.path.insert(0, str(Path(__file__).parent))

from cerebrum_artis.agents.colorista import ColoristaQuantitativo


def test_colorista():
    """Testa funcionalidade básica do Colorista."""

    print("="*70)
    print("🧪 TESTE: Agente 1 - Colorista Quantitativo (LAB)")
    print("="*70)
    print()

    # Inicializa agente
    print("🚀 Inicializando Colorista Quantitativo...")
    colorista = ColoristaQuantitativo(use_lab=True)
    print(f"✅ {colorista}")
    print()

    # Busca uma imagem de teste do dataset
    dataset_path = Path("/data/paloma/data/paintings/wikiart")

    if not dataset_path.exists():
        print(f"❌ Dataset não encontrado em {dataset_path}")
        print("   Usando caminho alternativo...")
        # Busca em outro local
        test_images = list(Path("artemis-v2/dataset").rglob("*.jpg"))
        if not test_images:
            print("❌ Nenhuma imagem de teste encontrada!")
            return
        test_image = test_images[0]
    else:
        # Pega primeira imagem do dataset
        test_images = list(dataset_path.rglob("*.jpg"))
        if not test_images:
            print(f"❌ Nenhuma imagem em {dataset_path}")
            return
        test_image = test_images[0]

    print(f"🖼️  Imagem de teste: {test_image.name}")
    print()

    # Testa análise
    print("🔍 Analisando imagem...")
    try:
        result = colorista.analyze(
            test_image,
            return_features=True,
            return_probabilities=True
        )

        print("✅ Análise concluída!")
        print()
        print(f"📊 Emoção Dominante: {result['emotion'].upper()}")
        print(f"   Confiança: {result['confidence']:.2%}")
        print(f"   Espaço de Cores: {result['color_space']}")
        print()

        print("🎨 Features LAB:")
        for feature, value in result['features'].items():
            print(f"   • {feature:20s}: {value:.4f}")
        print()

        print("📈 Top 5 Emoções:")
        sorted_probs = sorted(
            result['probabilities'].items(),
            key=lambda x: x[1],
            reverse=True
        )
        for i, (emotion, prob) in enumerate(sorted_probs[:5], 1):
            bar = "█" * int(prob * 30)
            print(f"   {i}. {emotion:15s} {bar:30s} {prob:.2%}")
        print()

        # Testa explicação textual
        print("="*70)
        print("📝 Explicação Textual:")
        print("="*70)
        explanation = colorista.explain(test_image)
        print(explanation)

    except Exception as e:
        print(f"❌ Erro durante análise: {e}")
        import traceback
        traceback.print_exc()
        return

    print()
    print("="*70)
    print("✅ TESTE CONCLUÍDO COM SUCESSO!")
    print("="*70)


if __name__ == "__main__":
    test_colorista()