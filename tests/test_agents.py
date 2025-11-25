#!/usr/bin/env python3
"""
Script de teste para o Agente 2 - Percepto Emocional

Testa a classificação emocional multimodal usando o modelo v1 baseline
treinado (ResNet50 + RoBERTa).
"""

from pathlib import Path
from cerebrum_artis.agents.percepto import PerceptoEmocional


def test_percepto_with_caption():
    """Teste com caption fornecida pelo usuário."""
    print("=" * 80)
    print("🧪 TESTE 1: Percepto Emocional com Caption Fornecida")
    print("=" * 80)

    # Initialize agent
    percepto = PerceptoEmocional()
    print(f"\n{percepto}\n")

    # Test image (Franz Kline painting)
    image_path = "/data/paloma/data/paintings/wikiart/Action_painting/franz-kline_horizontal-rust-1960.jpg"

    if not Path(image_path).exists():
        print(f"⚠️  Imagem de teste não encontrada: {image_path}")
        print("   Pulando teste...")
        return

    # User-provided caption
    caption = "A bold abstract expressionist painting with strong black brushstrokes on white canvas, evoking raw emotion and power."

    print(f"📷 Imagem: {Path(image_path).name}")
    print(f"📝 Caption: {caption}")
    print("\n🔍 Analisando...\n")

    # Analyze
    result = percepto.analyze(
        image=image_path,
        caption=caption,
        return_probabilities=True,
        auto_caption=False
    )

    # Display results
    print("=" * 80)
    print("📊 RESULTADO DA ANÁLISE")
    print("=" * 80)
    print(f"Emoção Dominante: {result['emotion'].upper()}")
    print(f"Confiança: {result['confidence']:.2%}")
    print(f"Caption Source: {result['caption_source']}")
    print(f"Modelo: {result['model']}")
    print(f"Acurácia Val: {result['val_acc']:.2%}")

    print("\n📈 Distribuição de Probabilidades:")
    print("-" * 80)
    sorted_probs = sorted(
        result['probabilities'].items(),
        key=lambda x: x[1],
        reverse=True
    )
    for emotion, prob in sorted_probs:
        bar = "█" * int(prob * 50)  # Bar chart
        print(f"  {emotion:15s} {prob:6.2%}  {bar}")

    print("=" * 80)


def test_percepto_without_caption():
    """Teste sem caption (usa caption padrão)."""
    print("\n\n" + "=" * 80)
    print("🧪 TESTE 2: Percepto Emocional SEM Caption (Default)")
    print("=" * 80)

    # Initialize agent
    percepto = PerceptoEmocional()

    # Test image
    image_path = "/data/paloma/data/paintings/wikiart/Action_painting/franz-kline_horizontal-rust-1960.jpg"

    if not Path(image_path).exists():
        print(f"⚠️  Imagem de teste não encontrada: {image_path}")
        print("   Pulando teste...")
        return

    print(f"📷 Imagem: {Path(image_path).name}")
    print(f"📝 Caption: [PADRÃO] - \"A painting.\"\n")

    print("🔍 Analisando...\n")

    # Analyze WITHOUT caption
    result = percepto.analyze(
        image=image_path,
        caption=None,
        auto_caption=False  # Don't generate, use default
    )

    # Display results
    print("=" * 80)
    print("📊 RESULTADO DA ANÁLISE")
    print("=" * 80)
    print(f"Emoção Dominante: {result['emotion'].upper()}")
    print(f"Confiança: {result['confidence']:.2%}")
    print(f"Caption Usada: {result['caption']}")
    print(f"Caption Source: {result['caption_source']}")

    print("\n📈 Top 3 Emoções:")
    sorted_probs = sorted(
        result['probabilities'].items(),
        key=lambda x: x[1],
        reverse=True
    )[:3]
    for i, (emotion, prob) in enumerate(sorted_probs, 1):
        print(f"  {i}. {emotion}: {prob:.2%}")

    print("=" * 80)


def test_percepto_comparison():
    """Teste comparando diferentes captions para a mesma imagem."""
    print("\n\n" + "=" * 80)
    print("🧪 TESTE 3: Comparação - Diferentes Captions")
    print("=" * 80)

    percepto = PerceptoEmocional()

    image_path = "/data/paloma/data/paintings/wikiart/Action_painting/franz-kline_horizontal-rust-1960.jpg"

    if not Path(image_path).exists():
        print(f"⚠️  Imagem de teste não encontrada: {image_path}")
        print("   Pulando teste...")
        return

    # Different captions for same image
    captions = [
        "A dark and aggressive abstract painting with harsh black strokes.",
        "A peaceful and harmonious composition of balanced forms.",
        "An exciting and energetic explosion of bold brushwork.",
    ]

    print(f"📷 Imagem: {Path(image_path).name}\n")

    results = []
    for i, caption in enumerate(captions, 1):
        print(f"{i}. Caption: {caption}")
        result = percepto.analyze(image_path, caption=caption)
        results.append(result)
        print(f"   → {result['emotion']} ({result['confidence']:.2%})\n")

    print("=" * 80)
    print("🔎 OBSERVAÇÃO: Diferentes captions podem influenciar a emoção detectada!")
    print("=" * 80)


def main():
    """Executa todos os testes."""
    print("\n")
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 15 + "TESTES DO AGENTE 2 - PERCEPTO EMOCIONAL" + " " * 23 + "║")
    print("╚" + "=" * 78 + "╝")
    print()

    try:
        # Test 1: With user caption
        test_percepto_with_caption()

        # Test 2: Without caption
        test_percepto_without_caption()

        # Test 3: Compare different captions
        test_percepto_comparison()

        print("\n\n" + "=" * 80)
        print("✅ TODOS OS TESTES CONCLUÍDOS COM SUCESSO!")
        print("=" * 80)

    except Exception as e:
        print(f"\n\n❌ ERRO durante os testes: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()