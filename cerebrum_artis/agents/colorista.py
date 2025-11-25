"""
Agente 1: Colorista Quantitativo

Responsável pela extração de features visuais quantitativas usando:
- Espaço de cores LAB (perceptualmente uniforme)
- Lógica Fuzzy (18 regras Mamdani)
- 7 features visuais: brightness, color_temperature, saturation,
  color_harmony, complexity, symmetry, texture_roughness

Este agente NÃO utiliza contexto linguístico - apenas análise visual pura.
"""

import sys
from pathlib import Path
from typing import Dict, Union, Optional
import numpy as np

# Importa módulos do fuzzy-brain
FUZZY_BRAIN_PATH = Path(__file__).parent.parent.parent / "fuzzy-brain"
sys.path.insert(0, str(FUZZY_BRAIN_PATH))

from fuzzy_brain.feature_extractor_lab import LABFeatureExtractor
from fuzzy_brain.fuzzy.system import FuzzyInferenceSystem


class ColoristaQuantitativo:
    """
    Agente 1: Colorista Quantitativo

    Extrai features visuais em LAB e infere emoções usando lógica fuzzy.

    Pipeline:
    1. Carrega imagem
    2. Converte RGB → LAB
    3. Extrai 7 features quantitativas
    4. Aplica 18 regras fuzzy
    5. Retorna distribuição de probabilidades sobre 9 emoções

    Emoções:
        ['amusement', 'awe', 'contentment', 'excitement',
         'anger', 'disgust', 'fear', 'sadness', 'something else']

    Exemplo:
        >>> colorista = ColoristaQuantitativo()
        >>> result = colorista.analyze("path/to/painting.jpg")
        >>> print(result['emotion'])  # Emoção dominante
        'awe'
        >>> print(result['confidence'])  # Confiança [0, 1]
        0.68
        >>> print(result['features'])  # 7 features LAB
        {'brightness': 0.72, 'saturation': 0.45, ...}
    """

    def __init__(
        self,
        target_size: tuple = (256, 256),
        use_lab: bool = True
    ):
        """
        Inicializa o Colorista Quantitativo.

        Args:
            target_size: Tamanho para redimensionar imagens (H, W)
            use_lab: Se True, usa LAB. Se False, usa RGB (não recomendado)
        """
        self.target_size = target_size
        self.use_lab = use_lab

        # Inicializa extrator de features
        if use_lab:
            self.feature_extractor = LABFeatureExtractor(target_size=target_size)
        else:
            # Fallback para RGB (mantido por compatibilidade)
            from fuzzy_brain.extractors.visual import VisualFeatureExtractor
            self.feature_extractor = VisualFeatureExtractor(target_size=target_size)

        # Inicializa sistema fuzzy
        self.fuzzy_system = FuzzyInferenceSystem()

        # Lista de emoções (ordem do ArtEmis)
        self.emotions = [
            'amusement', 'awe', 'contentment', 'excitement',
            'anger', 'disgust', 'fear', 'sadness', 'something else'
        ]

    def analyze(
        self,
        image: Union[str, Path, np.ndarray],
        return_features: bool = True,
        return_probabilities: bool = True
    ) -> Dict:
        """
        Analisa uma obra de arte e retorna emoção + features.

        Args:
            image: Caminho da imagem ou array numpy RGB
            return_features: Se True, inclui features visuais no resultado
            return_probabilities: Se True, inclui distribuição de probabilidades

        Returns:
            Dict com:
                - emotion: str - Emoção dominante
                - confidence: float - Confiança [0, 1]
                - features: Dict[str, float] - 7 features LAB (se return_features=True)
                - probabilities: Dict[str, float] - Distribuição sobre 9 emoções
                - color_space: str - Espaço de cores usado ('LAB' ou 'RGB')

        Raises:
            FileNotFoundError: Se imagem não existe
            ValueError: Se imagem inválida
        """
        # 1. Extrai features visuais
        try:
            features = self.feature_extractor.extract(image)
        except FileNotFoundError:
            raise FileNotFoundError(f"Imagem não encontrada: {image}")
        except Exception as e:
            raise ValueError(f"Erro ao processar imagem: {e}")

        # 2. Inferência fuzzy
        emotion_probs = self.fuzzy_system.infer(features)

        # 3. Identifica emoção dominante
        dominant_emotion = max(emotion_probs.items(), key=lambda x: x[1])
        emotion_name, confidence = dominant_emotion

        # 4. Monta resultado
        result = {
            'emotion': emotion_name,
            'confidence': confidence,
            'color_space': 'LAB' if self.use_lab else 'RGB',
        }

        if return_features:
            result['features'] = features

        if return_probabilities:
            result['probabilities'] = emotion_probs

        return result

    def batch_analyze(
        self,
        images: list,
        verbose: bool = False
    ) -> list:
        """
        Analisa múltiplas imagens em batch.

        Args:
            images: Lista de caminhos ou arrays
            verbose: Se True, mostra progresso

        Returns:
            Lista de dicts com resultados
        """
        results = []

        for i, image in enumerate(images):
            if verbose and (i + 1) % 100 == 0:
                print(f"Processadas {i + 1}/{len(images)} imagens...")

            try:
                result = self.analyze(image)
                results.append(result)
            except Exception as e:
                print(f"Erro ao processar imagem {i}: {e}")
                results.append(None)

        return results

    def get_feature_importance(self, features: Dict[str, float]) -> Dict[str, float]:
        """
        Calcula importância relativa de cada feature para a emoção detectada.

        Args:
            features: Dict com 7 features extraídas

        Returns:
            Dict com importância [0, 1] de cada feature
        """
        # Normaliza valores para comparação
        total = sum(features.values())
        if total == 0:
            return {k: 0.0 for k in features}

        return {
            feature: value / total
            for feature, value in features.items()
        }

    def explain(self, image: Union[str, Path, np.ndarray]) -> str:
        """
        Gera explicação textual da análise emocional.

        Args:
            image: Caminho da imagem ou array numpy

        Returns:
            String com explicação human-readable
        """
        result = self.analyze(image, return_features=True, return_probabilities=True)

        # Monta explicação
        explanation = []
        explanation.append(f"🎨 Análise Emocional (Espaço {result['color_space']})")
        explanation.append(f"\n📊 Emoção Dominante: {result['emotion'].upper()}")
        explanation.append(f"   Confiança: {result['confidence']:.1%}")

        explanation.append(f"\n🔍 Features Visuais:")
        for feature, value in result['features'].items():
            explanation.append(f"   • {feature}: {value:.3f}")

        explanation.append(f"\n📈 Distribuição de Emoções:")
        sorted_probs = sorted(
            result['probabilities'].items(),
            key=lambda x: x[1],
            reverse=True
        )
        for emotion, prob in sorted_probs[:5]:  # Top 5
            bar = "█" * int(prob * 20)
            explanation.append(f"   {emotion:15s} {bar} {prob:.1%}")

        return "\n".join(explanation)

    def __repr__(self) -> str:
        return (
            f"ColoristaQuantitativo("
            f"color_space={'LAB' if self.use_lab else 'RGB'}, "
            f"target_size={self.target_size})"
        )