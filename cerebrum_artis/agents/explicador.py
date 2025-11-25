"""
Agente 3: Explicador Visual

Responsável pela explicabilidade (XAI) usando:
- Grad-CAM (Gradient-weighted Class Activation Mapping)
- Visualização de regiões importantes
- Geração de mapas de calor

Este agente explica PORQUÊ o modelo detectou determinada emoção.

TODO: Implementar após Agente 2
"""

from typing import Dict, Union, Optional
from pathlib import Path
import numpy as np


class ExplicadorVisual:
    """
    Agente 3: Explicador Visual (XAI com Grad-CAM)

    TODO: Implementar Grad-CAM após Agente 2
    """

    def __init__(self, model_path: Optional[str] = None):
        """
        Inicializa o Explicador Visual.

        Args:
            model_path: Caminho para modelo do Agente 2
        """
        self.model_path = model_path
        self.model = None
        raise NotImplementedError(
            "Agente 3 ainda não implementado. "
            "Requer Agente 2 (Percepto Emocional) implementado primeiro."
        )

    def explain(
        self,
        image: Union[str, Path, np.ndarray],
        emotion: str,
        save_path: Optional[str] = None
    ) -> np.ndarray:
        """
        Gera mapa de calor Grad-CAM explicando a predição.

        Args:
            image: Caminho da imagem ou array numpy
            emotion: Emoção para explicar
            save_path: Se fornecido, salva mapa de calor

        Returns:
            Array numpy com mapa de calor
        """
        raise NotImplementedError()

    def visualize(
        self,
        image: Union[str, Path, np.ndarray],
        heatmap: np.ndarray,
        alpha: float = 0.4
    ) -> np.ndarray:
        """
        Sobrepõe mapa de calor na imagem original.

        Args:
            image: Imagem original
            heatmap: Mapa de calor Grad-CAM
            alpha: Transparência do heatmap [0, 1]

        Returns:
            Imagem com heatmap sobreposto
        """
        raise NotImplementedError()