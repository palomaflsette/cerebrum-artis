"""
Fuzzy-Brain: Sistema Híbrido Neural-Fuzzy para Explicabilidade em Arte

Este pacote implementa um sistema que combina Deep Learning (CNN) com Lógica Fuzzy
para gerar explicações interpretáveis sobre emoções evocadas por obras de arte.

Módulos:
    - extractors: Extração de features visuais e semânticas
    - fuzzy: Sistema de lógica fuzzy (variáveis, regras, inferência)
    - integration: Integração Neural-Fuzzy e geração de explicações
    - utils: Utilitários e visualizações
"""

__version__ = "0.1.0"
__author__ = "Fuzzy-Brain Research Team"

# Imports disponíveis quando implementados
# from cerebrum_artis.fuzzy.fuzzy_brain.extractors.visual import VisualFeatureExtractor

__all__ = [
    "WarmthExtractor",
    "ColdnessExtractor",
    "SaturationExtractor",
    "MutednessExtractor",
    "BrightnessExtractor",
    "DarknessExtractor",
    "HarmonyExtractor",
]
