"""
Módulo de extratores de features.

Contém:
- VisualFeatureExtractor: Features visuais (cor, textura, composição)
- SemanticFeatureExtractor: Features semânticas extraídas da CNN
"""

from fuzzy_brain.extractors.visual import VisualFeatureExtractor

__all__ = ["VisualFeatureExtractor"]
