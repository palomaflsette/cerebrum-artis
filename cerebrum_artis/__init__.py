"""
Cerebrum Artis: Sistema Multi-Agente para Análise Emocional de Arte

Sistema que combina Deep Learning com Lógica Fuzzy para análise emocional de obras de arte.

Módulos:
    - agents: Agentes especializados (PerceptoEmocional, Colorista, Explicador)
    - models: Arquiteturas de modelos (V1-V4.1)
    - fuzzy: Sistema de lógica fuzzy (7 features visuais)
    - data: Carregamento e processamento de dados
    - utils: Utilitários e configurações
"""

__version__ = "0.1.0"
__author__ = "Paloma Sette"

# Módulos disponíveis (importação sob demanda)
__all__ = [
    "agents",
    "models",
    "fuzzy",
    "data",
    "utils",
]
