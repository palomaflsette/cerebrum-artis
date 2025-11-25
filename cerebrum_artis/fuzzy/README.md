# Fuzzy Logic Module

> Fuzzy logic system for interpretable visual feature extraction.

[English](#english) | [Português](#português)

---

## English

### Overview

This module implements a fuzzy logic system for extracting interpretable visual features from artwork. These features provide human-understandable representations of visual properties related to emotional content.

### Fuzzy Features

The system extracts 7 fuzzy features:

1. **Warmth** - Degree of warm colors (reds, oranges, yellows)
2. **Coldness** - Degree of cold colors (blues, greens, purples)
3. **Saturation** - Color intensity and vividness
4. **Mutedness** - Degree of desaturated colors
5. **Brightness** - Overall luminosity
6. **Darkness** - Degree of dark tones
7. **Harmony** - Color harmony and complementarity

### Usage

```python
from cerebrum_artis.fuzzy import FuzzyFeatureExtractor

# Initialize extractor
extractor = FuzzyFeatureExtractor(color_space='LAB')

# Extract features from image
features = extractor.extract(image_path)

# Access individual features
warmth = features['warmth']          # 0.0 to 1.0
saturation = features['saturation']  # 0.0 to 1.0
harmony = features['harmony']        # 0.0 to 1.0

# Get all features as vector
feature_vector = extractor.to_vector(features)  # shape: (7,)
```

### Fuzzy Sets

Each feature is computed using fuzzy membership functions:

```python
# Example: Warmth computation
def compute_warmth(colors):
    """
    Computes warmth using fuzzy membership in warm color sets.
    
    Warm colors: Hue in [0°-60°] (reds, oranges, yellows)
    Membership function: Triangular/Trapezoidal
    """
    warm_membership = []
    for color in colors:
        hue = color['hue']
        # Fuzzy membership computation
        if 0 <= hue <= 60:
            membership = 1.0 - abs(hue - 30) / 30
        else:
            membership = 0.0
        warm_membership.append(membership)
    
    # Aggregate: weighted average by pixel count
    return weighted_mean(warm_membership)
```

### Color Space

The system supports multiple color spaces:

- **RGB**: Red-Green-Blue (default for most operations)
- **LAB**: Perceptually uniform color space (recommended)
- **HSV**: Hue-Saturation-Value (useful for color analysis)

```python
# Use LAB color space (recommended)
extractor = FuzzyFeatureExtractor(color_space='LAB')
```

### Extractors

The module provides specialized extractors:

#### DominantColorExtractor

```python
from cerebrum_artis.fuzzy.extractors import DominantColorExtractor

extractor = DominantColorExtractor(n_colors=5)
colors = extractor.extract(image)
# Returns: list of (color, percentage) tuples
```

#### HarmonyAnalyzer

```python
from cerebrum_artis.fuzzy.extractors import HarmonyAnalyzer

analyzer = HarmonyAnalyzer()
harmony_score = analyzer.analyze(color_palette)
# Returns: harmony score [0.0, 1.0]
```

### Configuration

```python
config = {
    'color_space': 'LAB',
    'n_clusters': 5,        # Number of dominant colors
    'min_pixels': 100,      # Minimum pixels for color consideration
    'fuzzy_sets': {
        'warm': {'range': [0, 60], 'type': 'triangular'},
        'cold': {'range': [180, 300], 'type': 'triangular'},
        'saturated': {'threshold': 0.5, 'type': 'sigmoid'}
    }
}

extractor = FuzzyFeatureExtractor(config)
```

---

## Português

<details>
<summary>Clique para ver a versão em português</summary>

### Visão Geral

Este módulo implementa um sistema de lógica fuzzy para extração de features visuais interpretáveis de obras de arte.

### Features Fuzzy

O sistema extrai 7 features fuzzy:

1. **Calor (Warmth)** - Grau de cores quentes
2. **Frio (Coldness)** - Grau de cores frias
3. **Saturação (Saturation)** - Intensidade das cores
4. **Dessaturação (Mutedness)** - Grau de cores dessaturadas
5. **Brilho (Brightness)** - Luminosidade geral
6. **Escuridão (Darkness)** - Grau de tons escuros
7. **Harmonia (Harmony)** - Harmonia de cores

### Uso

```python
from cerebrum_artis.fuzzy import FuzzyFeatureExtractor

# Inicializar extrator
extractor = FuzzyFeatureExtractor(color_space='LAB')

# Extrair features da imagem
features = extractor.extract(image_path)

# Acessar features individuais
warmth = features['warmth']          # 0.0 a 1.0
saturation = features['saturation']  # 0.0 a 1.0
```

</details>

---

**Module**: `cerebrum_artis.fuzzy`  
**Version**: 1.0.0  
**Last Updated**: November 25, 2025
