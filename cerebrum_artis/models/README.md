# Models Module

> Deep learning models for emotional classification of artwork.

[English](#english) | [Português](#português)

---

## English

### Overview

This module contains all model implementations for emotion classification, from baseline approaches to advanced ensemble methods. Each version represents an evolutionary step in the project's development.

### Model Versions

#### V1 - Baseline (v1_baseline/)

The initial baseline model using standard architectures without domain-specific features.

**Architecture:**
- **Vision**: ResNet50 pretrained on ImageNet
- **Text**: RoBERTa-base pretrained on common crawl
- **Fusion**: Simple concatenation + MLP classifier

**Performance:**
- Validation Accuracy: ~65%
- Training Time: ~4 hours (20 epochs)

**Key Files:**
- `model.py` - Model architecture
- `train.py` - Training script
- `config.json` - Model configuration

#### V2 - Improved (v2_improved/)

Enhanced version with better feature engineering and training strategies.

**Improvements:**
- Better image preprocessing
- Attention mechanisms
- Improved learning rate scheduling
- Data augmentation

**Performance:**
- Validation Accuracy: ~68%
- Training Time: ~5 hours (20 epochs)

#### V2 - Fuzzy Features (v2_fuzzy_features/)

**Major breakthrough:** Integration of fuzzy logic visual features.

**Architecture:**
```python
class MultimodalFuzzyClassifier(nn.Module):
    def __init__(self, num_classes=9):
        # Vision branch: ResNet50
        self.vision_encoder = ResNet50(pretrained=True)
        
        # Text branch: RoBERTa
        self.text_encoder = RobertaModel.from_pretrained('roberta-base')
        
        # Fuzzy features: 7 interpretable features
        self.fuzzy_dim = 7  # warm, cold, saturated, muted, bright, dark, harmonious
        
        # Fusion
        self.classifier = MLP(
            input_dim=vision_dim + text_dim + fuzzy_dim,
            hidden_dim=512,
            output_dim=num_classes
        )
```

**Fuzzy Features:**
1. **Warmth**: Degree of warm colors (reds, oranges, yellows)
2. **Coldness**: Degree of cold colors (blues, greens, purples)
3. **Saturation**: Color intensity and vividness
4. **Mutedness**: Degree of desaturated/muted colors
5. **Brightness**: Overall luminosity
6. **Darkness**: Degree of dark tones
7. **Harmony**: Color harmony and complementarity

**Performance:**
- **Validation Accuracy: 70.63%** (best individual model)
- Training Time: ~6 hours (20 epochs)
- Improvement: +5.63% over baseline

**Usage:**
```python
from cerebrum_artis.models.v2_fuzzy_features import MultimodalFuzzyClassifier

model = MultimodalFuzzyClassifier(num_classes=9)
model.load_checkpoint('path/to/checkpoint_best.pt')

# Inference
emotion = model.predict(image, caption)
```

#### V3 - Adaptive Gating (v3_adaptive_gating/)

Adaptive fusion mechanism using fuzzy features as gating signals.

**Key Innovation:**
```python
class FuzzyGatingClassifier(nn.Module):
    def forward(self, image, text, fuzzy_features):
        # Extract deep features
        vision_feat = self.vision_encoder(image)
        text_feat = self.text_encoder(text)
        
        # Compute gating weights from fuzzy features
        gate_weights = self.gating_network(fuzzy_features)
        
        # Adaptive fusion
        fused = gate_weights[0] * vision_feat + \
                gate_weights[1] * text_feat + \
                gate_weights[2] * fuzzy_features
        
        return self.classifier(fused)
```

**Performance:**
- Validation Accuracy: 70.37%
- Best Epoch: 5
- Note: Showed overfitting after epoch 5

#### V4.1 - Integrated Gating (v3_1_integrated/)

Integrated gating mechanism directly in the forward pass.

**Performance:**
- Validation Accuracy: 70.40%
- Best Epoch: 6
- Note: Suffered from severe overfitting (69.19% at epoch 10)

#### Ensemble (ensemble/)

**Best performing model:** Optimized weighted combination of V3, V4, and V4.1.

**Strategy:**
```python
class EnsembleClassifier:
    def __init__(self):
        self.models = [
            load_model('v2_fuzzy_features'),
            load_model('v3_adaptive_gating'),
            load_model('v3_1_integrated')
        ]
        
        # Optimized weights from grid search
        self.weights = [0.55, 0.30, 0.15]  # V3, V4, V4.1
    
    def predict(self, image, text):
        # Get predictions from all models
        probs = [model.predict_proba(image, text) for model in self.models]
        
        # Weighted average
        ensemble_probs = sum(w * p for w, p in zip(self.weights, probs))
        
        return ensemble_probs.argmax()
```

**Performance:**
- **Validation Accuracy: 71.47%** (state-of-the-art)
- Improvement: +0.84% over best individual model
- Grid Search: 441 weight combinations tested

**Ensemble Strategies Tested:**
1. Simple Average: 71.26%
2. Hard Voting: 71.13%
3. Performance-Weighted: 71.27%
4. **Optimized (Grid Search): 71.47%** (best)
5. V3+V4 Only: 71.32%

### Model Comparison

| Version | Val Acc | Parameters | Training Time | Key Feature |
|---------|---------|------------|---------------|-------------|
| V1 | ~65% | ~70M | 4h | Baseline |
| V2 | ~68% | ~72M | 5h | Better features |
| V3 | **70.63%** | ~75M | 6h | Fuzzy features |
| V4 | 70.37% | ~76M | 6h | Fuzzy gating |
| V4.1 | 70.40% | ~76M | 6h | Integrated gating |
| **Ensemble** | **71.47%** | ~227M | - | Weighted combination |

### Training a Model

```bash
# Train V3 (recommended)
cd /path/to/cerebrum-artis
python scripts/training/train_v3.py \
    --config configs/training/v3_config.json \
    --data_path /path/to/artemis/data \
    --checkpoint_dir /path/to/checkpoints \
    --num_epochs 20 \
    --batch_size 32

# Train with custom config
python scripts/training/train_v3.py \
    --lr 1e-4 \
    --weight_decay 1e-5 \
    --scheduler cosine \
    --warmup_epochs 3
```

### Loading Pretrained Models

```python
from cerebrum_artis.models.v2_fuzzy_features import MultimodalFuzzyClassifier
from cerebrum_artis.models.ensemble import EnsembleClassifier

# Load single model
model = MultimodalFuzzyClassifier.from_pretrained(
    'checkpoints/v2_fuzzy_features/checkpoint_best.pt'
)

# Load ensemble (recommended)
ensemble = EnsembleClassifier.from_pretrained(
    checkpoint_dir='checkpoints/'
)

# Inference
result = ensemble.predict(image, caption)
print(f"Emotion: {result['emotion']}")
print(f"Confidence: {result['confidence']:.2%}")
```

### Model Checkpoints

Pretrained checkpoints are available at:
```
/data/paloma/deep-mind-checkpoints/
├── v2_fuzzy_features/
│   └── checkpoint_best.pt (epoch 3, 70.63%)
├── v3_adaptive_gating/
│   └── checkpoint_best.pt (epoch 5, 70.37%)
└── v3_1_integrated/
    └── checkpoint_best.pt (epoch 6, 70.40%)
```

### Custom Model Development

To develop a new model version:

```python
from cerebrum_artis.models.base import BaseEmotionClassifier

class MyNewModel(BaseEmotionClassifier):
    def __init__(self, config):
        super().__init__(config)
        # Initialize your architecture
        self.custom_module = CustomModule()
    
    def forward(self, image, text):
        # Implement forward pass
        features = self.extract_features(image, text)
        logits = self.classifier(features)
        return logits
    
    def extract_features(self, image, text):
        # Your feature extraction logic
        pass
```

### Evaluation

```python
from cerebrum_artis.models import EnsembleClassifier
from cerebrum_artis.data import get_test_dataloader

# Load model
model = EnsembleClassifier.from_pretrained('checkpoints/')

# Load test data
test_loader = get_test_dataloader(batch_size=32)

# Evaluate
results = model.evaluate(test_loader)

print(f"Test Accuracy: {results['accuracy']:.2%}")
print(f"F1 Score: {results['f1_score']:.3f}")
print(f"Confusion Matrix:\n{results['confusion_matrix']}")
```

---

## Português

<details>
<summary>Clique para ver a versão em português</summary>

### Visão Geral

Este módulo contém todas as implementações de modelos para classificação emocional, desde abordagens baseline até métodos ensemble avançados.

### Versões dos Modelos

#### V1 - Baseline
Modelo baseline inicial usando arquiteturas padrão.

**Performance:**
- Acurácia de Validação: ~65%

#### V2 - Melhorado
Versão aprimorada com melhor engenharia de features.

**Performance:**
- Acurácia de Validação: ~68%

#### V3 - Fuzzy Features
**Avanço importante:** Integração de features visuais de lógica fuzzy.

**Performance:**
- **Acurácia de Validação: 70.63%** (melhor modelo individual)
- Melhoria: +5.63% sobre baseline

#### V4 - Fuzzy Gating
Mecanismo de fusão adaptativa usando features fuzzy como sinais de gating.

**Performance:**
- Acurácia de Validação: 70.37%

#### V3.1 - Gating Integrado
Mecanismo de gating integrado diretamente no forward pass.

**Performance:**
- Acurácia de Validação: 70.40%

#### Ensemble
**Melhor modelo:** Combinação ponderada otimizada de V3, V4 e V4.1.

**Performance:**
- **Acurácia de Validação: 71.47%** (state-of-the-art)
- Melhoria: +0.84% sobre melhor modelo individual

### Comparação de Modelos

| Versão | Acur. Val | Parâmetros | Tempo Treino | Feature Principal |
|---------|-----------|------------|--------------|-------------------|
| V1 | ~65% | ~70M | 4h | Baseline |
| V2 | ~68% | ~72M | 5h | Melhores features |
| V3 | **70.63%** | ~75M | 6h | Features fuzzy |
| V4 | 70.37% | ~76M | 6h | Fuzzy gating |
| V4.1 | 70.40% | ~76M | 6h | Gating integrado |
| **Ensemble** | **71.47%** | ~227M | - | Combinação ponderada |

### Treinando um Modelo

```bash
# Treinar V3 (recomendado)
python scripts/training/train_v3.py \
    --config configs/training/v3_config.json \
    --data_path /caminho/para/dados/artemis \
    --checkpoint_dir /caminho/para/checkpoints \
    --num_epochs 20 \
    --batch_size 32
```

### Carregando Modelos Pré-Treinados

```python
from cerebrum_artis.models.ensemble import EnsembleClassifier

# Carregar ensemble (recomendado)
ensemble = EnsembleClassifier.from_pretrained(
    checkpoint_dir='checkpoints/'
)

# Inferência
resultado = ensemble.predict(imagem, legenda)
print(f"Emoção: {resultado['emotion']}")
print(f"Confiança: {resultado['confidence']:.2%}")
```

</details>

---

**Module**: `cerebrum_artis.models`  
**Version**: 1.0.0  
**Last Updated**: November 25, 2025
