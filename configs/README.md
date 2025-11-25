# Configuration Files

> Training and inference configuration files for all models.

[English](#english) | [Português](#português)

---

## English

### Overview

Configuration files define hyperparameters, model settings, and training options. All configs use JSON format for easy editing.

### Directory Structure

```
configs/
├── training/           # Training configurations
│   ├── v1_baseline.json
│   ├── v2_improved.json
│   ├── v3_fuzzy_features.json
│   ├── v4_fuzzy_gating.json
│   └── v4_1_integrated.json
└── inference/          # Inference configurations
    ├── ensemble.json
    └── single_model.json
```

### Training Configurations

#### v3_fuzzy_features.json (Recommended)

```json
{
  "model": {
    "name": "v3_fuzzy_features",
    "architecture": "resnet50",
    "pretrained": true,
    "num_classes": 9,
    "fuzzy_features": {
      "enabled": true,
      "features": [
        "warmth", "coldness", "saturation",
        "mutedness", "brightness", "darkness", "harmony"
      ],
      "fusion_method": "concatenation"
    }
  },
  "training": {
    "num_epochs": 20,
    "batch_size": 32,
    "learning_rate": 1e-4,
    "optimizer": "adam",
    "scheduler": {
      "type": "reduce_on_plateau",
      "patience": 3,
      "factor": 0.5
    },
    "early_stopping": {
      "enabled": true,
      "patience": 5,
      "min_delta": 0.001
    }
  },
  "data": {
    "train_split": 0.8,
    "val_split": 0.1,
    "test_split": 0.1,
    "image_size": 224,
    "augmentation": {
      "horizontal_flip": true,
      "rotation": 15,
      "color_jitter": 0.2
    }
  },
  "checkpoint": {
    "save_best": true,
    "save_every_n_epochs": 5,
    "metric": "val_accuracy"
  }
}
```

#### v4_fuzzy_gating.json

```json
{
  "model": {
    "name": "v4_fuzzy_gating",
    "architecture": "resnet50",
    "pretrained": true,
    "num_classes": 9,
    "fuzzy_gating": {
      "enabled": true,
      "gate_type": "soft",
      "temperature": 1.0
    }
  },
  "training": {
    "num_epochs": 20,
    "batch_size": 32,
    "learning_rate": 1e-4,
    "optimizer": "adam",
    "warmup_epochs": 2
  }
}
```

### Inference Configurations

#### ensemble.json

```json
{
  "models": [
    {
      "name": "v3",
      "checkpoint": "checkpoints/v3_fuzzy_features/checkpoint_best.pt",
      "weight": 0.55
    },
    {
      "name": "v4",
      "checkpoint": "checkpoints/v4_fuzzy_gating/checkpoint_best.pt",
      "weight": 0.30
    },
    {
      "name": "v4_1",
      "checkpoint": "checkpoints/v4_1_integrated/checkpoint_best.pt",
      "weight": 0.15
    }
  ],
  "ensemble_strategy": "optimized_weighted",
  "post_processing": {
    "threshold": 0.5,
    "top_k": 3
  }
}
```

#### single_model.json

```json
{
  "model": {
    "name": "v3",
    "checkpoint": "checkpoints/v3_fuzzy_features/checkpoint_best.pt"
  },
  "inference": {
    "batch_size": 1,
    "device": "cuda",
    "precision": "fp32"
  }
}
```

### Configuration Parameters

#### Model Settings

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `name` | string | Model version identifier | required |
| `architecture` | string | Base architecture (resnet50, vit) | "resnet50" |
| `pretrained` | bool | Use ImageNet pretrained weights | true |
| `num_classes` | int | Number of emotion classes | 9 |

#### Training Settings

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `num_epochs` | int | Training epochs | 20 |
| `batch_size` | int | Batch size | 32 |
| `learning_rate` | float | Initial learning rate | 1e-4 |
| `optimizer` | string | Optimizer (adam, sgd, adamw) | "adam" |
| `weight_decay` | float | L2 regularization | 1e-5 |

#### Data Settings

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `train_split` | float | Training data ratio | 0.8 |
| `val_split` | float | Validation data ratio | 0.1 |
| `test_split` | float | Test data ratio | 0.1 |
| `image_size` | int | Input image size | 224 |
| `num_workers` | int | DataLoader workers | 4 |

### Using Configurations

**Load config in Python:**
```python
import json

with open('configs/training/v3_fuzzy_features.json', 'r') as f:
    config = json.load(f)

model_name = config['model']['name']
learning_rate = config['training']['learning_rate']
```

**Override config from command line:**
```bash
python scripts/training/train_v3.py \
    --config configs/training/v3_fuzzy_features.json \
    --learning_rate 5e-5 \
    --batch_size 64
```

### Creating Custom Configs

**1. Copy existing config:**
```bash
cp configs/training/v3_fuzzy_features.json configs/training/my_experiment.json
```

**2. Edit parameters:**
```json
{
  "model": {
    "name": "my_experiment",
    ...
  },
  "training": {
    "learning_rate": 5e-5,
    "batch_size": 64,
    ...
  }
}
```

**3. Run training:**
```bash
python scripts/training/train_v3.py \
    --config configs/training/my_experiment.json
```

### Best Practices

- Keep separate configs for experiments
- Use descriptive names (e.g., `v3_lr5e5_bs64.json`)
- Document changes in config comments (if using JSON5)
- Version control configs with git
- Never commit sensitive paths (use environment variables)

---

## Português

<details>
<summary>Clique para ver versão em português</summary>

### Visão Geral

Arquivos de configuração definem hiperparâmetros, configurações de modelo e opções de treinamento. Todas as configs usam formato JSON para fácil edição.

### Estrutura de Diretórios

```
configs/
├── training/           # Configurações de treinamento
│   ├── v1_baseline.json
│   ├── v2_improved.json
│   ├── v3_fuzzy_features.json
│   ├── v4_fuzzy_gating.json
│   └── v4_1_integrated.json
└── inference/          # Configurações de inferência
    ├── ensemble.json
    └── single_model.json
```

### Parâmetros de Configuração

#### Configurações de Modelo

| Parâmetro | Tipo | Descrição | Padrão |
|-----------|------|-----------|---------|
| `name` | string | Identificador da versão do modelo | obrigatório |
| `architecture` | string | Arquitetura base (resnet50, vit) | "resnet50" |
| `pretrained` | bool | Usar pesos pré-treinados do ImageNet | true |
| `num_classes` | int | Número de classes de emoção | 9 |

#### Configurações de Treinamento

| Parâmetro | Tipo | Descrição | Padrão |
|-----------|------|-----------|---------|
| `num_epochs` | int | Épocas de treinamento | 20 |
| `batch_size` | int | Tamanho do batch | 32 |
| `learning_rate` | float | Taxa de aprendizado inicial | 1e-4 |
| `optimizer` | string | Otimizador (adam, sgd, adamw) | "adam" |
| `weight_decay` | float | Regularização L2 | 1e-5 |

#### Configurações de Dados

| Parâmetro | Tipo | Descrição | Padrão |
|-----------|------|-----------|---------|
| `train_split` | float | Proporção de dados de treino | 0.8 |
| `val_split` | float | Proporção de dados de validação | 0.1 |
| `test_split` | float | Proporção de dados de teste | 0.1 |
| `image_size` | int | Tamanho da imagem de entrada | 224 |
| `num_workers` | int | Workers do DataLoader | 4 |

### Usando Configurações

**Carregar config em Python:**
```python
import json

with open('configs/training/v3_fuzzy_features.json', 'r') as f:
    config = json.load(f)

model_name = config['model']['name']
learning_rate = config['training']['learning_rate']
```

**Sobrescrever config da linha de comando:**
```bash
python scripts/training/train_v3.py \
    --config configs/training/v3_fuzzy_features.json \
    --learning_rate 5e-5 \
    --batch_size 64
```

### Criando Configs Customizadas

**1. Copiar config existente:**
```bash
cp configs/training/v3_fuzzy_features.json configs/training/meu_experimento.json
```

**2. Editar parâmetros:**
```json
{
  "model": {
    "name": "meu_experimento",
    ...
  },
  "training": {
    "learning_rate": 5e-5,
    "batch_size": 64,
    ...
  }
}
```

**3. Executar treinamento:**
```bash
python scripts/training/train_v3.py \
    --config configs/training/meu_experimento.json
```

### Melhores Práticas

- Manter configs separadas para experimentos
- Usar nomes descritivos (ex: `v3_lr5e5_bs64.json`)
- Documentar mudanças em comentários de config (se usar JSON5)
- Versionar configs com git
- Nunca commitar paths sensíveis (usar variáveis de ambiente)

</details>

---

**Directory**: `/home/paloma/cerebrum-artis/configs/`  
**Last Updated**: November 25, 2025
