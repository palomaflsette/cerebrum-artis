# Utility Functions

> Helper functions, common utilities, and shared components.

[English](#english) | [Português](#português)

---

## English

### Overview

This module contains utility functions used across the project for visualization, metrics, logging, and other common operations.

### Components

```
cerebrum_artis/utils/
├── __init__.py
├── metrics.py          # Evaluation metrics
├── visualization.py    # Plotting and visualization
├── logging.py          # Logging utilities
├── checkpoints.py      # Checkpoint management
└── config.py           # Configuration helpers
```

### Evaluation Metrics

Calculate performance metrics for emotion classification.

```python
from cerebrum_artis.utils.metrics import (
    compute_accuracy,
    compute_confusion_matrix,
    compute_per_class_metrics,
    compute_ensemble_metrics
)

# Basic accuracy
accuracy = compute_accuracy(predictions, ground_truth)
print(f"Accuracy: {accuracy:.2%}")

# Confusion matrix
cm = compute_confusion_matrix(predictions, ground_truth, num_classes=9)

# Per-class metrics
metrics = compute_per_class_metrics(predictions, ground_truth)
for emotion, scores in metrics.items():
    print(f"{emotion}:")
    print(f"  Precision: {scores['precision']:.3f}")
    print(f"  Recall: {scores['recall']:.3f}")
    print(f"  F1-Score: {scores['f1']:.3f}")
```

**Available metrics:**
- Accuracy (overall and per-class)
- Precision, Recall, F1-Score
- Confusion matrix
- Top-k accuracy
- Class-balanced accuracy

### Visualization

Plot training curves, confusion matrices, and results.

```python
from cerebrum_artis.utils.visualization import (
    plot_training_curve,
    plot_confusion_matrix,
    plot_emotion_distribution,
    plot_ensemble_comparison
)

# Plot training history
plot_training_curve(
    train_losses=[1.2, 1.0, 0.8, 0.7],
    val_losses=[1.3, 1.1, 0.9, 0.8],
    val_accuracies=[0.45, 0.55, 0.65, 0.70],
    save_path='outputs/training_curve.png'
)

# Plot confusion matrix
plot_confusion_matrix(
    confusion_matrix=cm,
    class_names=EMOTION_LABELS,
    normalize=True,
    save_path='outputs/confusion_matrix.png'
)

# Compare ensemble strategies
plot_ensemble_comparison(
    strategies=['simple', 'voting', 'weighted', 'optimized'],
    accuracies=[0.7126, 0.7113, 0.7127, 0.7147],
    save_path='outputs/ensemble_comparison.png'
)
```

**Visualization functions:**
- Training/validation curves
- Confusion matrices (raw and normalized)
- Emotion distribution histograms
- Model comparison bar charts
- Attention heatmaps
- Feature importance plots

### Logging

Structured logging for experiments.

```python
from cerebrum_artis.utils.logging import setup_logger, log_metrics

# Setup logger
logger = setup_logger(
    name='experiment',
    log_dir='logs',
    log_file='experiment.log',
    level='INFO'
)

# Log messages
logger.info("Starting training")
logger.debug(f"Batch size: {batch_size}")
logger.warning("Learning rate might be too high")

# Log metrics
log_metrics(
    logger=logger,
    epoch=10,
    train_loss=0.8234,
    val_loss=0.8912,
    val_accuracy=0.7063,
    learning_rate=1e-4
)
```

**Output format:**
```
[2025-11-25 14:30:00] INFO - Starting training
[2025-11-25 14:30:01] DEBUG - Batch size: 32
[2025-11-25 14:30:02] WARNING - Learning rate might be too high
[2025-11-25 14:35:00] INFO - Epoch 10/20
[2025-11-25 14:35:00] INFO -   Train Loss: 0.8234
[2025-11-25 14:35:00] INFO -   Val Loss: 0.8912
[2025-11-25 14:35:00] INFO -   Val Accuracy: 70.63%
[2025-11-25 14:35:00] INFO -   Learning Rate: 0.0001
```

### Checkpoint Management

Save and load model checkpoints.

```python
from cerebrum_artis.utils.checkpoints import (
    save_checkpoint,
    load_checkpoint,
    get_best_checkpoint
)

# Save checkpoint
save_checkpoint(
    model=model,
    optimizer=optimizer,
    scheduler=scheduler,
    epoch=15,
    val_accuracy=0.7063,
    checkpoint_dir='checkpoints/v3_fuzzy_features',
    is_best=True
)

# Load checkpoint
checkpoint = load_checkpoint(
    checkpoint_path='checkpoints/v3_fuzzy_features/checkpoint_best.pt',
    model=model,
    optimizer=optimizer  # optional
)
start_epoch = checkpoint['epoch'] + 1

# Get best checkpoint from directory
best_checkpoint_path = get_best_checkpoint(
    checkpoint_dir='checkpoints/v3_fuzzy_features',
    metric='val_accuracy'
)
```

### Configuration Helpers

Load and manage configuration files.

```python
from cerebrum_artis.utils.config import (
    load_config,
    save_config,
    merge_configs,
    override_config
)

# Load JSON config
config = load_config('configs/training/v3_fuzzy_features.json')

# Override with command-line arguments
config = override_config(
    config,
    overrides={
        'training.learning_rate': 5e-5,
        'training.batch_size': 64
    }
)

# Save updated config
save_config(
    config,
    'configs/training/v3_modified.json'
)

# Merge multiple configs
base_config = load_config('configs/base.json')
exp_config = load_config('configs/experiment.json')
final_config = merge_configs(base_config, exp_config)
```

### Device Management

Handle CPU/GPU device selection.

```python
from cerebrum_artis.utils import get_device, move_to_device

# Get available device
device = get_device(prefer_cuda=True)
print(f"Using device: {device}")

# Move model and data to device
model = model.to(device)
images = move_to_device(images, device)

# Multi-GPU support
if torch.cuda.device_count() > 1:
    model = torch.nn.DataParallel(model)
    print(f"Using {torch.cuda.device_count()} GPUs")
```

### Reproducibility

Set random seeds for reproducibility.

```python
from cerebrum_artis.utils import set_seed

# Set all random seeds
set_seed(42)

# Now all operations are deterministic:
# - PyTorch random operations
# - NumPy random operations
# - Python random module
# - CUDA operations
```

### Timer

Measure execution time.

```python
from cerebrum_artis.utils import Timer

# Context manager
with Timer("Training epoch"):
    train_epoch(model, train_loader)
# Output: Training epoch took 5m 32s

# Manual timing
timer = Timer()
timer.start()
process_data()
elapsed = timer.stop()
print(f"Processing took {elapsed:.2f}s")
```

### Progress Bar

Display training progress.

```python
from cerebrum_artis.utils import create_progress_bar

# Create progress bar
pbar = create_progress_bar(
    total=len(train_loader),
    desc="Training",
    unit="batch"
)

for batch in train_loader:
    # Training step
    loss = train_step(batch)
    
    # Update progress bar
    pbar.update(1)
    pbar.set_postfix({'loss': f'{loss:.4f}'})

pbar.close()
```

**Output:**
```
Training: 100%|██████████| 2000/2000 [05:32<00:00, 6.02batch/s, loss=0.8234]
```

### Common Utilities

```python
from cerebrum_artis.utils import (
    ensure_dir,          # Create directory if not exists
    count_parameters,    # Count model parameters
    format_time,         # Format seconds to human-readable
    get_timestamp,       # Get current timestamp string
    save_json,           # Save dict as JSON
    load_json            # Load JSON as dict
)

# Ensure directory exists
ensure_dir('outputs/visualizations')

# Count model parameters
num_params = count_parameters(model)
print(f"Model has {num_params:,} parameters")

# Format time
time_str = format_time(3725)  # "1h 2m 5s"

# Get timestamp
timestamp = get_timestamp()  # "2025-11-25_14-30-00"

# Save/load JSON
save_json({'accuracy': 0.7147}, 'results.json')
results = load_json('results.json')
```

---

## Português

<details>
<summary>Clique para ver versão em português</summary>

### Visão Geral

Este módulo contém funções utilitárias usadas em todo o projeto para visualização, métricas, logging e outras operações comuns.

### Métricas de Avaliação

Calcular métricas de performance para classificação de emoções.

```python
from cerebrum_artis.utils.metrics import (
    compute_accuracy,
    compute_confusion_matrix,
    compute_per_class_metrics
)

# Acurácia básica
accuracy = compute_accuracy(predictions, ground_truth)
print(f"Acurácia: {accuracy:.2%}")

# Matriz de confusão
cm = compute_confusion_matrix(predictions, ground_truth, num_classes=9)

# Métricas por classe
metrics = compute_per_class_metrics(predictions, ground_truth)
for emotion, scores in metrics.items():
    print(f"{emotion}:")
    print(f"  Precisão: {scores['precision']:.3f}")
    print(f"  Recall: {scores['recall']:.3f}")
    print(f"  F1-Score: {scores['f1']:.3f}")
```

**Métricas disponíveis:**
- Acurácia (geral e por classe)
- Precisão, Recall, F1-Score
- Matriz de confusão
- Top-k accuracy
- Acurácia balanceada por classe

### Visualização

Plotar curvas de treinamento, matrizes de confusão e resultados.

```python
from cerebrum_artis.utils.visualization import (
    plot_training_curve,
    plot_confusion_matrix,
    plot_emotion_distribution
)

# Plotar histórico de treinamento
plot_training_curve(
    train_losses=[1.2, 1.0, 0.8, 0.7],
    val_losses=[1.3, 1.1, 0.9, 0.8],
    val_accuracies=[0.45, 0.55, 0.65, 0.70],
    save_path='outputs/training_curve.png'
)

# Plotar matriz de confusão
plot_confusion_matrix(
    confusion_matrix=cm,
    class_names=EMOTION_LABELS,
    normalize=True,
    save_path='outputs/confusion_matrix.png'
)
```

### Logging

Logging estruturado para experimentos.

```python
from cerebrum_artis.utils.logging import setup_logger, log_metrics

# Configurar logger
logger = setup_logger(
    name='experiment',
    log_dir='logs',
    log_file='experiment.log',
    level='INFO'
)

# Logar mensagens
logger.info("Iniciando treinamento")
logger.debug(f"Tamanho do batch: {batch_size}")
logger.warning("Taxa de aprendizado pode estar muito alta")

# Logar métricas
log_metrics(
    logger=logger,
    epoch=10,
    train_loss=0.8234,
    val_loss=0.8912,
    val_accuracy=0.7063,
    learning_rate=1e-4
)
```

### Gerenciamento de Checkpoints

Salvar e carregar checkpoints de modelos.

```python
from cerebrum_artis.utils.checkpoints import (
    save_checkpoint,
    load_checkpoint,
    get_best_checkpoint
)

# Salvar checkpoint
save_checkpoint(
    model=model,
    optimizer=optimizer,
    scheduler=scheduler,
    epoch=15,
    val_accuracy=0.7063,
    checkpoint_dir='checkpoints/v3_fuzzy_features',
    is_best=True
)

# Carregar checkpoint
checkpoint = load_checkpoint(
    checkpoint_path='checkpoints/v3_fuzzy_features/checkpoint_best.pt',
    model=model,
    optimizer=optimizer  # opcional
)
start_epoch = checkpoint['epoch'] + 1
```

### Reprodutibilidade

Definir seeds aleatórias para reprodutibilidade.

```python
from cerebrum_artis.utils import set_seed

# Definir todas as seeds aleatórias
set_seed(42)

# Agora todas as operações são determinísticas:
# - Operações aleatórias do PyTorch
# - Operações aleatórias do NumPy
# - Módulo random do Python
# - Operações CUDA
```

### Utilitários Comuns

```python
from cerebrum_artis.utils import (
    ensure_dir,          # Criar diretório se não existir
    count_parameters,    # Contar parâmetros do modelo
    format_time,         # Formatar segundos para legível
    get_timestamp,       # Obter timestamp string atual
    save_json,           # Salvar dict como JSON
    load_json            # Carregar JSON como dict
)

# Garantir que diretório existe
ensure_dir('outputs/visualizations')

# Contar parâmetros do modelo
num_params = count_parameters(model)
print(f"Modelo tem {num_params:,} parâmetros")

# Formatar tempo
time_str = format_time(3725)  # "1h 2m 5s"

# Obter timestamp
timestamp = get_timestamp()  # "2025-11-25_14-30-00"

# Salvar/carregar JSON
save_json({'accuracy': 0.7147}, 'results.json')
results = load_json('results.json')
```

</details>

---

**Module**: `cerebrum_artis.utils`  
**Version**: 1.0.0  
**Last Updated**: November 25, 2025
