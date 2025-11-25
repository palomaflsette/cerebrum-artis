# Data Processing Utilities

> Data loading, preprocessing, and dataset management utilities.

[English](#english) | [Português](#português)

---

## English

### Overview

This module provides utilities for loading, preprocessing, and managing the ArtEmis dataset for training and inference.

### Components

```
cerebrum_artis/data/
├── __init__.py
├── dataset.py              # PyTorch Dataset classes
├── preprocess.py           # Data preprocessing
├── transforms.py           # Image transformations
├── cache.py                # Feature caching utilities
└── loaders.py              # DataLoader factories
```

### ArtEmisDataset

Main dataset class for loading ArtEmis data.

```python
from cerebrum_artis.data import ArtEmisDataset
from torch.utils.data import DataLoader

# Create dataset
dataset = ArtEmisDataset(
    data_file='data/artemis/processed/train.pkl',
    image_size=224,
    augment=True
)

# Create dataloader
loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4
)

# Iterate over batches
for batch in loader:
    images = batch['image']      # (B, 3, 224, 224)
    emotions = batch['emotion']  # (B,) - emotion labels (0-8)
    captions = batch['caption']  # List[str] - text descriptions
    painting_ids = batch['painting_id']  # List[str]
```

**Parameters:**
- `data_file` (str): Path to preprocessed pickle file
- `image_size` (int): Target image size (default: 224)
- `augment` (bool): Apply data augmentation (default: False)
- `transform` (callable): Custom transform function
- `cache_features` (bool): Cache extracted features (default: False)

### Data Preprocessing

Preprocess raw ArtEmis CSV into train/val/test splits.

```python
from cerebrum_artis.data import preprocess_artemis

# Preprocess dataset
preprocess_artemis(
    csv_file='data/artemis/raw/artemis_dataset_release_v0.csv',
    image_dir='data/artemis/raw/images',
    output_dir='data/artemis/processed',
    train_ratio=0.8,
    val_ratio=0.1,
    test_ratio=0.1,
    seed=42
)
```

**Output:**
```
data/artemis/processed/
├── train.pkl           # Training data
├── val.pkl             # Validation data
├── test.pkl            # Test data
├── metadata.json       # Dataset statistics
└── emotion_dist.json   # Emotion distribution
```

### Data Augmentation

Image transformations for training.

```python
from cerebrum_artis.data.transforms import (
    get_train_transforms,
    get_val_transforms
)

# Training transforms (with augmentation)
train_transform = get_train_transforms(
    image_size=224,
    horizontal_flip=True,
    rotation=15,
    color_jitter=0.2
)

# Validation transforms (no augmentation)
val_transform = get_val_transforms(image_size=224)
```

**Training augmentations:**
- Random horizontal flip (p=0.5)
- Random rotation (±15°)
- Color jitter (brightness, contrast, saturation)
- Random crop and resize
- Normalization (ImageNet stats)

**Validation/test:**
- Center crop
- Resize to target size
- Normalization

### Feature Caching

Cache extracted features for faster training.

```python
from cerebrum_artis.data.cache import FeatureCache

# Create cache
cache = FeatureCache(cache_dir='data/cache/features')

# Cache features during first epoch
for batch in train_loader:
    images = batch['image']
    painting_ids = batch['painting_id']
    
    # Extract features
    features = model.extract_features(images)
    
    # Cache for next epoch
    cache.save_batch(painting_ids, features)

# Load cached features
cached_features = cache.load_batch(painting_ids)
```

**Benefits:**
- 10x faster training iterations
- Reduced GPU memory usage
- Consistent features across experiments

### DataLoader Factories

Convenient dataloader creation.

```python
from cerebrum_artis.data.loaders import create_dataloaders

# Create all dataloaders at once
train_loader, val_loader, test_loader = create_dataloaders(
    data_dir='data/artemis/processed',
    batch_size=32,
    num_workers=4,
    augment_train=True,
    shuffle_train=True
)
```

### Emotion Mapping

```python
from cerebrum_artis.data import EMOTION_LABELS, EMOTION_TO_ID

# Emotion names
print(EMOTION_LABELS)
# ['amusement', 'awe', 'contentment', 'excitement', 
#  'anger', 'disgust', 'fear', 'sadness', 'something else']

# Convert emotion name to ID
emotion_id = EMOTION_TO_ID['awe']  # 1

# Convert ID to name
emotion_name = EMOTION_LABELS[emotion_id]  # 'awe'
```

### Dataset Statistics

```python
from cerebrum_artis.data import get_dataset_stats

# Get statistics
stats = get_dataset_stats('data/artemis/processed/train.pkl')

print(f"Total samples: {stats['total_samples']}")
print(f"Unique paintings: {stats['unique_paintings']}")
print(f"Emotion distribution:")
for emotion, count in stats['emotion_dist'].items():
    print(f"  {emotion}: {count} ({count/stats['total_samples']*100:.1f}%)")
```

**Example output:**
```
Total samples: 64000
Unique paintings: 35000
Emotion distribution:
  amusement: 3200 (5.0%)
  awe: 12800 (20.0%)
  contentment: 9600 (15.0%)
  excitement: 6400 (10.0%)
  anger: 3200 (5.0%)
  disgust: 3200 (5.0%)
  fear: 6400 (10.0%)
  sadness: 12800 (20.0%)
  something else: 6400 (10.0%)
```

### Custom Dataset

Create custom dataset for new data.

```python
from torch.utils.data import Dataset
from cerebrum_artis.data.transforms import get_train_transforms

class CustomArtDataset(Dataset):
    def __init__(self, image_paths, emotions, captions):
        self.image_paths = image_paths
        self.emotions = emotions
        self.captions = captions
        self.transform = get_train_transforms()
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Load image
        image = Image.open(self.image_paths[idx]).convert('RGB')
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        return {
            'image': image,
            'emotion': self.emotions[idx],
            'caption': self.captions[idx]
        }
```

---

## Português

<details>
<summary>Clique para ver versão em português</summary>

### Visão Geral

Este módulo fornece utilitários para carregar, pré-processar e gerenciar o dataset ArtEmis para treinamento e inferência.

### ArtEmisDataset

Classe principal de dataset para carregar dados ArtEmis.

```python
from cerebrum_artis.data import ArtEmisDataset
from torch.utils.data import DataLoader

# Criar dataset
dataset = ArtEmisDataset(
    data_file='data/artemis/processed/train.pkl',
    image_size=224,
    augment=True
)

# Criar dataloader
loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4
)

# Iterar sobre batches
for batch in loader:
    images = batch['image']      # (B, 3, 224, 224)
    emotions = batch['emotion']  # (B,) - labels de emoção (0-8)
    captions = batch['caption']  # List[str] - descrições textuais
    painting_ids = batch['painting_id']  # List[str]
```

### Pré-processamento de Dados

Pré-processar CSV bruto do ArtEmis em splits train/val/test.

```python
from cerebrum_artis.data import preprocess_artemis

# Pré-processar dataset
preprocess_artemis(
    csv_file='data/artemis/raw/artemis_dataset_release_v0.csv',
    image_dir='data/artemis/raw/images',
    output_dir='data/artemis/processed',
    train_ratio=0.8,
    val_ratio=0.1,
    test_ratio=0.1,
    seed=42
)
```

### Aumento de Dados

Transformações de imagem para treinamento.

```python
from cerebrum_artis.data.transforms import (
    get_train_transforms,
    get_val_transforms
)

# Transformações de treino (com aumento)
train_transform = get_train_transforms(
    image_size=224,
    horizontal_flip=True,
    rotation=15,
    color_jitter=0.2
)

# Transformações de validação (sem aumento)
val_transform = get_val_transforms(image_size=224)
```

**Aumentos de treinamento:**
- Flip horizontal aleatório (p=0.5)
- Rotação aleatória (±15°)
- Color jitter (brilho, contraste, saturação)
- Crop e resize aleatórios
- Normalização (stats do ImageNet)

### Mapeamento de Emoções

```python
from cerebrum_artis.data import EMOTION_LABELS, EMOTION_TO_ID

# Nomes de emoções
print(EMOTION_LABELS)
# ['amusement', 'awe', 'contentment', 'excitement', 
#  'anger', 'disgust', 'fear', 'sadness', 'something else']

# Converter nome de emoção para ID
emotion_id = EMOTION_TO_ID['awe']  # 1

# Converter ID para nome
emotion_name = EMOTION_LABELS[emotion_id]  # 'awe'
```

</details>

---

**Module**: `cerebrum_artis.data`  
**Version**: 1.0.0  
**Last Updated**: November 25, 2025
