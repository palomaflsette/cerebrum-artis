# Data Directory

> Dataset files, preprocessed data, and data processing utilities.

[English](#english) | [Português](#português)

---

## English

### Overview

This directory contains all data-related files for training and evaluation. The main dataset used is **ArtEmis** (Art Emotions in the Wild).

### Directory Structure

```
data/
├── artemis/                    # ArtEmis dataset
│   ├── raw/                    # Original dataset files
│   │   ├── artemis_dataset_release_v0.csv
│   │   └── images/
│   ├── processed/              # Preprocessed data
│   │   ├── train.pkl
│   │   ├── val.pkl
│   │   └── test.pkl
│   └── splits/                 # Train/val/test splits
│       ├── train_ids.txt
│       ├── val_ids.txt
│       └── test_ids.txt
├── cache/                      # Cached features
│   ├── image_features/
│   ├── text_features/
│   └── fuzzy_features/
└── examples/                   # Sample data for testing
    ├── sample_images/
    └── sample_annotations.json
```

### ArtEmis Dataset

**About:**
- 80,000+ artwork-emotion-description triplets
- 9 emotion categories
- Rich textual descriptions of emotional responses
- WikiArt images (40,000+ artworks)

**Emotion Categories:**
1. Amusement
2. Awe
3. Contentment
4. Excitement
5. Anger
6. Disgust
7. Fear
8. Sadness
9. Something else

**Dataset Statistics:**
- Total samples: ~80,000
- Train split: 64,000 (80%)
- Validation split: 8,000 (10%)
- Test split: 8,000 (10%)
- Average description length: 15 words
- Images: JPEG format, various sizes

### Downloading the Dataset

**1. Download ArtEmis:**
```bash
cd data/artemis
wget https://www.artemisdataset.org/download/artemis_dataset_release_v0.csv
```

**2. Download WikiArt images:**
```bash
# Clone the WikiArt downloader
git clone https://github.com/oprahhh/artemis-wikiart-downloader
cd artemis-wikiart-downloader

# Download images (requires ~20GB)
python download_images.py --output_dir ../data/artemis/raw/images
```

**3. Verify download:**
```bash
# Check CSV file
wc -l data/artemis/raw/artemis_dataset_release_v0.csv
# Expected: ~80,000 lines

# Check images
ls data/artemis/raw/images | wc -l
# Expected: ~40,000 images
```

### Data Preprocessing

**Preprocess dataset:**
```bash
python cerebrum_artis/data/preprocess.py \
    --input_csv data/artemis/raw/artemis_dataset_release_v0.csv \
    --image_dir data/artemis/raw/images \
    --output_dir data/artemis/processed \
    --splits 0.8 0.1 0.1
```

**What preprocessing does:**
- Loads and validates CSV annotations
- Filters invalid/missing images
- Creates train/val/test splits
- Saves preprocessed pickle files
- Generates split ID files

**Output files:**
```
data/artemis/processed/
├── train.pkl       # Training data (64K samples)
├── val.pkl         # Validation data (8K samples)
├── test.pkl        # Test data (8K samples)
└── metadata.json   # Dataset statistics
```

### Using the Data

**Load preprocessed data:**
```python
import pickle

# Load training data
with open('data/artemis/processed/train.pkl', 'rb') as f:
    train_data = pickle.load(f)

# Access samples
sample = train_data[0]
print(sample.keys())
# ['image_path', 'emotion', 'caption', 'painting_id']
```

**Create PyTorch DataLoader:**
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

# Iterate
for batch in loader:
    images = batch['image']      # (32, 3, 224, 224)
    emotions = batch['emotion']  # (32,)
    captions = batch['caption']  # list of 32 strings
```

### Data Augmentation

**Training augmentations:**
- Random horizontal flip (p=0.5)
- Random rotation (±15°)
- Color jitter (brightness, contrast, saturation)
- Random crop and resize

**Validation/test:**
- Center crop
- Resize to 224x224
- No random augmentations

### Cached Features

**Caching preprocessed features:**
```bash
# Cache image features (ResNet backbone)
python cerebrum_artis/data/cache_features.py \
    --split train \
    --output data/cache/image_features/train.h5

# Cache fuzzy features
python cerebrum_artis/fuzzy/cache_fuzzy_features.py \
    --split train \
    --output data/cache/fuzzy_features/train.h5
```

**Loading cached features:**
```python
import h5py

# Load cached features
with h5py.File('data/cache/image_features/train.h5', 'r') as f:
    image_features = f['features'][:]  # (64000, 2048)
    labels = f['labels'][:]            # (64000,)
```

**Benefits:**
- 10x faster training (no image loading/preprocessing)
- Reduced GPU memory usage
- Consistent features across runs

### Data Format

**CSV format (ArtEmis):**
```csv
art_style,painting,emotion,utterance,repetition
impressionism,starry_night.jpg,awe,"The swirling sky creates a sense of wonder",0
cubism,guernica.jpg,sadness,"The fragmented forms evoke pain and suffering",0
```

**Pickle format (processed):**
```python
{
    'image_path': 'data/artemis/raw/images/starry_night.jpg',
    'emotion': 1,  # awe (0-8 index)
    'emotion_name': 'awe',
    'caption': 'The swirling sky creates a sense of wonder',
    'painting_id': 'starry_night',
    'art_style': 'impressionism'
}
```

### Example Data

Sample data for quick testing:

```
data/examples/
├── sample_images/
│   ├── example_awe.jpg
│   ├── example_contentment.jpg
│   └── example_sadness.jpg
└── sample_annotations.json
```

**Quick test:**
```python
from cerebrum_artis.agents import PerceptoEmocional

agent = PerceptoEmocional()
result = agent.analyze_painting(
    image_path='data/examples/sample_images/example_awe.jpg',
    caption='A beautiful starry night'
)
print(result)
```

### Data License

**ArtEmis Dataset:**
- Licensed under CC BY-NC 4.0
- For research and educational purposes only
- Original paper: [ArtEmis: Affective Language for Visual Art](https://arxiv.org/abs/2101.07396)

**WikiArt Images:**
- Images are property of WikiArt.org
- Used for research purposes under fair use

### Citation

If you use the ArtEmis dataset, please cite:

```bibtex
@inproceedings{achlioptas2021artemis,
  title={ArtEmis: Affective Language for Visual Art},
  author={Achlioptas, Panos and Ovsjanikov, Maks and Haydarov, Kilichbek and Elhoseiny, Mohamed and Guibas, Leonidas},
  booktitle={CVPR},
  year={2021}
}
```

---

## Português

<details>
<summary>Clique para ver versão em português</summary>

### Visão Geral

Este diretório contém todos os arquivos relacionados a dados para treinamento e avaliação. O dataset principal usado é **ArtEmis** (Art Emotions in the Wild).

### Estrutura de Diretórios

```
data/
├── artemis/                    # Dataset ArtEmis
│   ├── raw/                    # Arquivos originais do dataset
│   │   ├── artemis_dataset_release_v0.csv
│   │   └── images/
│   ├── processed/              # Dados pré-processados
│   │   ├── train.pkl
│   │   ├── val.pkl
│   │   └── test.pkl
│   └── splits/                 # Splits train/val/test
│       ├── train_ids.txt
│       ├── val_ids.txt
│       └── test_ids.txt
├── cache/                      # Features em cache
│   ├── image_features/
│   ├── text_features/
│   └── fuzzy_features/
└── examples/                   # Dados de exemplo para testes
    ├── sample_images/
    └── sample_annotations.json
```

### Dataset ArtEmis

**Sobre:**
- 80.000+ triplas obra-emoção-descrição
- 9 categorias de emoção
- Descrições textuais ricas de respostas emocionais
- Imagens WikiArt (40.000+ obras)

**Categorias de Emoção:**
1. Amusement (Diversão)
2. Awe (Admiração)
3. Contentment (Contentamento)
4. Excitement (Excitação)
5. Anger (Raiva)
6. Disgust (Nojo)
7. Fear (Medo)
8. Sadness (Tristeza)
9. Something else (Algo mais)

**Estatísticas do Dataset:**
- Total de amostras: ~80.000
- Split de treino: 64.000 (80%)
- Split de validação: 8.000 (10%)
- Split de teste: 8.000 (10%)
- Comprimento médio de descrição: 15 palavras
- Imagens: formato JPEG, tamanhos variados

### Baixando o Dataset

**1. Baixar ArtEmis:**
```bash
cd data/artemis
wget https://www.artemisdataset.org/download/artemis_dataset_release_v0.csv
```

**2. Baixar imagens WikiArt:**
```bash
# Clonar o downloader WikiArt
git clone https://github.com/oprahhh/artemis-wikiart-downloader
cd artemis-wikiart-downloader

# Baixar imagens (requer ~20GB)
python download_images.py --output_dir ../data/artemis/raw/images
```

**3. Verificar download:**
```bash
# Checar arquivo CSV
wc -l data/artemis/raw/artemis_dataset_release_v0.csv
# Esperado: ~80.000 linhas

# Checar imagens
ls data/artemis/raw/images | wc -l
# Esperado: ~40.000 imagens
```

### Pré-processamento de Dados

**Pré-processar dataset:**
```bash
python cerebrum_artis/data/preprocess.py \
    --input_csv data/artemis/raw/artemis_dataset_release_v0.csv \
    --image_dir data/artemis/raw/images \
    --output_dir data/artemis/processed \
    --splits 0.8 0.1 0.1
```

**O que o pré-processamento faz:**
- Carrega e valida anotações CSV
- Filtra imagens inválidas/faltantes
- Cria splits train/val/test
- Salva arquivos pickle pré-processados
- Gera arquivos de IDs dos splits

### Usando os Dados

**Carregar dados pré-processados:**
```python
import pickle

# Carregar dados de treino
with open('data/artemis/processed/train.pkl', 'rb') as f:
    train_data = pickle.load(f)

# Acessar amostras
sample = train_data[0]
print(sample.keys())
# ['image_path', 'emotion', 'caption', 'painting_id']
```

**Criar PyTorch DataLoader:**
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

# Iterar
for batch in loader:
    images = batch['image']      # (32, 3, 224, 224)
    emotions = batch['emotion']  # (32,)
    captions = batch['caption']  # lista de 32 strings
```

### Licença dos Dados

**Dataset ArtEmis:**
- Licenciado sob CC BY-NC 4.0
- Apenas para fins de pesquisa e educação
- Paper original: [ArtEmis: Affective Language for Visual Art](https://arxiv.org/abs/2101.07396)

**Imagens WikiArt:**
- Imagens são propriedade de WikiArt.org
- Usadas para fins de pesquisa sob fair use

### Citação

Se você usar o dataset ArtEmis, por favor cite:

```bibtex
@inproceedings{achlioptas2021artemis,
  title={ArtEmis: Affective Language for Visual Art},
  author={Achlioptas, Panos and Ovsjanikov, Maks and Haydarov, Kilichbek and Elhoseiny, Mohamed and Guibas, Leonidas},
  booktitle={CVPR},
  year={2021}
}
```

</details>

---

**Directory**: `/home/paloma/cerebrum-artis/data/`  
**Last Updated**: November 25, 2025
