# Deep-Mind: Neural Emotion Classifier

Sistema de classificação multimodal de emoções para arte usando Imagem + Texto.

## 🏗️ Arquitetura

```
IMAGE → ResNet50 (2048-dim)  ─┐
                              ├─→ Fusion MLP → Softmax(9 emotions)
TEXT  → RoBERTa (768-dim)    ─┘
```

## 📦 Instalação

```bash
# Criar ambiente (se ainda não tiver)
conda create -n deep-mind python=3.10
conda activate deep-mind

# Instalar PyTorch (CUDA 11.8 - ajuste conforme sua GPU)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Instalar dependências
pip install -r requirements.txt
```

## 🚀 Treino

### Quick Start (padrões otimizados)

```bash
cd /home/paloma/cerebrum-artis/deep-mind
python train_emotion_classifier.py --epochs 15 --batch-size 64 --gpu 0
```

### Configuração Completa

```bash
python train_emotion_classifier.py \
  --csv-path /home/paloma/cerebrum-artis/artemis/dataset/official_data/combined_artemis.csv \
  --img-dir /data/paloma/data/paintings/wikiart \
  --epochs 15 \
  --batch-size 64 \
  --lr 2e-5 \
  --dropout 0.3 \
  --gpu 0 \
  --num-workers 4 \
  --save-dir checkpoints
```

### Hiperparâmetros

| Parâmetro | Padrão | Descrição |
|-----------|--------|-----------|
| `--epochs` | 15 | Número de épocas |
| `--batch-size` | 64 | Tamanho do batch |
| `--lr` | 2e-5 | Learning rate (AdamW) |
| `--weight-decay` | 1e-4 | Weight decay |
| `--dropout` | 0.3 | Dropout rate no MLP |
| `--freeze-image` | True | Congelar ResNet50 |
| `--freeze-text` | False | Fine-tune RoBERTa |
| `--val-split` | 0.1 | Fração para validação |
| `--test-split` | 0.1 | Fração para teste |

## 📊 Performance Esperada

- **Acurácia**: 75-82% no test set
- **Tempo de treino**: ~2-3 horas (Tesla V100)
- **Params treinables**: ~85M (RoBERTa) + 3M (Fusion MLP)

## 📁 Estrutura de Outputs

```
checkpoints/
└── multimodal_YYYYMMDD_HHMMSS/
    ├── config.json              # Configuração do treino
    ├── checkpoint_epoch5.pt     # Checkpoints periódicos
    ├── checkpoint_epoch5_best.pt # Melhor modelo
    ├── test_report.txt          # Métricas finais
    ├── confusion_matrix.npy     # Matriz de confusão
    └── tb_logs/                 # TensorBoard logs
```

## 🔬 Uso do Modelo Treinado

```python
from multimodal_classifier import load_multimodal_classifier
from PIL import Image
import torch

# Carregar modelo
model, tokenizer = load_multimodal_classifier(
    'checkpoints/multimodal_20251119_043000/checkpoint_epoch15_best.pt',
    device='cuda'
)

# Preparar input
image = Image.open('painting.jpg').convert('RGB')
text = "This painting makes me feel contemplative and serene"

# Preprocessar imagem
from torchvision import transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
])
image_tensor = transform(image).unsqueeze(0).to('cuda')

# Tokenizar texto
encoding = tokenizer(text, max_length=128, padding='max_length',
                    truncation=True, return_tensors='pt')
input_ids = encoding['input_ids'].to('cuda')
attention_mask = encoding['attention_mask'].to('cuda')

# Predição
with torch.no_grad():
    probs = model.predict_proba(image_tensor, input_ids, attention_mask)
    
emotions = ['amusement', 'awe', 'contentment', 'excitement',
           'anger', 'disgust', 'fear', 'sadness', 'something else']

for emo, prob in zip(emotions, probs[0]):
    print(f"{emo}: {prob:.4f}")
```

## 📈 Monitoramento (TensorBoard)

```bash
tensorboard --logdir checkpoints/multimodal_YYYYMMDD_HHMMSS/tb_logs --port 6006
```

## 🔧 Troubleshooting

### CUDA Out of Memory
- Reduzir `--batch-size` (tente 32, 16)
- Aumentar `--num-workers` para balancear CPU/GPU

### Slow Training
- Verificar se `--freeze-image=True` (ResNet50 congelado)
- Aumentar `--num-workers` (4-8 recomendado)

### Low Accuracy
- Aumentar epochs (20-25)
- Descongelar ResNet50: `--freeze-image=False --lr 1e-5`
- Reduzir dropout: `--dropout 0.2`

## 📚 Dataset (ArtEmis)

O modelo espera CSV com colunas:
- `art_style`: Estilo da arte (para path)
- `painting`: Nome do arquivo (sem .jpg)
- `utterance`: Texto explicativo da emoção
- `emotion`: Label (amusement, awe, contentment, etc.)

## 🎯 Próximos Passos

1. ✅ Treinar modelo base
2. ⏳ Integrar com fuzzy-brain
3. ⏳ Implementar ensemble neural-fuzzy
4. ⏳ Avaliar em WikiArt completo
