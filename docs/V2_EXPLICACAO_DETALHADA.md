# 🧠 V2 Fuzzy Features - Explicação Detalhada

## 📊 Visão Geral Arquitetural

### ⚠️ IMPORTANTE: TREINO/TESTE vs INFERÊNCIA REAL

**Durante TREINO/TESTE**: Ambos (imagem + utterance) vêm do dataset ArtEmis  
**Durante INFERÊNCIA REAL**: Depende se usuário fornece utterance ou não

---

### 🎓 CENÁRIO 1: Treino/Validação (Dataset ArtEmis)

```
┌─────────────────────────────────────────────────────────────────────┐
│              📁 DATASET ARTEMIS (já tem tudo pronto)                │
│  painting.jpg + "This painting makes me feel sad" + label=sadness  │
│                                                                     │
│  ⚠️ A utterance JÁ EXISTE! Foi escrita por um humano real          │
│     que olhou a pintura e descreveu o que sentiu                   │
└────────────────────────────┬────────────────────────────────────────┘
                             │
        ┌────────────────────┴────────────────────┐
        │                                         │
        ▼                                         ▼
┌───────────────────┐                    ┌────────────────────┐
│  BRANCH VISUAL    │                    │   BRANCH TEXTUAL   │
│   (ResNet50)      │                    │    (RoBERTa)       │
└─────────┬─────────┘                    └──────────┬─────────┘
          │                                         │
          │ [B, 3, 224, 224]                       │ tokens
          │                                         │
          ▼                                         ▼
┌─────────────────────┐              ┌──────────────────────────┐
│  CNN Convolutions   │              │  Transformer Encoder     │
│  (5 blocos ResNet)  │              │  (12 camadas attention)  │
└─────────┬───────────┘              └────────────┬─────────────┘
          │                                       │
          ▼                                       ▼
    [B, 2048]                              [B, 768]
    visual_feats                           text_feats
          │                                       │
          │                                       │
          │       ┌─────────────────────┐        │
          │       │  FUZZY EXTRACTOR    │        │
          │       │  (7 features)       │        │
          │       └──────────┬──────────┘        │
          │                  │                   │
          │                  ▼                   │
          │            [B, 7]                    │
          │         fuzzy_features               │
          │                  │                   │
          └──────────────────┴───────────────────┘
                             │
                             ▼
                   ┌──────────────────┐
                   │  CONCATENAÇÃO    │
                   │  [2048+768+7]    │
                   └────────┬─────────┘
                            │
                            ▼
                      [B, 2823]
                            │
                            ▼
                   ┌──────────────────┐
                   │   MLP FUSION     │
                   │  2823→1024→512→9 │
                   └────────┬─────────┘
                            │
                            ▼
                       [B, 9]
                       logits
                            │
                            ▼
                   ┌──────────────────┐
                   │    SOFTMAX       │
                   └────────┬─────────┘
                            │
                            ▼
                   Probabilidades
                   [0.02, 0.65, ...]
                            │
                            ▼
                      sadness (65%)
```

---

### 🚀 CENÁRIO 2: Inferência Real (Pintura Nova)

#### **2A: Usuário FORNECE utterance**
```
┌─────────────────────────────────────────────────────────────────┐
│  ENTRADA: nova_pintura.jpg + usuário digita texto              │
│  "This abstract painting makes me feel confused and curious"   │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                  (FLUXO IDÊNTICO AO TREINO)
                           │
                           ▼
                      Emoção predita
```

#### **2B: Usuário NÃO FORNECE utterance (só imagem)**
```
┌────────────────────────────────────────────────────────────────┐
│  ENTRADA: nova_pintura.jpg (SEM utterance)                     │
└──────────────────────┬─────────────────────────────────────────┘
                       │
                       ├─→ OPÇÃO A: Gerar utterance com SAT
                       │   sat_caption = "A painting with dark colors..."
                       │   model(image, sat_caption) → emoção
                       │
                       └─→ OPÇÃO B: Usar só visual features
                           (requer modelo re-treinado sem texto)
                           visual_only_model(image) → emoção
```

**⚠️ NOTA**: O V2 atual **EXIGE** utterance! Para funcionar sem texto, seria necessário:
- Usar SAT para gerar caption automática, OU
- Retreinar modelo sem branch textual (só visual + fuzzy)

---

## 🔍 Fluxo Algorítmico Passo-a-Passo

### **ETAPA 1: PRÉ-PROCESSAMENTO DA IMAGEM**

```python
# Input: imagem RGB (altura variável × largura variável × 3 canais)
# Exemplo: starry_night.jpg (768×960×3)

1. Carregar imagem do disco
   image = PIL.Image.open("starry_night.jpg").convert('RGB')

2. Aplicar transformações (ImageNet normalization)
   transforms = Compose([
       Resize(256),              # Redimensiona menor lado para 256px
       CenterCrop(224),          # Corta centro 224×224
       ToTensor(),               # Converte para [0,1] e formato [C,H,W]
       Normalize(                # Normaliza com média/std do ImageNet
           mean=[0.485, 0.456, 0.406],
           std=[0.229, 0.224, 0.225]
       )
   ])
   
   image_tensor = transforms(image)  # [3, 224, 224]

3. Adicionar dimensão de batch
   image_batch = image_tensor.unsqueeze(0)  # [1, 3, 224, 224]
```

**Output**: Tensor `[B, 3, 224, 224]` normalizado

---

### **ETAPA 2: EXTRAÇÃO DE FEATURES VISUAIS (ResNet50)**

```python
# ResNet50 pré-treinada no ImageNet (frozen weights)

1. Layer 1: Conv inicial + Batch Norm + ReLU + MaxPool
   [1, 3, 224, 224] → [1, 64, 56, 56]
   
2. Layer 2: Residual Block 1 (3 blocos)
   [1, 64, 56, 56] → [1, 256, 56, 56]
   
   Cada bloco:
   - Conv 1×1 (reduz dimensões)
   - Conv 3×3 (feature extraction)
   - Conv 1×1 (expande dimensões)
   - Skip connection (adiciona input ao output)
   
3. Layer 3: Residual Block 2 (4 blocos, downsampling)
   [1, 256, 56, 56] → [1, 512, 28, 28]
   
4. Layer 4: Residual Block 3 (6 blocos, downsampling)
   [1, 512, 28, 28] → [1, 1024, 14, 14]
   
5. Layer 5: Residual Block 4 (3 blocos, downsampling)
   [1, 1024, 14, 14] → [1, 2048, 7, 7]
   
6. Global Average Pooling
   [1, 2048, 7, 7] → [1, 2048, 1, 1]
   
7. Flatten
   [1, 2048, 1, 1] → [1, 2048]
```

**Output**: `visual_feats` = `[B, 2048]`

**Interpretação**: Vetor denso de 2048 dimensões representando características visuais de alto nível (formas, texturas, composição, cores abstratas)

---

### **ETAPA 3: EXTRAÇÃO DE FUZZY FEATURES**

#### **3.1. Cálculo das 7 Features Crisp**

```python
# Input: imagem PIL original (antes das transformações)

import cv2
import numpy as np

1. BRIGHTNESS (Brilho médio)
   hsv = cv2.cvtColor(image_np, cv2.COLOR_RGB2HSV)
   brightness = hsv[:, :, 2].mean() / 255.0  # [0, 1]
   
   Exemplo: Starry Night → brightness = 0.35 (pintura escura)

2. COLOR_TEMPERATURE (Quente vs Frio)
   r_mean = image_np[:, :, 0].mean()
   b_mean = image_np[:, :, 2].mean()
   temp = (r_mean - b_mean) / 255.0  # [-1, 1]
   color_temperature = (temp + 1) / 2  # Normaliza para [0, 1]
   
   Exemplo: Starry Night → 0.52 (neutro, azul+amarelo equilibrados)

3. SATURATION (Intensidade das cores)
   saturation = hsv[:, :, 1].mean() / 255.0  # [0, 1]
   
   Exemplo: Starry Night → 0.68 (cores vívidas)

4. COLOR_HARMONY (Diversidade de matizes)
   hue_std = hsv[:, :, 0].std()
   harmony = np.exp(-hue_std / 50.0)  # [0, 1], maior = mais harmônico
   
   Exemplo: Starry Night → 0.45 (paleta diversa)

5. COMPLEXITY (Entropia de gradientes)
   gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
   gradients = cv2.Sobel(gray, cv2.CV_64F, 1, 1)
   complexity = np.std(gradients) / 100.0  # Normalizado
   
   Exemplo: Starry Night → 0.72 (pinceladas complexas)

6. SYMMETRY (Simetria vertical)
   left_half = image_np[:, :width//2]
   right_half = np.fliplr(image_np[:, width//2:])
   diff = np.abs(left_half - right_half).mean()
   symmetry = 1.0 - (diff / 255.0)  # [0, 1]
   
   Exemplo: Starry Night → 0.42 (assimétrica)

7. TEXTURE_ROUGHNESS (Rugosidade da textura)
   laplacian = cv2.Laplacian(gray, cv2.CV_64F)
   roughness = np.std(laplacian) / 50.0  # Normalizado
   
   Exemplo: Starry Night → 0.78 (textura rugosa, pinceladas visíveis)
```

**Output**: `crisp_features` = `[0.35, 0.52, 0.68, 0.45, 0.72, 0.42, 0.78]`

---

#### **3.2. Fuzzificação (Crisp → Fuzzy)**

```python
# Para cada feature, aplicar funções de pertinência (membership functions)

Exemplo com BRIGHTNESS = 0.35:

1. Definir termos linguísticos (5 conjuntos fuzzy triangulares)
   - muito_escuro:  trimf(x, [0.0, 0.0, 0.2])
   - escuro:        trimf(x, [0.1, 0.3, 0.5])
   - medio:         trimf(x, [0.4, 0.6, 0.8])
   - claro:         trimf(x, [0.7, 0.9, 1.0])
   - muito_claro:   trimf(x, [0.9, 1.0, 1.0])

2. Calcular grau de pertinência para x=0.35
   trimf(x, [a, b, c]) = max(0, min((x-a)/(b-a), (c-x)/(c-b)))
   
   muito_escuro(0.35) = 0.0      # fora do triângulo
   escuro(0.35)       = 0.75     # 75% pertence a "escuro"
   medio(0.35)        = 0.0      # fora do triângulo
   claro(0.35)        = 0.0
   muito_claro(0.35)  = 0.0

3. Repetir para todas as 7 features
   brightness_fuzzy     = [0.00, 0.75, 0.00, 0.00, 0.00]
   color_temp_fuzzy     = [0.00, 0.00, 1.00, 0.00, 0.00]
   saturation_fuzzy     = [0.00, 0.00, 0.40, 0.60, 0.00]
   harmony_fuzzy        = [0.00, 0.25, 0.75, 0.00, 0.00]
   complexity_fuzzy     = [0.00, 0.00, 0.00, 0.60, 0.40]
   symmetry_fuzzy       = [0.00, 0.80, 0.20, 0.00, 0.00]
   roughness_fuzzy      = [0.00, 0.00, 0.00, 0.40, 0.60]
   
   Total: 7 features × 5 termos = 35 valores fuzzy
```

**IMPORTANTE**: No V2, **NÃO usamos as regras fuzzy**! 

O sistema fuzzy completo (com regras de inferência) só é usado no **V3** e **V3.1**.

No **V2**, fazemos algo mais simples:

```python
# V2: Apenas usa os 7 valores CRISP como features extras

fuzzy_features = torch.tensor([
    0.35,  # brightness (valor crisp)
    0.52,  # color_temperature
    0.68,  # saturation
    0.45,  # color_harmony
    0.72,  # complexity
    0.42,  # symmetry
    0.78   # texture_roughness
], dtype=torch.float32)

# Shape: [7]
```

**Output**: `fuzzy_features` = `[B, 7]` (valores normalizados [0,1])

---

### **ETAPA 4: PROCESSAMENTO DO TEXTO (RoBERTa)**

```python
# Input: utterance = "This painting makes me feel sad and lonely"

1. Tokenização
   tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
   
   tokens = tokenizer(
       "This painting makes me feel sad and lonely",
       max_length=128,
       padding='max_length',
       truncation=True,
       return_tensors='pt'
   )
   
   # Output:
   input_ids = [0, 713, 8376, 817, 162, 619, 5074, 8, 14142, 2, 1, 1, ...]
   #            [CLS] This painting makes me feel sad and lonely [SEP] [PAD]...
   
   attention_mask = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, ...]
   #                 [atenção nos tokens reais, ignora padding]

2. Embedding Layer
   [B, 128] (índices) → [B, 128, 768] (embeddings densos)

3. Transformer Encoder (12 camadas)
   Cada camada:
   a) Multi-Head Self-Attention (12 cabeças)
      - Query, Key, Value projections
      - Attention weights = softmax(QK^T / √d)
      - Context vectors = Attention × V
      
   b) Feed-Forward Network
      - Linear(768 → 3072) + GELU
      - Linear(3072 → 768)
      
   c) Layer Normalization + Residual Connections
   
   [B, 128, 768] → [B, 128, 768] (12× camadas)

4. Extração do token [CLS]
   text_feats = last_hidden_state[:, 0, :]  # [B, 768]
   
   # O token [CLS] é treinado para representar todo o sentido da frase
```

**Output**: `text_feats` = `[B, 768]`

**Interpretação**: Vetor denso representando o significado semântico completo do utterance

---

### **ETAPA 5: FUSÃO MULTIMODAL (MLP)**

```python
# Concatenar os 3 vetores

combined = torch.cat([
    visual_feats,      # [B, 2048]
    text_feats,        # [B, 768]
    fuzzy_features     # [B, 7]
], dim=1)

# combined shape: [B, 2823]

# MLP de 3 camadas com dropout

1. Camada 1: Compressão inicial
   x = Linear(2823, 1024)(combined)  # [B, 1024]
   x = ReLU(x)
   x = Dropout(0.3)(x)
   
2. Camada 2: Compressão intermediária
   x = Linear(1024, 512)(x)          # [B, 512]
   x = ReLU(x)
   x = Dropout(0.3)(x)
   
3. Camada 3: Classificação
   logits = Linear(512, 9)(x)        # [B, 9]
```

**Output**: `logits` = `[B, 9]` (scores brutos, não normalizados)

Exemplo: `[-2.3, 1.8, -0.5, 0.2, -1.1, 3.5, -0.8, 2.1, -1.9]`

---

### **ETAPA 6: SOFTMAX E PREDIÇÃO FINAL**

```python
# Converter logits em probabilidades

probs = torch.softmax(logits, dim=1)

# Softmax formula: P(y=k|x) = exp(logit_k) / Σ exp(logit_j)

# Exemplo:
logits = [-2.3, 1.8, -0.5, 0.2, -1.1, 3.5, -0.8, 2.1, -1.9]

probs = [
    0.01,  # amusement
    0.06,  # awe
    0.06,  # contentment
    0.12,  # excitement
    0.03,  # anger
    0.33,  # disgust ← maior prob (mas não é a emoção correta)
    0.04,  # fear
    0.81,  # sadness ← CORRETO! (label verdadeiro)
    0.01   # something else
]

# Predição final
predicted_class = torch.argmax(probs)  # 7 (sadness)
confidence = probs[predicted_class]    # 0.81 (81%)
```

**Output**: 
- Emoção prevista: `sadness`
- Confiança: `81%`

---

## 📐 Dimensões dos Tensores em Cada Etapa

| Etapa | Tensor | Shape | Tamanho |
|-------|--------|-------|---------|
| Input Image | `image` | `[B, 3, 224, 224]` | 150,528 valores |
| ResNet Output | `visual_feats` | `[B, 2048]` | 2,048 valores |
| Fuzzy Features | `fuzzy_features` | `[B, 7]` | 7 valores |
| RoBERTa Input | `input_ids` | `[B, 128]` | 128 tokens |
| RoBERTa Output | `text_feats` | `[B, 768]` | 768 valores |
| Concatenação | `combined` | `[B, 2823]` | 2,823 valores |
| MLP Camada 1 | `hidden1` | `[B, 1024]` | 1,024 valores |
| MLP Camada 2 | `hidden2` | `[B, 512]` | 512 valores |
| Logits | `logits` | `[B, 9]` | 9 scores |
| Probabilidades | `probs` | `[B, 9]` | 9 probs (soma=1) |

**B** = batch size (normalmente 32)

---

## 🎯 Exemplo Concreto: "Starry Night" + "This makes me feel awe"

### Entrada
```
Imagem: starry_night.jpg (Van Gogh)
Texto: "This painting makes me feel awe and wonder"
Label: awe (classe 1)
```

### Processamento

**1. Features Visuais (ResNet50)**
```python
visual_feats = [0.23, -0.15, 0.89, ..., 0.42, -0.11, 0.67]  # 2048 dims
# Representa: pinceladas swirling, céu noturno, contraste alto, 
#             composição dinâmica, cores azul/amarelo
```

## 🤔 FAQ: Perguntas Frequentes

### **P1: O diagrama mostra TREINO ou INFERÊNCIA?**
**R**: Mostra **AMBOS!** Durante treino/validação, a utterance vem do dataset. Durante inferência real, depende se o usuário fornece texto ou não.

### **P2: No teste a gente também tem utterance?**
**R**: **SIM!** O dataset ArtEmis tem utterances para TODOS os 80k+ exemplos (treino + validação + teste). O modelo SEMPRE recebe imagem + texto durante avaliação.

### **P3: E se eu quiser usar uma pintura nova sem utterance?**
**R**: Você tem 2 opções:
1. **Usar SAT**: Gerar caption automática e passar pro modelo
2. **Retreinar sem texto**: Criar versão visual-only (ResNet + Fuzzy apenas)

### **P4: O SAT é usado no V2?**
**R**: **NÃO!** O SAT é usado apenas:
- No **agente Explicador** (gerar descrições visuais)
- Em **inferência real** quando não há utterance do usuário
- **NUNCA** durante treino/validação (utterances já existem no dataset)

---

## 🔬 Papel das Fuzzy Features no V2

**Pergunta**: Se o V2 não usa regras fuzzy, qual a vantagem das fuzzy features?
    0.35,  # brightness: escuro (noite)
    0.52,  # color_temp: neutro (azul frio + amarelo quente)
    0.68,  # saturation: alta (cores vívidas)
    0.45,  # harmony: média (paleta contrastante)
    0.72,  # complexity: alta (pinceladas turbulentas)
    0.42,  # symmetry: baixa (composição assimétrica)
    0.78   # roughness: alta (impasto, textura visível)
]
```

**3. Features Textuais (RoBERTa)**
```python
text_feats = [-0.08, 0.34, -0.21, ..., 0.19, -0.45, 0.12]  # 768 dims
# Representa: sentimento positivo ("awe", "wonder"),
#             admiração, escala grandiosa, emoção elevada
```

**4. Fusão e Classificação**
```python
combined = concat(visual_feats, text_feats, fuzzy_features)  # [2823]

# MLP processa e gera scores
logits = [-1.2, 4.5, 0.8, 1.3, -2.1, -0.9, -1.5, 0.2, -3.0]

# Softmax
probs = [0.01, 0.87, 0.03, 0.04, 0.00, 0.01, 0.00, 0.02, 0.00]
#        amus   AWE   cont  exci  ang   disg  fear  sad   else
#              ^^^^
#              87% confiança em AWE → CORRETO!
```

---

## 🔬 Papel das Fuzzy Features no V2

**Pergunta**: Se o V2 não usa regras fuzzy, qual a vantagem das fuzzy features?

**Resposta**: As 7 features fuzzy adicionam **informação interpretável de baixo nível** que complementa as features de alto nível da ResNet:

| Feature | O que a ResNet "vê" | O que Fuzzy adiciona |
|---------|---------------------|----------------------|
| **Brightness** | Padrões de luz abstratos | Valor médio objetivo [0,1] |
| **Saturation** | Cores em contexto | Intensidade cromática pura |
| **Complexity** | Texturas aprendidas | Métrica objetiva de gradientes |
| **Symmetry** | Composição implícita | Simetria explícita calculada |

**Resultado**: A MLP aprende a **combinar** features de alto nível (ResNet) com métricas objetivas (fuzzy) para decisões mais robustas.

**Ganho de performance**: +3.04% (67.59% → 70.63%)

---

## ⚙️ Parâmetros Treináveis

| Componente | Params | Status |
|------------|--------|--------|
| ResNet50 | ~23M | **Frozen** (não treina) |
| RoBERTa | ~125M | **Frozen** (não treina) |
| MLP Fusion | ~3M | **Trainable** ✅ |
| **Total** | **~3M trainable** | - |

**Estratégia**: Transfer learning - aproveita conhecimento pré-treinado e só treina a camada de fusão.

---

## 📊 Treinamento

```python
# Loss function
criterion = CrossEntropyLoss()

# Optimizer
optimizer = AdamW(
    model.fusion.parameters(),  # Só otimiza a MLP
    lr=2e-5,
    weight_decay=0.01
)

# Training loop
for epoch in range(20):
    for batch in train_loader:
        # Forward pass
        logits = model(
            batch['image'],
            batch['input_ids'],
            batch['attention_mask'],
            batch['fuzzy_features']
        )
        
        # Compute loss
        loss = criterion(logits, batch['label'])
        
        # Backward pass
        loss.backward()
        
        # Update weights
        optimizer.step()
        optimizer.zero_grad()
```

**Resultado**:
- Época 3: 70.63% val accuracy (melhor)
- Early stopping: parou na época 3
- Treinamento: ~6 horas em GPU RTX 3090

---

## 🎨 Exemplo Visual do Fluxo

```
🖼️ INPUT
   │
   ├─→ ResNet50 ─────────→ [2048 visual features]
   │                       (formas, composição, cores abstratas)
   │
   ├─→ Fuzzy Extractor ──→ [7 fuzzy features]
   │                       (brilho=0.35, saturação=0.68, ...)
   │
   └─→ RoBERTa ──────────→ [768 text features]
                          (sentimento, contexto semântico)
                          
                          ↓ CONCATENATE
                          
                      [2823 combined]
                          
                          ↓ MLP
                          
                    [9 probabilities]
                    
                    amusement:  1%
                    awe:        6%
                    contentment:6%
                    excitement:12%
                    anger:      3%
                    disgust:   33%
                    fear:       4%
                    sadness:   81% ← WINNER!
                    else:       1%
```

---

## 🔑 Diferenças V2 vs V3 vs V3.1

| Aspecto | V2 | V3 | V3.1 |
|---------|----|----|------|
| **Fuzzy Features** | ✅ 7 valores crisp | ✅ 7 valores crisp | ✅ 7 valores crisp |
| **Regras Fuzzy** | ❌ Não usa | ✅ 18 regras Mamdani | ✅ 18 regras Mamdani |
| **Gating** | ❌ Simples concat | ✅ Adaptive (external) | ✅ Integrated (internal) |
| **Arquitetura** | Concat + MLP | Concat + Gating + MLP | Forward integrado |
| **Performance** | 70.63% | 70.37% | 70.40% |

---

## 💡 Por que V2 funciona?

1. **Transfer Learning**: Aproveita conhecimento ImageNet + roberta
2. **Multimodalidade**: Combina visual + texto = contexto completo
3. **Features interpretáveis**: 7 métricas objetivas ajudam a MLP
4. **Simplicidade**: Arquitetura direta, fácil de treinar e debugar
5. **Regularização**: Dropout previne overfitting

---

## 📝 Resumo Executivo

**V2 = ResNet50 (frozen) + RoBERTa (frozen) + 7 Fuzzy Features + MLP (trainable)**

**Pipeline**:
1. Imagem → ResNet50 → `[2048]`
2. Imagem → Fuzzy Extractor → `[7]`
3. Texto → RoBERTa → `[768]`
4. Concatenar → `[2823]`
5. MLP (3 camadas) → `[9]`
6. Softmax → Probabilidades → Predição

**Resultado**: 70.63% accuracy (melhor modelo individual)

**Vantagens**:
- ✅ Simples e eficiente
- ✅ Features interpretáveis
- ✅ Transfer learning robusto

**Limitações**:
- ⚠️ Não usa raciocínio fuzzy completo (isso vem no V3)
- ⚠️ Fusão fixa (não adaptativa)
