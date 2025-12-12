# Prompt para NotebookLM - Infográfico

Com base no documento "DEEP_LEARNING_DETAILS.md", crie um **infográfico visual** (poster acadêmico, 1 página A3/A2) sobre o projeto Cerebrum Artis.

## Estrutura do Infográfico

### CABEÇALHO (topo, 15% da altura)
**Título:** "Cerebrum Artis: Ensemble Multimodal para Classificação de Emoções em Arte"  
**Subtítulo:** Deep Learning | Transfer Learning | Multimodal Fusion  
**Logos:** Universidade, curso  
**Autora e data**

---

### COLUNA 1 (esquerda, 30% da largura): PROBLEMA & DADOS

**📊 O Desafio**
- Classificar emoções evocadas por pinturas
- 9 classes: contentment, amusement, awe, excitement, sadness, fear, disgust, anger, something else
- Dataset ArtEmis: 549k train, 68k val, 68k test
- Multimodal: Imagens (pinturas) + Texto (descrições)

**⚖️ Desbalanceamento**
```
Contentment: 21.57% ████████████
Sadness:     11.65% ██████
Amusement:   10.76% █████
...
Anger:        2.96% █
```

**🎯 Métricas**
Por que F1 > Accuracy?  
→ Dataset desbalanceado  
→ F1 = média harmônica precision/recall  
→ Avalia todas as classes igualmente

---

### COLUNA 2 (centro, 40% da largura): ARQUITETURAS

**🏗️ Pipeline Multimodal**

```
┌─────────┐           ┌──────────┐
│ Imagem  │──────────▶│ ResNet50 │──▶ [2048]
└─────────┘  frozen   └──────────┘
                                        │
                                        ▼
┌─────────┐           ┌──────────┐   ┌─────┐
│ Texto   │──────────▶│ RoBERTa  │──▶│ MLP │──▶ 9 classes
└─────────┘fine-tuned └──────────┘   └─────┘
              ▲                          ▲
              │         [768]            │
              └──────────────────────────┘
```

**V2: Concatenação Simples**
```
[2048] + [768] + [7] = [2823]
         ↓
    MLP: 1024 → 512 → 9
    Dropout: 0.3
    Params: 128M
```

**V3: Adaptive Gating**
```
Neural branch: [2816] → [9]
Fuzzy branch:  [7] → [9]
         ↓
α = f(cosine_similarity)
final = α×neural + (1-α)×fuzzy
```

**V4: Ensemble**
```
V2_probs ──┐
           ├──▶ weighted_avg ──▶ final
V3_probs ──┘     (50% / 50%)
```

---

### COLUNA 3 (direita, 30% da largura): RESULTADOS & INSIGHTS

**🏆 Performance (Test Set)**

| Modelo | F1 Score | Accuracy |
|--------|----------|----------|
| V2     | 65.61%   | 70.45%   |
| V3     | 65.47%   | 70.19%   |
| **V4** | **66.26%** | **70.97%** |

**📈 Ganho do Ensemble:** +0.79% F1

**✅ Generalização Excelente**
```
Validation: 66.44% F1
Test:       66.26% F1
Δ:          -0.18%  ← quase zero!
```

**🎭 Melhores Classes (F1)**
- Sadness: 82.42%
- Fear: 78.21%
- Contentment: 73.97%

**⚡ Classes Difíceis**
- Excitement: 54.61%
- Anger: 54.92%

**🔬 Ablation Study**
V3.1 (fuzzy integrado): **FALHOU**
- F1: 55.20% (-10 pontos!)
- Underfitting severo
- Lição: ensemble > monolítico

**🚀 vs Estado da Arte**
- Baseline (2021): 60% F1
- **Cerebrum V4**: 66.26% (+6.26%)
- SOTA (ViT): ~68-70%

---

### RODAPÉ (fundo, 10% da altura): TÉCNICAS & CONCLUSÕES

**🧠 Técnicas de Deep Learning Aplicadas:**
✓ Transfer Learning (ImageNet→Arte)  
✓ Fine-tuning (RoBERTa)  
✓ Multimodal Fusion  
✓ Attention Mechanisms  
✓ Residual Connections  
✓ Ensemble Methods  
✓ Regularização (Dropout 0.3 + Weight Decay 0.01 + Early Stop)

**💡 Principais Conclusões:**
1. Transfer learning essencial (ResNet + RoBERTa)
2. Ensemble sempre melhora performance
3. Simplicidade vence complexidade (V2 > V3.1)
4. F1 Score > Accuracy para dados desbalanceados
5. Regularização crucial para generalização

**⏱️ Custo Computacional:** 144h treinamento total | 128M params/modelo | GPU NVIDIA A100

---

## Instruções Visuais

**Paleta de Cores:**
- V2: Azul (#2E86DE)
- V3: Verde (#27AE60)
- V4: Roxo (#8E44AD)
- Destaque resultados: Laranja (#E67E22)
- Texto: Cinza escuro (#2C3E50)

**Elementos Gráficos:**
- Diagramas de arquitetura com setas grossas
- Ícones: 🎨 (arte), 📊 (dados), 🧠 (DL), 🏆 (resultados)
- Gráficos de barras para comparações
- Boxes coloridos para destacar números importantes
- Fontes: Sans-serif moderna (Roboto, Inter)

**Hierarquia:**
- Números grandes: 72pt (66.26%, +6.26%)
- Títulos seção: 36pt
- Texto corpo: 18-24pt
- Legendas: 14pt

**Layout:**
- 3 colunas balanceadas
- Espaçamento generoso (evitar poluição)
- Bordas arredondadas em boxes
- Sombras leves para profundidade

**Dados Visuais Prioritários:**
1. Pipeline multimodal (diagrama central)
2. Tabela de resultados (destaque V4)
3. Gráfico de barras desbalanceamento
4. Comparação val vs test (generalização)
5. Técnicas DL aplicadas (checklist)

---

**Objetivo:** Infográfico acadêmico profissional, visualmente atraente, com foco em resultados e arquiteturas. Deve ser compreensível em 2-3 minutos, destacando métricas (66.26% F1) e técnicas (ensemble, transfer learning).
