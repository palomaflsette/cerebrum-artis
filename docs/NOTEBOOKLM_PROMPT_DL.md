# Prompt para NotebookLM - Apresentação Deep Learning

Com base no documento "DEEP_LEARNING_DETAILS.md", crie uma apresentação técnica (15-20min) para disciplina de Deep Learning focando em redes neurais, arquiteturas e otimização.

## Slides (12 total)

**SLIDE 1: Título**  
"Cerebrum Artis: Classificação Multimodal de Emoções em Arte com Deep Learning"

**SLIDE 2: Problema**  
Dataset ArtEmis: 549k treino, 68k val/test. 9 classes de emoções. Modalidades: Imagem + Texto. Desafio: desbalanceamento (21.57% vs 2.96%).

**SLIDE 3: Transfer Learning Base**  
ResNet50 (frozen, 2048 dims) + RoBERTa-base (fine-tuned, 768 dims). Por quê? Skip connections, transformer attention, features pré-treinadas.

**SLIDE 4: V2 - Concatenação**  
[2048] + [768] + [7 fuzzy] = [2823] → MLP (1024→512→9). Dropout 0.3, 128M params. Simples e efetivo.

**SLIDE 5: V3 - Adaptive Gating**  
Peso adaptativo via cosine similarity: `α = f(agreement)`, `final = α×neural + (1-α)×fuzzy`. Combina branches dinamicamente.

**SLIDE 6: V4 - Ensemble**  
Weighted average: `probs = 0.5×V2 + 0.5×V3`. Inference-only, combina precision (V2) e recall (V3).

**SLIDE 7: Treinamento**  
AdamW (lr=1e-5), ReduceLROnPlateau, Early Stop (patience=5), CrossEntropy. Regularização: Dropout 0.3 + Weight Decay 0.01.

**SLIDE 8: Resultados**  
```
       V2      V3      V4
F1    65.61%  65.47%  66.26% ← melhor
Acc   70.45%  70.19%  70.97%
```
Generalização: val→test = -0.18% F1 (excelente).

**SLIDE 9: Ablation - V3.1 Falhou**  
Hipótese: fuzzy integrado. Resultado: F1=55.20% (-10 pts!). Problemas: underfitting (60% train acc), conflito gradientes. Lição: ensemble > monolítico.

**SLIDE 10: Técnicas DL**  
✓ Transfer Learning ✓ Multimodal Fusion ✓ Attention (RoBERTa) ✓ Residual (ResNet) ✓ Ensemble ✓ Dropout + Weight Decay ✓ Fine-tuning

**SLIDE 11: SOTA Comparison**  
Baseline: 60% → V4: 66.26% (+6.26%). Competitivo com Vision Transformers (~68-70%).

**SLIDE 12: Conclusões**  
✅ Transfer learning funciona  
✅ Ensemble melhora (+0.79%)  
✅ Regularização essencial  
✅ F1 > Accuracy (desbalanceamento)  
💡 Simplicidade > Complexidade  

---

**Instruções:** Use diagramas de arquitetura, destaque números importantes, cores por modelo (V2=azul, V3=verde, V4=roxo). Foco 80% DL / 20% fuzzy. Narrativa: problema → arquiteturas → resultados → ablation → conclusões.
