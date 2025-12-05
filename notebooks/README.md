# Analysis Notebooks

Este diretório contém notebooks Jupyter para análise completa do sistema Cerebrum Artis de classificação de emoções em arte.

## 📚 Notebooks (em ordem de execução)

### 0. `00_data_exploration_features.ipynb` - **Exploração e Feature Engineering**
**Análise exploratória completa e extração de features.**

Cobre:
- Carregamento e overview do dataset ArtEmis
- Distribuição de emoções
- Análise de captions (comprimento, palavras-chave)
- Extração das 7 features fuzzy (warmth, coldness, saturation, mutedness, brightness, darkness, harmony)
- Correlações entre features e emoções
- Visualizações exploratórias

**Outputs**:
- `outputs/figures/eda/emotion_distribution.png`
- `outputs/figures/eda/caption_length_distribution.png`
- `outputs/figures/features/feature_distributions.png`
- `outputs/figures/features/feature_emotion_heatmap.png`
- `outputs/figures/features/feature_correlation_matrix.png`
- `outputs/tables/feature_statistics.csv`
- `outputs/tables/feature_emotion_means.csv`

---

### 1. `01_model_evaluation.ipynb` - **Treinamento e Avaliação de Modelos**
**Avaliação detalhada de todas as versões (V1-V4).**

Cobre:
- Avaliação de cada modelo no test set
- Métricas: accuracy, precision, recall, F1
- Confusion matrices
- Comparação entre versões
- Análise de performance por classe

**Outputs**:
- `outputs/figures/model_comparison.png`
- `outputs/figures/confusion_matrix_v3.png`
- `outputs/tables/model_comparison.csv`
- `outputs/metrics/model_evaluation.json`

---

### 2. `02_agents_demo.ipynb` - **Demonstração dos Agentes**
**Pipeline completo com os 3 agentes trabalhando juntos.**

Demonstra:
- **PerceptoEmocional**: Classificação de emoções (V4 ensemble: V2+V3+V3.1)
- **Colorista**: Análise de paleta de cores dominantes
- **Explicador**: Explicações textuais + visuais (Grad-CAM)
- Pipeline integrado de análise completa

**Outputs**:
- `outputs/explanation_gradcam.png`
- `outputs/complete_analysis.png`
- Demonstração interativa de cada agente

---

### 3. `03_ensemble_analysis.ipynb` (TODO)
**Análise de estratégias de ensemble.**

Cobre:
- Diferentes métodos de ensemble (voting, averaging, weighted)
- Otimização de pesos do ensemble
- Comparação ensemble vs. modelos individuais
- Testes de significância estatística

**Outputs**:
- `outputs/figures/ensemble_comparison.png`
- `outputs/tables/ensemble_results.csv`

---

### 4. `04_fuzzy_features_analysis.ipynb` (TODO)
**Análise profunda das features fuzzy.**

Cobre:
- Feature importance analysis
- Correlação entre features e emoções específicas
- Ablation studies (remoção de features)
- Distribuições e estatísticas detalhadas

**Outputs**:
- `outputs/figures/feature_importance.png`
- `outputs/figures/feature_correlations.png`
- `outputs/tables/ablation_results.csv`

---

### 5. `05_visualizations_for_paper.ipynb` (TODO)
**Geração de todas as figuras para publicação.**

Cobre:
- Diagramas de arquitetura dos modelos
- Gráficos de comparação de performance
- Confusion matrices (todos os modelos)
- Visualizações de features
- Exemplos de predições com explicações

**Outputs**:
- Todas as figuras em `outputs/figures/` (300 DPI)
- Prontas para inclusão direta no paper

---

### 6. `06_error_analysis.ipynb` (TODO)
**Análise de erros e casos extremos.**

Cobre:
- Identificação de padrões de falha
- Análise de erros por classe
- Confusão entre emoções similares
- Investigação de casos difíceis

**Outputs**:
- `outputs/figures/error_patterns.png`
- `outputs/tables/failure_cases.csv`

---

### 7. `07_statistical_tests.ipynb` (TODO)
**Validação estatística dos resultados.**

Cobre:
- McNemar's test para comparação de modelos
- Bootstrap confidence intervals
- Análise de cross-validation
- Significância estatística das melhorias

**Outputs**:
- `outputs/tables/statistical_tests.csv`

---

## 📁 Organização dos Outputs

Todos os outputs dos notebooks são salvos em `outputs/`:

```
outputs/
├── figures/          # Figuras para publicação (PNG, 300 DPI)
│   ├── eda/         # Análise exploratória
│   ├── features/    # Features fuzzy
│   ├── models/      # Performance de modelos
│   └── ...
├── metrics/          # Métricas detalhadas (JSON)
│   ├── model_evaluation.json
│   └── ensemble_results.json
└── tables/           # Tabelas de resultados (CSV + LaTeX)
    ├── model_comparison.csv
    ├── model_comparison.tex
    ├── feature_statistics.csv
    └── ...
```

## 🚀 Como Usar

**Ordem de execução recomendada**:

1. **`00_data_exploration_features.ipynb`** → Entender o dataset e extrair features
2. **`02_agents_demo.ipynb`** → Ver o sistema funcionando end-to-end
3. **`01_model_evaluation.ipynb`** → Avaliar performance dos modelos
4. **`03_ensemble_analysis.ipynb`** → Analisar estratégias de ensemble
5. **`04_fuzzy_features_analysis.ipynb`** → Feature importance e ablations
6. **`05_visualizations_for_paper.ipynb`** → Gerar figuras finais
7. **`06_error_analysis.ipynb`** → Investigar falhas
8. **`07_statistical_tests.ipynb`** → Validação estatística

Cada notebook é auto-contido, mas alguns dependem de outputs anteriores.

## 🔬 Reprodutibilidade

- Todos os notebooks usam seeds fixas
- Caminhos relativos ao project root
- Outputs versionados em `outputs/`
- Documentação clara de todos os parâmetros

## 📄 Seções do Paper

Estes notebooks geram resultados para:
- **Methods**: Arquiteturas, features fuzzy, XAI
- **Results**: Métricas de performance, comparações
- **Discussion**: Análise de erros, ablations
- **Figures/Tables**: Todas as visualizações e resultados numéricos
