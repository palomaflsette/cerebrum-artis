# Analysis Notebooks

> Jupyter notebooks for extracting metrics, generating visualizations, and analyzing results for the research paper.

## Notebooks

### 01_model_evaluation.ipynb
Comprehensive evaluation of all model versions (V1-V4.1) on the test set.

**Outputs:**
- Accuracy metrics for each model
- Confusion matrices
- Per-class performance
- Statistical significance tests

### 02_ensemble_analysis.ipynb
Analysis of ensemble methods and optimization.

**Outputs:**
- Comparison of 5 ensemble strategies
- Grid search results visualization
- Optimal weight determination
- Performance gains analysis

### 03_fuzzy_features_analysis.ipynb
Analysis of fuzzy logic features contribution.

**Outputs:**
- Feature importance visualization
- Ablation study results
- Correlation analysis between fuzzy features and emotions
- Color distribution patterns

### 04_visualizations_for_paper.ipynb
Generate all visualizations for the research paper.

**Outputs:**
- Training curves (all models)
- Model comparison bar charts
- Confusion matrices (normalized and raw)
- Ensemble strategy comparison
- Feature importance plots
- Example predictions with explanations

### 05_error_analysis.ipynb
Deep dive into model errors and failure cases.

**Outputs:**
- Most confused emotion pairs
- Failure case examples
- Error distribution by art style
- Difficult examples analysis

### 06_statistical_tests.ipynb
Statistical significance testing between models.

**Outputs:**
- McNemar's test results
- Confidence intervals
- Pairwise model comparisons
- Bootstrap significance tests

## Usage

```bash
# Launch Jupyter
jupyter notebook

# Or Jupyter Lab
jupyter lab
```

## Output Directories

All notebook outputs are saved to organized directories:

```
outputs/
├── figures/          # Publication-ready figures
│   ├── training_curves/
│   ├── confusion_matrices/
│   ├── comparisons/
│   └── feature_analysis/
├── tables/           # LaTeX tables for paper
├── metrics/          # JSON files with detailed metrics
└── examples/         # Example predictions and visualizations
```

## Requirements

All notebooks use the main project environment. Additional visualization libraries:

```bash
pip install jupyter matplotlib seaborn plotly
```

## Reproducibility

All notebooks are designed to be reproducible:
- Random seeds are set
- Paths are relative to project root
- Model checkpoints are loaded from `checkpoints/`
- Data is loaded from `data/artemis/`

## Paper Sections

Each notebook corresponds to specific paper sections:

| Notebook | Paper Section |
|----------|---------------|
| 01_model_evaluation.ipynb | Results - Individual Models |
| 02_ensemble_analysis.ipynb | Results - Ensemble Methods |
| 03_fuzzy_features_analysis.ipynb | Methodology - Fuzzy Features |
| 04_visualizations_for_paper.ipynb | Figures (all sections) |
| 05_error_analysis.ipynb | Discussion - Error Analysis |
| 06_statistical_tests.ipynb | Results - Statistical Significance |

---

**Last Updated**: November 25, 2025
