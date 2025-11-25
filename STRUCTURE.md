# Cerebrum Artis - Project Structure

## Working Directory Structure

```
cerebrum-artis/
├── cerebrum_artis/          # Main Python package
│   ├── agents/              # Multi-agent system (PerceptoEmocional, Colorista, Explicador)
│   ├── models/              # Deep learning models (V1-V4.1)
│   │   ├── v1_baseline/
│   │   ├── v2_improved/
│   │   ├── v3_fuzzy_features/
│   │   ├── v4_fuzzy_gating/
│   │   ├── v4_1_integrated/
│   │   └── ensemble/
│   ├── fuzzy/               # Fuzzy logic system
│   ├── data/                # Data loading and preprocessing
│   └── utils/               # Utility functions
│
├── scripts/                 # Executable scripts
│   ├── training/            # train_v3.py, train_v4.py, train_v4_1.py
│   ├── evaluation/          # ensemble_test.py, eval_single_model.py
│   └── demo/                # demo_percepto.py
│
├── notebooks/               # Jupyter notebooks for analysis
│   └── 01_model_evaluation.ipynb
│
├── configs/                 # Configuration files
│   ├── training/            # Training configurations
│   └── inference/           # Inference configurations
│
├── tests/                   # Unit tests
├── docs/                    # Documentation
│   ├── RELATORIO.md         # Complete technical report (PT)
│   ├── ARCHITECTURE.md      # Architecture documentation (PT)
│   └── api/                 # API documentation
│
├── data/                    # Dataset directory (ignored by git)
│   └── artemis/             # ArtEmis dataset
│
├── checkpoints/             # Model weights (ignored by git)
│   ├── v1_baseline/
│   ├── v2_improved/
│   ├── v3_fuzzy_features/
│   ├── v4_fuzzy_gating/
│   ├── v4_1_integrated/
│   └── ensemble/
│
└── outputs/                 # Analysis outputs (ignored by git)
    ├── figures/             # Publication-ready figures
    ├── metrics/             # Evaluation metrics (JSON)
    └── tables/              # LaTeX tables for paper
```

## Key Files

- `README.md` - Project overview and quick start
- `requirements.txt` - Python dependencies
- `.gitignore` - Git ignore rules
- `STRUCTURE.md` - This file

## What's Ignored by Git

The following directories are in `.gitignore` and not versioned:

- `checkpoints/` - Model weights (large files)
- `data/` - Dataset files
- `outputs/` - Generated figures and metrics
- `logs/` - Training logs
- `garbage/` - Deprecated code (local only)
- `__pycache__/` - Python cache files

## Usage

### Training
```bash
python scripts/training/train_v3.py --data_path data/artemis/
```

### Evaluation
```bash
python scripts/evaluation/ensemble_test.py
```

### Analysis
```bash
jupyter notebook notebooks/01_model_evaluation.ipynb
```

### Demo
```bash
python scripts/demo/demo_percepto.py --image_path path/to/image.jpg
```

## Documentation

- Main README: `README.md`
- Agents: `cerebrum_artis/agents/README.md`
- Models: `cerebrum_artis/models/README.md`
- Fuzzy Logic: `cerebrum_artis/fuzzy/README.md`
- Scripts: `scripts/README.md`
- Notebooks: `notebooks/README.md`

---

Last Updated: November 25, 2025
