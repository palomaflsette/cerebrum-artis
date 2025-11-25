# Cerebrum Artis

> A multi-agent system for emotional analysis of paintings combining deep learning, fuzzy logic, and image captioning.

[English](#english) | [Português](#português)

---

## English

### Overview

Cerebrum Artis is a research project that analyzes the emotional content of artwork through a hybrid approach combining:

- **Deep Learning**: Multimodal classifiers (image + text)
- **Fuzzy Logic**: Interpretable visual features based on color psychology
- **Image Captioning**: Automatic generation of emotional descriptions using SAT (Show, Attend & Tell)
- **Ensemble Methods**: Optimized combination of multiple models

The system classifies paintings into 9 emotion categories:
```
amusement, awe, contentment, excitement, anger, disgust, fear, sadness, something else
```

### Key Results

- **Best Individual Model**: V3 Fuzzy Features - 70.63% validation accuracy
- **Ensemble Model**: 71.47% validation accuracy (+0.84% improvement)
- **Dataset**: ArtEmis (80,000+ paintings with emotional annotations)

### Project Structure

```
cerebrum-artis/
├── cerebrum_artis/          # Main Python package
│   ├── agents/              # Multi-agent system (PerceptoEmocional, Colorista, Explicador)
│   ├── models/              # Deep learning model versions (V1-V4.1 + Ensemble)
│   ├── fuzzy/               # Fuzzy logic system
│   ├── data/                # Data processing utilities
│   └── utils/               # General utilities
│
├── scripts/                 # Executable scripts
│   ├── training/            # Model training scripts
│   ├── evaluation/          # Evaluation and testing
│   └── demo/                # Demo applications
│
├── notebooks/               # Jupyter notebooks for analysis
├── configs/                 # Configuration files
├── tests/                   # Unit tests
├── docs/                    # Documentation (technical reports, architecture)
├── data/                    # Data directory (see .gitignore)
├── checkpoints/             # Model checkpoints (see .gitignore)
└── outputs/                 # Analysis outputs and visualizations
```

### Quick Start

#### Installation

```bash
# Clone the repository
git clone https://github.com/palomaflsette/cerebrum-artis.git
cd cerebrum-artis

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install package in development mode
pip install -e .
```

#### Basic Usage

```python
from cerebrum_artis.agents import PerceptoEmocional
from cerebrum_artis.models.ensemble import EnsembleClassifier

# Initialize the emotion perception agent
percepto = PerceptoEmocional()

# Analyze an artwork
result = percepto.analyze_painting(
    image_path="path/to/painting.jpg",
    caption="A vibrant sunset over mountains"
)

print(f"Detected emotion: {result['emotion']}")
print(f"Confidence: {result['confidence']:.2%}")
```

### Model Versions

| Version | Architecture | Val Accuracy | Description |
|---------|-------------|--------------|-------------|
| V1 | Baseline | ~65% | Simple ResNet50 + RoBERTa |
| V2 | Improved | ~68% | Enhanced features |
| V3 | Fuzzy Features | **70.63%** | Added 7 fuzzy visual features |
| V4 | Fuzzy Gating | 70.37% | Adaptive fusion mechanism |
| V4.1 | Integrated Gating | 70.40% | Integrated gating in forward pass |
| **Ensemble** | V3+V4+V4.1 | **71.47%** | Optimized weighted combination |

### Training Models

```bash
# Train V3 (Fuzzy Features)
python scripts/training/train_v3.py --config configs/training/v3_config.json

# Train V4 (Fuzzy Gating)
python scripts/training/train_v4.py --config configs/training/v4_config.json

# Evaluate ensemble
python scripts/evaluation/ensemble_test.py
```

### Running Tests

```bash
# Run all tests
pytest tests/

# Run specific test suite
pytest tests/test_agents.py
pytest tests/test_models.py
```

### Documentation

- [Technical Report](docs/RELATORIO.md) (PT) - Detailed project documentation
- [Architecture](docs/ARCHITECTURE.md) (PT) - System architecture and design decisions
- [Model Documentation](cerebrum_artis/models/README.md) - Model implementations
- [Agent Documentation](cerebrum_artis/agents/README.md) - Multi-agent system details

### Research Paper

This project is being prepared for academic publication. Key contributions:

1. **Hybrid Approach**: Novel combination of fuzzy logic features with deep learning
2. **Ensemble Strategy**: Demonstrating that model diversity improves emotion classification
3. **Interpretability**: Fuzzy features provide human-understandable explanations
4. **Benchmark Results**: State-of-the-art results on ArtEmis dataset

### Requirements

- Python 3.8+
- PyTorch 1.12+
- transformers (HuggingFace)
- OpenCV
- scikit-fuzzy
- pandas, numpy

See [requirements.txt](requirements.txt) for complete list.

### License

This project is licensed under the MIT License - see the LICENSE file for details.

### Citation

If you use this work in your research, please cite:

```bibtex
@misc{cerebrum-artis-2025,
  author = {Paloma Sette},
  title = {Cerebrum Artis: Multi-Agent System for Emotional Analysis of Paintings},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/palomaflsette/cerebrum-artis}
}
```

### Contributing

Contributions are welcome! Please read the contributing guidelines before submitting pull requests.

### Contact

- **Author**: Paloma Sette
- **Email**: palomaflsette@gmail.com
- **GitHub**: [@palomaflsette](https://github.com/palomaflsette)

---

## Português

<details>
<summary>Clique para ver a versão em português</summary>

### Visão Geral

Cerebrum Artis é um projeto de pesquisa que analisa o conteúdo emocional de obras de arte através de uma abordagem híbrida combinando:

- **Deep Learning**: Classificadores multimodais (imagem + texto)
- **Lógica Fuzzy**: Features visuais interpretáveis baseadas em psicologia das cores
- **Image Captioning**: Geração automática de descrições emocionais usando SAT (Show, Attend & Tell)
- **Métodos Ensemble**: Combinação otimizada de múltiplos modelos

O sistema classifica pinturas em 9 categorias de emoção:
```
amusement, awe, contentment, excitement, anger, disgust, fear, sadness, something else
```

### Resultados Principais

- **Melhor Modelo Individual**: V3 Fuzzy Features - 70.63% de acurácia na validação
- **Modelo Ensemble**: 71.47% de acurácia na validação (+0.84% de melhoria)
- **Dataset**: ArtEmis (80.000+ pinturas com anotações emocionais)

### Início Rápido

#### Instalação

```bash
# Clonar o repositório
git clone https://github.com/palomaflsette/cerebrum-artis.git
cd cerebrum-artis

# Criar ambiente virtual
python -m venv venv
source venv/bin/activate  # No Windows: venv\Scripts\activate

# Instalar dependências
pip install -r requirements.txt

# Instalar pacote em modo desenvolvimento
pip install -e .
```

#### Uso Básico

```python
from cerebrum_artis.agents import PerceptoEmocional
from cerebrum_artis.models.ensemble import EnsembleClassifier

# Inicializar o agente de percepção emocional
percepto = PerceptoEmocional()

# Analisar uma obra de arte
resultado = percepto.analyze_painting(
    image_path="caminho/para/pintura.jpg",
    caption="Um pôr do sol vibrante sobre montanhas"
)

print(f"Emoção detectada: {resultado['emotion']}")
print(f"Confiança: {resultado['confidence']:.2%}")
```

### Versões dos Modelos

| Versão | Arquitetura | Acurácia Val | Descrição |
|---------|-------------|--------------|-----------|
| V1 | Baseline | ~65% | ResNet50 + RoBERTa simples |
| V2 | Improved | ~68% | Features melhoradas |
| V3 | Fuzzy Features | **70.63%** | 7 features fuzzy visuais adicionadas |
| V4 | Fuzzy Gating | 70.37% | Mecanismo de fusão adaptativa |
| V4.1 | Integrated Gating | 70.40% | Gating integrado no forward pass |
| **Ensemble** | V3+V4+V4.1 | **71.47%** | Combinação ponderada otimizada |

### Treinando Modelos

```bash
# Treinar V3 (Fuzzy Features)
python scripts/training/train_v3.py --config configs/training/v3_config.json

# Treinar V4 (Fuzzy Gating)
python scripts/training/train_v4.py --config configs/training/v4_config.json

# Avaliar ensemble
python scripts/evaluation/ensemble_test.py
```

### Executando Testes

```bash
# Executar todos os testes
pytest tests/

# Executar suíte de teste específica
pytest tests/test_agents.py
pytest tests/test_models.py
```

### Documentação

- [Relatório Técnico](docs/RELATORIO.md) - Documentação detalhada do projeto
- [Arquitetura](docs/ARCHITECTURE.md) - Arquitetura do sistema e decisões de design
- [Documentação dos Modelos](cerebrum_artis/models/README.md) - Implementações dos modelos
- [Documentação dos Agentes](cerebrum_artis/agents/README.md) - Detalhes do sistema multi-agente

### Artigo de Pesquisa

Este projeto está sendo preparado para publicação acadêmica. Principais contribuições:

1. **Abordagem Híbrida**: Combinação novel de features de lógica fuzzy com deep learning
2. **Estratégia Ensemble**: Demonstração de que diversidade de modelos melhora classificação emocional
3. **Interpretabilidade**: Features fuzzy fornecem explicações humanamente compreensíveis
4. **Resultados Benchmark**: Resultados state-of-the-art no dataset ArtEmis

### Requisitos

- Python 3.8+
- PyTorch 1.12+
- transformers (HuggingFace)
- OpenCV
- scikit-fuzzy
- pandas, numpy

Veja [requirements.txt](requirements.txt) para lista completa.

### Licença

Este projeto está licenciado sob a Licença MIT - veja o arquivo LICENSE para detalhes.

### Citação

Se você usar este trabalho em sua pesquisa, por favor cite:

```bibtex
@misc{cerebrum-artis-2025,
  author = {Paloma Sette},
  title = {Cerebrum Artis: Sistema Multi-Agente para Análise Emocional de Pinturas},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/palomaflsette/cerebrum-artis}
}
```

### Contribuindo

Contribuições são bem-vindas! Por favor, leia as diretrizes de contribuição antes de submeter pull requests.

### Contato

- **Autora**: Paloma Sette
- **Email**: palomaflsette@gmail.com
- **GitHub**: [@palomaflsette](https://github.com/palomaflsette)

</details>

---

**Last Updated**: November 25, 2025  
**Version**: 1.0.0  
**Status**: Active Development
