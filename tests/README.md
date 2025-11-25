# Tests

> Unit tests and integration tests for the Cerebrum Artis project.

[English](#english) | [Português](#português)

---

## English

### Overview

This directory contains all tests for validating the functionality of models, agents, and utilities. We use pytest as the testing framework.

### Running Tests

**Run all tests:**
```bash
pytest tests/
```

**Run with coverage:**
```bash
pytest tests/ --cov=cerebrum_artis --cov-report=html
```

**Run specific test file:**
```bash
pytest tests/test_agents.py -v
```

**Run specific test:**
```bash
pytest tests/test_models.py::test_v3_model_inference -v
```

### Test Structure

```
tests/
├── test_agents.py          # Agent system tests
├── test_models.py          # Model architecture tests
├── test_fuzzy.py           # Fuzzy logic tests
├── test_data.py            # Data processing tests
├── test_utils.py           # Utility function tests
└── test_integration.py     # End-to-end integration tests
```

### Test Categories

#### Unit Tests
Test individual components in isolation.

```python
# Example: Testing fuzzy feature extraction
def test_warmth_extraction():
    from cerebrum_artis.fuzzy import WarmthExtractor
    
    extractor = WarmthExtractor()
    # Red image should have high warmth
    red_image = np.zeros((100, 100, 3), dtype=np.uint8)
    red_image[:, :, 0] = 255  # Red channel
    
    warmth = extractor.extract(red_image)
    assert warmth > 0.8, "Red image should have high warmth"
```

#### Integration Tests
Test components working together.

```python
# Example: Testing PerceptoEmocional with all components
def test_percepto_full_pipeline():
    from cerebrum_artis.agents import PerceptoEmocional
    
    agent = PerceptoEmocional()
    result = agent.analyze_painting(
        image_path="test_images/sample.jpg",
        caption="A peaceful landscape"
    )
    
    assert "emotion" in result
    assert "confidence" in result
    assert "explanation" in result
```

#### Performance Tests
Test model accuracy and speed.

```python
# Example: Testing ensemble performance
def test_ensemble_accuracy():
    from cerebrum_artis.models import EnsembleModel
    
    ensemble = EnsembleModel(models=['v3', 'v4', 'v4_1'])
    accuracy = ensemble.evaluate(test_dataset)
    
    assert accuracy >= 0.70, f"Ensemble should achieve ≥70% (got {accuracy})"
```

### Writing New Tests

**1. Create test file:**
```python
# tests/test_new_feature.py
import pytest
from cerebrum_artis.new_module import NewFeature

def test_basic_functionality():
    feature = NewFeature()
    result = feature.process("input")
    assert result is not None

def test_edge_cases():
    feature = NewFeature()
    with pytest.raises(ValueError):
        feature.process(None)  # Should raise error
```

**2. Use fixtures for common setup:**
```python
@pytest.fixture
def sample_image():
    return np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)

@pytest.fixture
def sample_model():
    from cerebrum_artis.models import V3Model
    return V3Model(pretrained=True)

def test_with_fixtures(sample_image, sample_model):
    result = sample_model.predict(sample_image)
    assert result.shape == (9,)  # 9 emotion classes
```

**3. Mark tests appropriately:**
```python
@pytest.mark.slow
def test_full_training():
    # This test takes a long time
    pass

@pytest.mark.gpu
def test_gpu_inference():
    # This test requires GPU
    pass

# Run only fast tests:
# pytest tests/ -m "not slow"
```

### Test Data

Test data is located in `tests/fixtures/`:

```
tests/fixtures/
├── images/              # Test images
│   ├── sample_painting.jpg
│   └── test_set/
├── captions.json        # Test captions
└── expected_outputs.json # Expected results
```

### Continuous Integration

Tests run automatically on GitHub Actions:

```yaml
# .github/workflows/tests.yml
name: Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Install dependencies
        run: pip install -r requirements.txt
      - name: Run tests
        run: pytest tests/ --cov=cerebrum_artis
```

### Coverage Reports

After running tests with coverage:

```bash
pytest tests/ --cov=cerebrum_artis --cov-report=html
```

Open `htmlcov/index.html` in browser to view detailed coverage report.

**Target Coverage:**
- Overall: ≥80%
- Critical modules (models, agents): ≥90%
- Utilities: ≥70%

---

## Português

<details>
<summary>Clique para ver versão em português</summary>

### Visão Geral

Este diretório contém todos os testes para validar a funcionalidade de modelos, agentes e utilitários. Usamos pytest como framework de testes.

### Executando Testes

**Executar todos os testes:**
```bash
pytest tests/
```

**Executar com cobertura:**
```bash
pytest tests/ --cov=cerebrum_artis --cov-report=html
```

**Executar arquivo específico:**
```bash
pytest tests/test_agents.py -v
```

**Executar teste específico:**
```bash
pytest tests/test_models.py::test_v3_model_inference -v
```

### Estrutura de Testes

```
tests/
├── test_agents.py          # Testes do sistema de agentes
├── test_models.py          # Testes de arquitetura de modelos
├── test_fuzzy.py           # Testes de lógica fuzzy
├── test_data.py            # Testes de processamento de dados
├── test_utils.py           # Testes de funções utilitárias
└── test_integration.py     # Testes de integração end-to-end
```

### Categorias de Testes

#### Testes Unitários
Testam componentes individuais isoladamente.

```python
# Exemplo: Testando extração de features fuzzy
def test_warmth_extraction():
    from cerebrum_artis.fuzzy import WarmthExtractor
    
    extractor = WarmthExtractor()
    # Imagem vermelha deve ter alta warmth
    red_image = np.zeros((100, 100, 3), dtype=np.uint8)
    red_image[:, :, 0] = 255  # Canal vermelho
    
    warmth = extractor.extract(red_image)
    assert warmth > 0.8, "Imagem vermelha deve ter alta warmth"
```

#### Testes de Integração
Testam componentes trabalhando juntos.

```python
# Exemplo: Testando PerceptoEmocional com todos os componentes
def test_percepto_full_pipeline():
    from cerebrum_artis.agents import PerceptoEmocional
    
    agent = PerceptoEmocional()
    result = agent.analyze_painting(
        image_path="test_images/sample.jpg",
        caption="Uma paisagem tranquila"
    )
    
    assert "emotion" in result
    assert "confidence" in result
    assert "explanation" in result
```

#### Testes de Performance
Testam precisão e velocidade dos modelos.

```python
# Exemplo: Testando precisão do ensemble
def test_ensemble_accuracy():
    from cerebrum_artis.models import EnsembleModel
    
    ensemble = EnsembleModel(models=['v3', 'v4', 'v4_1'])
    accuracy = ensemble.evaluate(test_dataset)
    
    assert accuracy >= 0.70, f"Ensemble deve atingir ≥70% (obteve {accuracy})"
```

### Escrevendo Novos Testes

**1. Criar arquivo de teste:**
```python
# tests/test_new_feature.py
import pytest
from cerebrum_artis.new_module import NewFeature

def test_basic_functionality():
    feature = NewFeature()
    result = feature.process("input")
    assert result is not None

def test_edge_cases():
    feature = NewFeature()
    with pytest.raises(ValueError):
        feature.process(None)  # Deve levantar erro
```

**2. Usar fixtures para setup comum:**
```python
@pytest.fixture
def sample_image():
    return np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)

@pytest.fixture
def sample_model():
    from cerebrum_artis.models import V3Model
    return V3Model(pretrained=True)

def test_with_fixtures(sample_image, sample_model):
    result = sample_model.predict(sample_image)
    assert result.shape == (9,)  # 9 classes de emoção
```

**3. Marcar testes apropriadamente:**
```python
@pytest.mark.slow
def test_full_training():
    # Este teste demora muito
    pass

@pytest.mark.gpu
def test_gpu_inference():
    # Este teste requer GPU
    pass

# Executar apenas testes rápidos:
# pytest tests/ -m "not slow"
```

### Dados de Teste

Dados de teste estão em `tests/fixtures/`:

```
tests/fixtures/
├── images/              # Imagens de teste
│   ├── sample_painting.jpg
│   └── test_set/
├── captions.json        # Captions de teste
└── expected_outputs.json # Resultados esperados
```

### Integração Contínua

Testes executam automaticamente no GitHub Actions:

```yaml
# .github/workflows/tests.yml
name: Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Install dependencies
        run: pip install -r requirements.txt
      - name: Run tests
        run: pytest tests/ --cov=cerebrum_artis
```

### Relatórios de Cobertura

Após executar testes com cobertura:

```bash
pytest tests/ --cov=cerebrum_artis --cov-report=html
```

Abrir `htmlcov/index.html` no navegador para ver relatório detalhado de cobertura.

**Meta de Cobertura:**
- Geral: ≥80%
- Módulos críticos (models, agents): ≥90%
- Utilitários: ≥70%

</details>

---

**Directory**: `/home/paloma/cerebrum-artis/tests/`  
**Last Updated**: November 25, 2025
