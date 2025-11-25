# Agents Module

> Multi-agent system for emotional analysis of artwork.

[English](#english) | [Português](#português)

---

## English

### Overview

The agents module implements a multi-agent system where specialized agents collaborate to analyze the emotional content of paintings. Each agent has a specific role in the perception and interpretation process.

### Available Agents

#### PerceptoEmocional (Emotional Perceiver)

The main agent responsible for perceiving and classifying emotions in artwork.

**Capabilities:**
- Emotion classification using deep learning models
- Integration with fuzzy logic features
- Confidence scoring
- Multi-modal analysis (image + text)

**Usage:**
```python
from cerebrum_artis.agents import PerceptoEmocional

# Initialize agent
percepto = PerceptoEmocional()

# Analyze a painting
result = percepto.analyze_painting(
    image_path="path/to/painting.jpg",
    caption="A serene landscape at dawn"
)

# Access results
emotion = result['emotion']
confidence = result['confidence']
features = result['features']
```

**Available Methods:**
- `analyze_painting(image_path, caption)` - Main analysis method
- `get_emotion_probabilities(image, text)` - Get probability distribution
- `explain_prediction(result)` - Generate explanation for prediction

#### Colorista (Color Analyst)

Specialized agent for color analysis and fuzzy feature extraction.

**Capabilities:**
- Color palette extraction
- Dominant color identification
- Color harmony analysis
- Fuzzy membership computation

**Usage:**
```python
from cerebrum_artis.agents import Colorista

colorista = Colorista()

# Extract color features
color_features = colorista.analyze_colors(image_path)

# Get dominant colors
dominant_colors = colorista.get_dominant_colors(image, n_colors=5)

# Analyze color harmony
harmony = colorista.analyze_harmony(color_palette)
```

#### Explicador (Explainer)

Agent responsible for generating human-understandable explanations.

**Capabilities:**
- Natural language explanations
- Feature importance visualization
- Reasoning chain generation
- Multi-lingual support (EN/PT)

**Usage:**
```python
from cerebrum_artis.agents import Explicador

explicador = Explicador()

# Generate explanation
explanation = explicador.explain(
    emotion="awe",
    features=feature_vector,
    confidence=0.85
)

# Get reasoning chain
reasoning = explicador.get_reasoning_chain(result)
```

### Agent Communication

Agents can communicate and collaborate through a shared message bus:

```python
from cerebrum_artis.agents import PerceptoEmocional, Colorista, Explicador

# Initialize agents
percepto = PerceptoEmocional()
colorista = Colorista()
explicador = Explicador()

# Collaborative analysis
image_path = "painting.jpg"
caption = "A vibrant sunset"

# 1. Color analysis
colors = colorista.analyze_colors(image_path)

# 2. Emotion perception
emotion_result = percepto.analyze_painting(
    image_path=image_path,
    caption=caption,
    color_features=colors  # Pass color features
)

# 3. Generate explanation
explanation = explicador.explain(emotion_result)
```

### Architecture

```
┌─────────────────────────────────────────┐
│         User Application                │
└──────────────┬──────────────────────────┘
               │
       ┌───────┴───────┐
       │  Message Bus   │
       └───────┬───────┘
               │
    ┌──────────┼──────────┐
    │          │          │
┌───▼───┐  ┌──▼────┐  ┌──▼────────┐
│Percepto│  │Colorista│  │Explicador│
└───┬───┘  └──┬────┘  └──┬────────┘
    │         │          │
    └─────────┴──────────┘
          Deep Models
```

### Configuration

Agents can be configured via JSON or Python dictionaries:

```python
config = {
    "percepto": {
        "model_version": "ensemble",
        "confidence_threshold": 0.5,
        "use_fuzzy_features": True
    },
    "colorista": {
        "color_space": "LAB",
        "n_clusters": 5,
        "fuzzy_sets": ["warm", "cold", "saturated", "muted"]
    },
    "explicador": {
        "language": "en",
        "verbosity": "detailed"
    }
}

percepto = PerceptoEmocional(config["percepto"])
```

### Extending Agents

To create a custom agent, inherit from the base `Agent` class:

```python
from cerebrum_artis.agents.base import Agent

class MyCustomAgent(Agent):
    def __init__(self, config=None):
        super().__init__(name="MyAgent", config=config)
    
    def process(self, input_data):
        # Implement your logic
        result = self.custom_processing(input_data)
        return result
    
    def custom_processing(self, data):
        # Your custom implementation
        pass
```

### API Reference

See the [API documentation](../../docs/api/agents.md) for detailed method signatures and parameters.

---

## Português

<details>
<summary>Clique para ver a versão em português</summary>

### Visão Geral

O módulo de agentes implementa um sistema multi-agente onde agentes especializados colaboram para analisar o conteúdo emocional de pinturas. Cada agente tem um papel específico no processo de percepção e interpretação.

### Agentes Disponíveis

#### PerceptoEmocional (Percebedor Emocional)

O agente principal responsável por perceber e classificar emoções em obras de arte.

**Capacidades:**
- Classificação de emoções usando modelos de deep learning
- Integração com features de lógica fuzzy
- Pontuação de confiança
- Análise multi-modal (imagem + texto)

**Uso:**
```python
from cerebrum_artis.agents import PerceptoEmocional

# Inicializar agente
percepto = PerceptoEmocional()

# Analisar uma pintura
resultado = percepto.analyze_painting(
    image_path="caminho/para/pintura.jpg",
    caption="Uma paisagem serena ao amanhecer"
)

# Acessar resultados
emocao = resultado['emotion']
confianca = resultado['confidence']
features = resultado['features']
```

#### Colorista (Analista de Cores)

Agente especializado em análise de cores e extração de features fuzzy.

**Capacidades:**
- Extração de paleta de cores
- Identificação de cores dominantes
- Análise de harmonia de cores
- Computação de pertinência fuzzy

#### Explicador

Agente responsável por gerar explicações humanamente compreensíveis.

**Capacidades:**
- Explicações em linguagem natural
- Visualização de importância de features
- Geração de cadeia de raciocínio
- Suporte multi-língua (EN/PT)

### Comunicação entre Agentes

Agentes podem comunicar e colaborar através de um barramento de mensagens compartilhado.

### Configuração

Agentes podem ser configurados via JSON ou dicionários Python.

### Estendendo Agentes

Para criar um agente customizado, herde da classe base `Agent`.

</details>

---

**Module**: `cerebrum_artis.agents`  
**Version**: 1.0.0  
**Last Updated**: November 25, 2025
