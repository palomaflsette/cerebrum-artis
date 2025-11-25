# Documentation

> Comprehensive project documentation including reports, architecture, and API references.

[English](#english) | [Português](#português)

---

## English

### Documentation Index

This directory contains all project documentation organized by type.

### Main Documents

#### [RELATORIO.md](RELATORIO.md)
**Complete Technical Report** (Portuguese)

Detailed report documenting the entire project development process, including:
- Initial baseline models (V1, V2)
- Fuzzy logic integration (V3)
- Advanced gating mechanisms (V4, V4.1)
- Ensemble methods and optimization
- Training logs and performance analysis
- Failure analysis and lessons learned

**Key Results:**
- V3 (Fuzzy Features): 70.63% accuracy
- Ensemble (Optimized): 71.47% accuracy
- 441 ensemble combinations tested
- Optimal weights: V3=55%, V4=30%, V4.1=15%

#### [ARCHITECTURE.md](ARCHITECTURE.md)
**System Architecture Documentation** (Portuguese)

Comprehensive architecture overview covering:
- Multi-agent system design
- Model architectures (V1-V4.1)
- Fuzzy logic system
- Data flow and processing pipeline
- Component interactions
- Extension points

### API Documentation

Located in `docs/api/`:

```
docs/api/
├── models.md           # Model API reference
├── agents.md           # Agent system API
├── fuzzy.md            # Fuzzy logic API
└── utils.md            # Utility functions API
```

### Quick Links

| Document | Description | Language |
|----------|-------------|----------|
| [RELATORIO.md](RELATORIO.md) | Complete technical report | PT-BR |
| [ARCHITECTURE.md](ARCHITECTURE.md) | System architecture | PT-BR |
| [../README.md](../README.md) | Project overview | EN/PT-BR |
| [../cerebrum_artis/models/README.md](../cerebrum_artis/models/README.md) | Model documentation | EN/PT-BR |
| [../cerebrum_artis/agents/README.md](../cerebrum_artis/agents/README.md) | Agent documentation | EN/PT-BR |
| [../cerebrum_artis/fuzzy/README.md](../cerebrum_artis/fuzzy/README.md) | Fuzzy logic docs | EN/PT-BR |

### Research Documentation

#### Published Results

**Emotion Classification:**
- 9 emotion categories (amusement, awe, contentment, excitement, anger, disgust, fear, sadness, something else)
- ArtEmis dataset: 80K+ artwork-emotion pairs
- State-of-the-art results: 71.47% accuracy

**Novel Contributions:**
1. Fuzzy logic feature extraction for artwork analysis
2. Multi-agent collaborative system
3. Ensemble optimization through grid search
4. Interpretable emotion explanations

#### Citing This Work

```bibtex
@software{cerebrum_artis_2025,
  title = {Cerebrum Artis: Hybrid Deep Learning and Fuzzy Logic for Artwork Emotion Classification},
  author = {[Your Name]},
  year = {2025},
  url = {https://github.com/yourusername/cerebrum-artis}
}
```

### Development Documentation

#### Training Guides

- [Training V3 Model](../scripts/README.md#train_v3py)
- [Training V4 Models](../scripts/README.md#train_v4py)
- [Ensemble Evaluation](../scripts/README.md#ensemble_testpy)

#### Testing Guides

- [Running Tests](../tests/README.md#running-tests)
- [Writing Tests](../tests/README.md#writing-new-tests)
- [Coverage Reports](../tests/README.md#coverage-reports)

#### Configuration Guides

- [Training Configs](../configs/README.md#training-configurations)
- [Inference Configs](../configs/README.md#inference-configurations)
- [Custom Configs](../configs/README.md#creating-custom-configs)

### Images and Diagrams

Located in `docs/imgs/`:

```
docs/imgs/
├── architecture_overview.png
├── fuzzy_membership_functions.png
├── ensemble_performance.png
├── training_curves/
│   ├── v3_accuracy.png
│   ├── v4_accuracy.png
│   └── ensemble_comparison.png
└── agent_system/
    ├── percepto_pipeline.png
    ├── colorista_analysis.png
    └── explicador_reasoning.png
```

### Contributing to Documentation

#### Adding New Documentation

1. **Create markdown file** in appropriate directory
2. **Follow existing structure** (bilingual EN/PT-BR)
3. **Include code examples** where relevant
4. **Add to this index** in appropriate section
5. **Cross-reference** related documents

#### Documentation Style Guide

- Use clear, concise language
- Provide working code examples
- Include expected outputs
- Document edge cases
- Link to related sections
- Keep bilingual versions synchronized

#### Building Documentation

Generate HTML documentation:

```bash
# Install dependencies
pip install sphinx sphinx-rtd-theme

# Build docs
cd docs
make html

# View in browser
open _build/html/index.html
```

### Support and Questions

For questions about the documentation:

1. Check [README.md](../README.md) for overview
2. Search existing [GitHub Issues](https://github.com/yourusername/cerebrum-artis/issues)
3. Open new issue with `[docs]` tag
4. Contact: your.email@example.com

---

## Português

<details>
<summary>Clique para ver versão em português</summary>

### Índice de Documentação

Este diretório contém toda a documentação do projeto organizada por tipo.

### Documentos Principais

#### [RELATORIO.md](RELATORIO.md)
**Relatório Técnico Completo**

Relatório detalhado documentando todo o processo de desenvolvimento do projeto, incluindo:
- Modelos baseline iniciais (V1, V2)
- Integração de lógica fuzzy (V3)
- Mecanismos avançados de gating (V4, V4.1)
- Métodos de ensemble e otimização
- Logs de treinamento e análise de performance
- Análise de falhas e lições aprendidas

**Resultados Chave:**
- V3 (Fuzzy Features): 70.63% acurácia
- Ensemble (Otimizado): 71.47% acurácia
- 441 combinações de ensemble testadas
- Pesos ótimos: V3=55%, V4=30%, V4.1=15%

#### [ARCHITECTURE.md](ARCHITECTURE.md)
**Documentação de Arquitetura do Sistema**

Visão geral abrangente da arquitetura cobrindo:
- Design do sistema multi-agente
- Arquiteturas de modelos (V1-V4.1)
- Sistema de lógica fuzzy
- Fluxo de dados e pipeline de processamento
- Interações entre componentes
- Pontos de extensão

### Documentação da API

Localizada em `docs/api/`:

```
docs/api/
├── models.md           # Referência da API de modelos
├── agents.md           # API do sistema de agentes
├── fuzzy.md            # API da lógica fuzzy
└── utils.md            # API de funções utilitárias
```

### Links Rápidos

| Documento | Descrição | Idioma |
|----------|-----------|---------|
| [RELATORIO.md](RELATORIO.md) | Relatório técnico completo | PT-BR |
| [ARCHITECTURE.md](ARCHITECTURE.md) | Arquitetura do sistema | PT-BR |
| [../README.md](../README.md) | Visão geral do projeto | EN/PT-BR |
| [../cerebrum_artis/models/README.md](../cerebrum_artis/models/README.md) | Documentação de modelos | EN/PT-BR |
| [../cerebrum_artis/agents/README.md](../cerebrum_artis/agents/README.md) | Documentação de agentes | EN/PT-BR |
| [../cerebrum_artis/fuzzy/README.md](../cerebrum_artis/fuzzy/README.md) | Docs de lógica fuzzy | EN/PT-BR |

### Documentação de Pesquisa

#### Resultados Publicados

**Classificação de Emoções:**
- 9 categorias de emoção (amusement, awe, contentment, excitement, anger, disgust, fear, sadness, something else)
- Dataset ArtEmis: 80K+ pares obra-emoção
- Resultados estado-da-arte: 71.47% acurácia

**Contribuições Inovadoras:**
1. Extração de features usando lógica fuzzy para análise de arte
2. Sistema colaborativo multi-agente
3. Otimização de ensemble através de grid search
4. Explicações interpretáveis de emoções

#### Citando Este Trabalho

```bibtex
@software{cerebrum_artis_2025,
  title = {Cerebrum Artis: Hybrid Deep Learning and Fuzzy Logic for Artwork Emotion Classification},
  author = {[Seu Nome]},
  year = {2025},
  url = {https://github.com/seuusuario/cerebrum-artis}
}
```

### Documentação de Desenvolvimento

#### Guias de Treinamento

- [Treinando Modelo V3](../scripts/README.md#train_v3py)
- [Treinando Modelos V4](../scripts/README.md#train_v4py)
- [Avaliação de Ensemble](../scripts/README.md#ensemble_testpy)

#### Guias de Teste

- [Executando Testes](../tests/README.md#running-tests)
- [Escrevendo Testes](../tests/README.md#writing-new-tests)
- [Relatórios de Cobertura](../tests/README.md#coverage-reports)

#### Guias de Configuração

- [Configs de Treinamento](../configs/README.md#training-configurations)
- [Configs de Inferência](../configs/README.md#inference-configurations)
- [Configs Customizadas](../configs/README.md#creating-custom-configs)

### Imagens e Diagramas

Localizadas em `docs/imgs/`:

```
docs/imgs/
├── architecture_overview.png
├── fuzzy_membership_functions.png
├── ensemble_performance.png
├── training_curves/
│   ├── v3_accuracy.png
│   ├── v4_accuracy.png
│   └── ensemble_comparison.png
└── agent_system/
    ├── percepto_pipeline.png
    ├── colorista_analysis.png
    └── explicador_reasoning.png
```

### Contribuindo com a Documentação

#### Adicionando Nova Documentação

1. **Criar arquivo markdown** no diretório apropriado
2. **Seguir estrutura existente** (bilíngue EN/PT-BR)
3. **Incluir exemplos de código** quando relevante
4. **Adicionar a este índice** na seção apropriada
5. **Cross-referenciar** documentos relacionados

#### Guia de Estilo de Documentação

- Usar linguagem clara e concisa
- Fornecer exemplos de código funcionais
- Incluir outputs esperados
- Documentar casos extremos
- Linkar para seções relacionadas
- Manter versões bilíngues sincronizadas

#### Construindo Documentação

Gerar documentação HTML:

```bash
# Instalar dependências
pip install sphinx sphinx-rtd-theme

# Construir docs
cd docs
make html

# Ver no navegador
open _build/html/index.html
```

### Suporte e Questões

Para questões sobre a documentação:

1. Verificar [README.md](../README.md) para visão geral
2. Buscar [GitHub Issues](https://github.com/seuusuario/cerebrum-artis/issues) existentes
3. Abrir nova issue com tag `[docs]`
4. Contato: seu.email@exemplo.com

</details>

---

**Directory**: `/home/paloma/cerebrum-artis/docs/`  
**Last Updated**: November 25, 2025
