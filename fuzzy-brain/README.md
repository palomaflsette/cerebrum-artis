# 🧠 Fuzzy-Brain: Sistema Híbrido Neural-Fuzzy para Explicabilidade em Arte

> **Projeto de Pesquisa - Disciplina de Lógica Fuzzy**  
> Integrando Deep Learning com Lógica Fuzzy para gerar explicações interpretáveis sobre emoções evocadas por obras de arte.

---

## 🎯 **Objetivo**

Criar um sistema que não apenas **prediz** qual emoção uma obra de arte evoca, mas **explica o porquê** de forma interpretável, usando:

- 🧠 **CNN (ResNet)**: Extrai features visuais semânticas de alto nível
- 🔀 **Lógica Fuzzy**: Aplica raciocínio interpretável baseado em regras
- 🔗 **Sistema Híbrido**: Combina precisão neural com explicabilidade fuzzy

---

## 📊 **Arquitetura do Sistema**

```
                    IMAGEM (Pintura)
                          ↓
            ┌─────────────┴─────────────┐
            ↓                           ↓
    ┌──────────────┐          ┌──────────────┐
    │ Extração de  │          │  Extração de │
    │   Features   │          │   Features   │
    │   Visuais    │          │  Semânticas  │
    │              │          │    (CNN)     │
    └──────┬───────┘          └──────┬───────┘
           │                         │
           │  - Brilho              │  - Presença de faces
           │  - Saturação           │  - Complexidade
           │  - Temperatura         │  - Densidade
           │    de cor              │
           │                         │
           └──────────┬──────────────┘
                      ↓
            ┌─────────────────────┐
            │  FUZZIFICAÇÃO       │
            │  (valores → fuzzy)  │
            └──────────┬──────────┘
                      ↓
            ┌─────────────────────┐
            │  INFERÊNCIA FUZZY   │
            │  (regras)           │
            └──────────┬──────────┘
                      ↓
            ┌─────────────────────┐
            │  DEFUZZIFICAÇÃO     │
            │  (fuzzy → valores)  │
            └──────────┬──────────┘
                      ↓
              Emoção + Explicação
```

---

## 🔬 **Fundamentos Teóricos**

### **Lógica Fuzzy**

Ao contrário da lógica booleana (0 ou 1), a lógica fuzzy permite **graus de verdade**:

```python
# Lógica Clássica:
if brilho < 0.3:
    return "escuro"  # Abrupto!

# Lógica Fuzzy:
μ(muito_escuro) = 0.7  # 70% muito escuro
μ(escuro) = 0.3        # 30% escuro
# Transição suave!
```

**Componentes:**
1. **Fuzzificação**: Converter valores numéricos em graus de pertinência
2. **Regras Fuzzy**: `SE brightness É muito_escuro E color É frio ENTÃO sadness É alta`
3. **Inferência**: Aplicar regras e combinar resultados
4. **Defuzzificação**: Converter de volta para valores numéricos

### **Integração com CNN**

A CNN (ResNet) **não é substituída**, ela é **complementada**:

- **CNN faz**: Reconhecimento de padrões complexos (faces, objetos, texturas)
- **Fuzzy faz**: Raciocínio interpretável sobre esses padrões
- **Resultado**: Precisão + Explicabilidade

---

## 📁 **Estrutura do Projeto**

```
fuzzy-brain/
├── configs/                    # Configurações
│   └── fuzzy_rules.yaml       # Regras fuzzy
│
├── fuzzy_brain/               # Pacote principal
│   ├── extractors/            # Extração de features
│   │   ├── visual.py          # Features visuais (cor, textura)
│   │   └── semantic.py        # Features semânticas (CNN)
│   │
│   ├── fuzzy/                 # Sistema Fuzzy
│   │   ├── variables.py       # Variáveis fuzzy
│   │   ├── rules.py           # Regras fuzzy
│   │   └── system.py          # Inferência fuzzy
│   │
│   ├── integration/           # Integração Neural-Fuzzy
│   │   ├── hybrid.py          # Sistema híbrido
│   │   └── explainer.py       # Geração de explicações
│   │
│   └── utils/                 # Utilitários
│       └── visualization.py   # Visualizações
│
├── notebooks/                 # Análises exploratórias
│   ├── 01_feature_analysis.ipynb
│   ├── 02_fuzzy_system_test.ipynb
│   └── 03_hybrid_evaluation.ipynb
│
├── scripts/                   # Scripts executáveis
│   ├── extract_features.py
│   ├── test_fuzzy.py
│   └── demo.py
│
└── tests/                     # Testes unitários
```

---

## 🚀 **Instalação**

### **1. Criar ambiente virtual**

```bash
cd fuzzy-brain
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate     # Windows
```

### **2. Instalar dependências**

```bash
pip install -r requirements.txt
```

### **3. Verificar instalação**

```bash
python -c "import skfuzzy; import cv2; import torch; print('✅ Tudo OK!')"
```

---

## 📚 **Dataset: ArtEmis**

Este projeto usa o **ArtEmis dataset** (CVPR 2021):

- 80.031 pinturas do WikiArt
- 454.684 anotações humanas
- 9 emoções: amusement, awe, contentment, excitement, anger, disgust, fear, sadness, something_else

**Referência:** [ArtEmis: Affective Language for Visual Art](https://arxiv.org/abs/2101.07396)

---

## 🎯 **Roadmap de Desenvolvimento**

### **Fase 1: Prototipagem (2 semanas)** ← VOCÊ ESTÁ AQUI
- [x] Estrutura do projeto
- [ ] Extrator de features visuais
- [ ] Sistema fuzzy básico (15-20 regras)
- [ ] Testes unitários

### **Fase 2: Integração (2 semanas)**
- [ ] Carregar modelo SAT treinado
- [ ] Integração Neural-Fuzzy
- [ ] Gerador de explicações
- [ ] Visualizações

### **Fase 3: Avaliação (1 semana)**
- [ ] Métricas quantitativas
- [ ] Estudo com usuários
- [ ] Comparação com baseline

---

## 📖 **Referências Teóricas**

### **Lógica Fuzzy:**
- Zadeh, L. A. (1965). "Fuzzy Sets". *Information and Control*
- Mamdani, E. H. (1974). "Application of fuzzy algorithms for control of simple dynamic plant"

### **Psicologia das Cores:**
- Valdez, P. & Mehrabian, A. (1994). "Effects of color on emotions"
- Palmer, S. E. & Schloss, K. B. (2010). "An ecological valence theory of human color preference"

### **Deep Learning + Fuzzy:**
- Melin, P. & Castillo, O. (2014). "A review on type-2 fuzzy logic applications in clustering, classification and pattern recognition"

---

## 👥 **Contribuição**

Este é um projeto de pesquisa acadêmica. Sugestões e melhorias são bem-vindas!

---

## 📄 **Licença**

MIT License - Veja LICENSE para detalhes.

---

**Desenvolvido com 🧠 e ❤️ para a disciplina de Lógica Fuzzy**
