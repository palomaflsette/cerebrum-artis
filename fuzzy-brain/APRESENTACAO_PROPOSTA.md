# 🎨 Apresentação: Sistema Híbrido Neural-Fuzzy para Explicabilidade em Affective Image Captioning

> **Disciplina:** Lógica Fuzzy  
> **Formato:** Vídeo de apresentação (7-10 minutos)  
> **Estrutura:** Introdução → Motivação → Objetivo → Trabalhos Relacionados → Metodologia → Conclusão

---

## 📊 **ESTRUTURA DA APRESENTAÇÃO**

| # | Seção | Tempo | Slides |
|---|-------|-------|--------|
| 1 | Introdução | 0.5min | 1 slide |
| 2 | Contexto: Projeto ArtEmis | 1.5min | 1 slide |
| 3 | Motivação | 1.5min | 2 slides |
| 4 | Objetivo | 1min | 1 slide |
| 5 | Trabalhos Relacionados | 1min | 1 slide |
| 6 | Metodologia Proposta | 3min | 4 slides |
| 7 | Status Atual | 0.5min | 1 slide |
| 8 | Resultados Esperados | 0.5min | 1 slide |
| 9 | Conclusão | 0.5min | 1 slide |
| **TOTAL** | **10min** | **13 slides** |

---

# 📑 **SLIDES COM FALAS**

---

## **SLIDE 1: Introdução** [30 segundos]

### Visual:
```
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║   Sistema Híbrido Neural-Fuzzy para                        ║
║   Explicabilidade em Affective Image Captioning            ║
║                                                              ║
║   Adicionando Lógica Fuzzy ao Projeto ArtEmis              ║
║   para Gerar Explicações Interpretáveis                    ║
║                                                              ║
║   [Imagem: Pintura + CNN + Fuzzy + Explicação]             ║
║                                                              ║
║   Aluna: Paloma Sette                                       ║
║   Disciplina: Lógica Fuzzy                                  ║
║   Novembro 2025                                             ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
```

### Fala:
> "Olá, professora! Meu nome é Paloma e hoje vou apresentar a proposta do meu projeto para a disciplina de Lógica Fuzzy. 
>
> O tema é **Sistema Híbrido Neural-Fuzzy para Explicabilidade em Affective Image Captioning**, onde vou adicionar uma camada de lógica fuzzy ao projeto ArtEmis - um sistema já existente de análise emocional de arte - para gerar explicações interpretáveis sobre as emoções que as pinturas evocam.
>
> Vou começar contextualizando o projeto ArtEmis que serve de base, depois apresentar a motivação, objetivos e a metodologia proposta."

**[Transição: 0.5 minuto]**

---

## **SLIDE 2: Contexto - O Projeto ArtEmis** [1.5 minutos]

### Visual:
```
╔══════════════════════════════════════════════════════════════╗
║  📚 CONTEXTO: O PROJETO ARTEMIS (BASE DO TRABALHO)          ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  O QUE É ARTEMIS?                                           ║
║    • Dataset publicado no CVPR 2021                         ║
║    • Stanford + École Polytechnique + KAUST                 ║
║    • "Affective Language for Visual Art"                    ║
║                                                              ║
║  DATASET:                                                   ║
║    • 80.031 pinturas do WikiArt                            ║
║    • 454.684 anotações emocionais humanas                  ║
║    • 9 emoções: amusement, awe, contentment,               ║
║      excitement, anger, disgust, fear, sadness,            ║
║      something_else                                         ║
║                                                              ║
║  SISTEMA NEURAL JÁ DISPONÍVEL:                              ║
║    • Show, Attend and Tell (SAT)                           ║
║    • CNN (ResNet-34) + LSTM + Attention                    ║
║    • ~75% accuracy em classificação emocional               ║
║    • Gera captions: "This painting makes me feel..."       ║
║                                                              ║
║  ✅ JÁ TENHO:                                               ║
║    • Dataset completo e preprocessado                       ║
║    • Modelo SAT treinado e funcional                       ║
║    • Pipeline de avaliação                                  ║
║                                                              ║
║  ❌ O QUE FALTA:                                            ║
║    • EXPLICABILIDADE - Por que aquela emoção?              ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
```

### Fala:
> "Antes de falar sobre meu projeto, preciso contextualizar a base sobre a qual vou trabalhar: o projeto **ArtEmis**.
>
> ArtEmis é um dataset e sistema de pesquisa publicado no CVPR 2021 por uma colaboração entre Stanford, École Polytechnique e KAUST. O dataset contém mais de 80 mil pinturas do WikiArt com 454 mil anotações emocionais feitas por humanos, cobrindo 9 categorias de emoções.
>
> O projeto já vem com um sistema neural completo: o **Show, Attend and Tell** - que usa uma CNN ResNet-34 combinada com LSTM e mecanismo de atenção. Esse modelo atinge cerca de 75% de acurácia na classificação emocional e gera captions do tipo 'esta pintura me faz sentir triste porque...'.
>
> **Importante**: eu **já tenho** todo o dataset preprocessado, o modelo SAT já treinado e funcionando, e o pipeline de avaliação pronto. Ou seja, **não vou treinar CNN do zero** - vou trabalhar em cima de uma base sólida que já existe.
>
> O que **falta** nesse sistema - e é aí que entra minha contribuição com lógica fuzzy - é a **explicabilidade**. O modelo prevê a emoção, mas não explica de forma interpretável **por que** chegou naquela conclusão. É isso que vou resolver."

**[Transição: 2 minutos acumulados]**

---

## **SLIDE 3: Motivação - O Problema** [1 minuto]

### Visual:
```
╔══════════════════════════════════════════════════════════════╗
║  ❓ MOTIVAÇÃO: O Problema da Black Box no ArtEmis          ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║   MODELO SAT ATUAL:                                         ║
║                                                              ║
║   Imagem → [CNN + LSTM 🔲] → "Tristeza" (0.85)             ║
║                               ↑                              ║
║                          BLACK BOX                           ║
║                                                              ║
║   OUTPUT:                                                   ║
║   "This painting makes me feel sad"                         ║
║                                                              ║
║   ❌ PROBLEMAS:                                             ║
║   • NÃO explica POR QUE é tristeza                         ║
║   • Não é interpretável por humanos                         ║
║   • Difícil validar cientificamente                         ║
║   • Pouco útil para educação/museus                        ║
║                                                              ║
║   💡 SOLUÇÃO PROPOSTA:                                      ║
║   Adicionar camada de Lógica Fuzzy para gerar              ║
║   explicações interpretáveis baseadas em propriedades       ║
║   visuais (cor, composição, textura)                        ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
```

### Fala:
> "Agora que entendemos o contexto, vamos à **motivação** do projeto.
>
> O modelo SAT do ArtEmis, embora preciso, é uma **black box**. Ele diz 'esta pintura evoca tristeza com 85% de confiança' e gera um caption genérico como 'this painting makes me feel sad', mas **não explica o porquê**.
>
> Isso gera quatro problemas principais: primeiro, não sabemos se foi pelas cores escuras, composição, ou outro fator. Segundo, não é interpretável - humanos não conseguem entender o raciocínio. Terceiro, dificulta validação científica. E quarto, limita aplicações educativas em museus ou ensino de arte.
>
> Minha proposta é adicionar uma **camada de lógica fuzzy** ao sistema existente para gerar explicações interpretáveis baseadas em propriedades visuais concretas como cor, composição e textura - mantendo o modelo neural que já funciona bem, mas adicionando a explicabilidade que falta."

**[Transição: 3 minutos acumulados]**

---

## **SLIDE 4: Motivação - Por que Lógica Fuzzy?** [30 segundos]

### Visual:
```
╔══════════════════════════════════════════════════════════════╗
║  💡 POR QUE LÓGICA FUZZY É A SOLUÇÃO?                       ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  1. NATUREZA DA ARTE                                        ║
║     • Conceitos artísticos são VAGOS e SUBJETIVOS           ║
║     • "Escuro", "vibrante", "harmônico" → Fuzzy!           ║
║                                                              ║
║  2. INTERPRETABILIDADE                                      ║
║     • Regras fuzzy são legíveis por humanos                ║
║     • "SE muito_escuro E frio → tristeza"                   ║
║                                                              ║
║  3. GRADUALIDADE                                            ║
║     • Transições suaves entre estados                       ║
║     • Combina múltiplos fatores de forma natural            ║
║                                                              ║
║  4. BASE CIENTÍFICA                                         ║
║     • Regras baseadas em psicologia das cores               ║
║     • Teoria da estética validada                           ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
```

### Fala:
> "E por que especificamente lógica fuzzy? Por quatro motivos principais:
>
> **Primeiro**, a própria natureza da arte: conceitos artísticos são inerentemente vagos e subjetivos. Palavras como 'escuro', 'vibrante', 'harmônico' não têm fronteiras rígidas - são fuzzy por natureza.
>
> **Segundo**, interpretabilidade: regras fuzzy podem ser lidas e entendidas por humanos. Por exemplo, 'SE o brilho é muito escuro E a temperatura é fria, ENTÃO evoca tristeza'.
>
> **Terceiro**, gradualidade: a lógica fuzzy permite transições suaves e combina múltiplos fatores de forma natural, exatamente como fazemos julgamentos estéticos.
>
> **E quarto**, as regras podem ser fundamentadas em conhecimento científico estabelecido - psicologia das cores, teoria da estética - tornando o sistema validável."

**[Transição: 3.5 minutos acumulados]**

---

## **SLIDE 5: Objetivo** [1 minuto]

### Visual:
```
╔══════════════════════════════════════════════════════════════╗
║  🎯 OBJETIVO DO PROJETO                                     ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  ADICIONAR ao ArtEmis existente:                            ║
║                                                              ║
║  ✅ Camada de LÓGICA FUZZY para explicabilidade            ║
║                                                              ║
║  ✅ Mantém modelo SAT já treinado (precisão)               ║
║     + Adiciona sistema fuzzy (interpretabilidade)           ║
║                                                              ║
║  ✅ Sistema híbrido que gera:                              ║
║                                                              ║
║     OUTPUT ATUAL (SAT):                                     ║
║     "This painting makes me feel sad"                       ║
║                                                              ║
║     OUTPUT PROPOSTO (SAT + Fuzzy):                         ║
║     "Esta pintura evoca TRISTEZA (0.83) porque:            ║
║      • Brilho: muito_escuro (0.15)                         ║
║      • Temperatura: fria (0.25)                            ║
║      • Saturação: dessaturada (0.22)                       ║
║      • Composição: simples e solitária (0.30)              ║
║                                                              ║
║      Regras fuzzy ativadas: RULE1 (0.8), RULE2 (0.6)"     ║
║                                                              ║
║  🎯 FOCO: Explicabilidade via Lógica Fuzzy                  ║
║     (não treinar CNN do zero!)                              ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
```

### Fala:
> "O objetivo é claro: **adicionar** uma camada de lógica fuzzy ao ArtEmis existente - não criar tudo do zero.
>
> Vou **manter** o modelo SAT que já está treinado e funcionando bem, garantindo a precisão, e **adicionar** em cima dele um sistema de lógica fuzzy para interpretabilidade.
>
> A diferença no output será significativa: hoje, o SAT gera 'this painting makes me feel sad' - genérico e sem explicação. Com minha proposta, o sistema dirá 'esta pintura evoca tristeza com 83% de confiança **porque** o brilho é muito escuro, a temperatura de cor é fria, a saturação é baixa, e a composição é simples e solitária' - mostrando inclusive quais regras fuzzy foram ativadas.
>
> **Importante reforçar**: meu foco é na **explicabilidade via lógica fuzzy**. A parte de deep learning já está pronta - não vou treinar CNN do zero. Vou trabalhar de forma inteligente em cima do que já existe."

**[Transição: 4.5 minutos acumulados]**

---

## **SLIDE 6: Trabalhos Relacionados** [1 minuto]

### Visual:
```
╔══════════════════════════════════════════════════════════════╗
║  📚 TRABALHOS RELACIONADOS                                  ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  1️⃣ ARTEMIS (CVPR 2021)                                    ║
║     • Show, Attend and Tell (SAT) para arte                ║
║     • 75% accuracy em classificação emocional               ║
║     ❌ Sem explicabilidade                                  ║
║                                                              ║
║  2️⃣ ARTEMIS v2 (CVPR 2022)                                 ║
║     • Contrastive learning para emoções                     ║
║     • Meshed Memory Transformer (M2)                        ║
║     ❌ Ainda black box                                      ║
║                                                              ║
║  3️⃣ Psicologia das Cores                                   ║
║     • Valdez & Mehrabian (1994): cores → emoções           ║
║     • Palmer & Schloss (2010): preferências cromáticas     ║
║     ✅ Base científica para regras fuzzy                    ║
║                                                              ║
║  4️⃣ Neuro-Fuzzy Systems                                    ║
║     • Melin & Castillo (2014): Type-2 fuzzy + DL           ║
║     • Aplicações: classificação, pattern recognition        ║
║     💡 Mas não em affective captioning                      ║
║                                                              ║
║  🆕 NOSSA CONTRIBUIÇÃO:                                     ║
║     Primeira aplicação de Lógica Fuzzy em                   ║
║     affective image captioning para explicabilidade         ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
```

### Fala:
> "Rapidamente sobre trabalhos relacionados - divididos em três áreas:
>
> **Base técnica**: O próprio ArtEmis que já descrevi, com os modelos SAT e M2 Transformer, ambos precisos mas sem explicabilidade.
>
> **Base teórica**: Trabalhos de psicologia das cores - Valdez e Mehrabian (1994) sobre cores e emoções, Palmer e Schloss (2010) sobre preferências cromáticas. Esses fornecem a fundamentação científica para as regras fuzzy.
>
> **Sistemas híbridos**: Melin e Castillo (2014) revisaram aplicações de neuro-fuzzy em classificação, mas não em affective captioning.
>
> **Nossa contribuição** é aplicar lógica fuzzy especificamente para **adicionar explicabilidade** a um sistema neural já existente de affective captioning - algo inédito nesse domínio."

**[Transição: 5.5 minutos acumulados]**

---

## **SLIDE 7: Metodologia - Arquitetura Geral** [1 minuto]

### Visual:
```
╔══════════════════════════════════════════════════════════════╗
║  🏗️ ARQUITETURA DO SISTEMA HÍBRIDO                         ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║                    IMAGEM (Pintura)                          ║
║                          ↓                                   ║
║          ┌───────────────┴───────────────┐                  ║
║          ↓                               ↓                   ║
║   ┌─────────────┐               ┌─────────────┐             ║
║   │   CAMINHO   │               │   CAMINHO   │             ║
║   │   NEURAL    │               │    FUZZY    │             ║
║   │  (EXISTENTE)│               │   (NOVO!)   │             ║
║   │             │               │             │             ║
║   │ CNN (ResNet)│               │  Extração   │             ║
║   │ ✅ treinada │               │    Visual   │             ║
║   │     ↓       │               │     ↓       │             ║
║   │  Features   │               │  Features   │             ║
║   │     ↓       │               │  (brilho,   │             ║
║   │ SAT Model   │               │   saturação,│             ║
║   │ ✅ treinado │               │   etc.)     │             ║
║   │     ↓       │               │     ↓       │             ║
║   │ Emoção:0.85 │               │ Fuzzificação│             ║
║   │             │               │     ↓       │             ║
║   │             │               │  Inferência │             ║
║   │             │               │    Fuzzy    │             ║
║   │             │               │     ↓       │             ║
║   │             │               │ Emoção:0.78 │             ║
║   └──────┬──────┘               └──────┬──────┘             ║
║          │                             │                     ║
║          └─────────┬───────────────────┘                    ║
║                    ↓                                         ║
║          ┌─────────────────┐                                ║
║          │  FUSÃO (70-30%) │                                ║
║          └────────┬─────────                                ║
║                   ↓                                          ║
║          Emoção Final: 0.83                                 ║
║               +                                              ║
║          Explicação Interpretável                           ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
```

### Fala:
> "A arquitetura proposta é híbrida com dois caminhos paralelos:
>
> **Caminho Neural - que já existe**: Uso a CNN ResNet e o modelo SAT que **já estão treinados**. Eles geram a predição emocional com boa precisão - por exemplo, tristeza 0.85. Este caminho garante a **precisão** e já está pronto.
>
> **Caminho Fuzzy - minha contribuição**: Extraio features visuais básicas como brilho, saturação, temperatura de cor diretamente da imagem. Estas features passam por fuzzificação e alimentam um sistema de inferência fuzzy com regras interpretáveis que vou desenvolver. Este caminho gera também uma predição - por exemplo, tristeza 0.78 - **mas com explicação**.
>
> A fusão ponderada dos dois caminhos - 70% neural, 30% fuzzy - gera a predição final (0.83) **mais** a explicação completa. O diferencial é que não estou reinventando a roda - estou adicionando explicabilidade a um sistema neural que já funciona."

**[Transição: 6.5 minutos acumulados]**

---

## **SLIDE 8: Metodologia - Features Visuais** [1 minuto]

### Visual:
```
╔══════════════════════════════════════════════════════════════╗
║  📊 FEATURES VISUAIS EXTRAÍDAS                              ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  A. FEATURES DE COR                                         ║
║     • Brightness (Brilho): [0,1] - HSV Value                ║
║     • Color Temperature: [0,1] - Quente vs Frio             ║
║     • Saturation: [0,1] - Vivacidade                        ║
║     • Color Harmony: [0,1] - Entropia de Matizes            ║
║                                                              ║
║  B. FEATURES DE COMPOSIÇÃO                                  ║
║     • Complexity: [0,1] - Densidade de Edges (Canny)        ║
║     • Symmetry: [0,1] - Correlação Espacial                 ║
║                                                              ║
║  C. FEATURES DE TEXTURA                                     ║
║     • Roughness: [0,1] - Local Binary Patterns (LBP)        ║
║                                                              ║
║  Todas normalizadas em [0,1] e                              ║
║  baseadas em literatura científica!                         ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
```

### Fala:
> "As features visuais que extraímos se dividem em três categorias:
>
> **Features de cor**: Brilho calculado via canal V do espaço HSV, temperatura de cor baseada no ratio de cores quentes versus frias, saturação média, e harmonia cromática usando entropia da distribuição de matizes.
>
> **Features de composição**: Complexidade medida pela densidade de edges usando detector Canny, e simetria via correlação espacial entre a imagem e sua versão espelhada.
>
> **Features de textura**: Aspereza calculada usando Local Binary Patterns, que capturam micropadrões na textura da pincelada.
>
> Todas estas features são normalizadas entre 0 e 1, e cada uma tem fundamentação em literatura científica de visão computacional e psicologia da percepção."

**[Transição: 7.5 minutos acumulados]**

---

## **SLIDE 9: Metodologia - Sistema Fuzzy** [1 minuto]

### Visual:
```
╔══════════════════════════════════════════════════════════════╗
║  🔀 SISTEMA DE LÓGICA FUZZY                                 ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  1️⃣ VARIÁVEIS FUZZY (Membership Functions)                 ║
║                                                              ║
║     Brightness: {muito_escuro, escuro, médio,               ║
║                  claro, muito_claro}                        ║
║                                                              ║
║     μ(x)  1.0 ┤  ╱╲     ╱╲     ╱╲                          ║
║           0.5 ┤ ╱  ╲   ╱  ╲   ╱  ╲                         ║
║           0.0 ┼┴────┴─┴────┴─┴────┴─► x                     ║
║               0   0.3  0.5  0.7  1.0                        ║
║                                                              ║
║  2️⃣ REGRAS FUZZY (Exemplos)                                ║
║                                                              ║
║     REGRA 1 (Tristeza):                                     ║
║     SE brightness É muito_escuro                            ║
║     E color_temp É frio                                     ║
║     E saturation É dessaturado                              ║
║     ENTÃO sadness É alta (0.8)                              ║
║                                                              ║
║     REGRA 2 (Admiração):                                    ║
║     SE symmetry É muito_simétrico                           ║
║     E color_harmony É muito_harmônico                       ║
║     ENTÃO awe É alta (0.85)                                 ║
║                                                              ║
║  3️⃣ INFERÊNCIA: Mamdani + Defuzzificação (CoG)             ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
```

### Fala:
> "O sistema de lógica fuzzy tem três componentes principais:
>
> **Primeiro**, definimos variáveis fuzzy com funções de pertinência triangulares. Por exemplo, brilho tem cinco termos linguísticos: muito escuro, escuro, médio, claro, muito claro. Cada valor numérico tem graus de pertinência parciais em múltiplos termos.
>
> **Segundo**, criamos regras fuzzy baseadas em conhecimento especialista e psicologia. Por exemplo: 'SE o brilho é muito escuro E a temperatura é fria E a saturação é baixa, ENTÃO tristeza é alta'. Ou 'SE há alta simetria E alta harmonia cromática, ENTÃO admiração é alta'. Planejamos implementar 15 a 20 regras cobrindo as 9 emoções do dataset.
>
> **Terceiro**, usamos inferência Mamdani com defuzzificação por centro de gravidade para converter os resultados fuzzy de volta em valores numéricos que podem ser fundidos com o caminho neural."

**[Transição: 8.5 minutos acumulados]**

---

## **SLIDE 10: Metodologia - Implementação** [30 segundos]

### Visual:
```
╔══════════════════════════════════════════════════════════════╗
║  💻 IMPLEMENTAÇÃO TÉCNICA                                   ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  TECNOLOGIAS:                                               ║
║                                                              ║
║    • Python 3.8+                                            ║
║    • scikit-fuzzy (sistema fuzzy)                          ║
║    • PyTorch (CNN e modelo SAT)                            ║
║    • OpenCV (processamento de imagem)                       ║
║    • scikit-image (features de textura)                    ║
║                                                              ║
║  ESTRUTURA DO CÓDIGO:                                       ║
║                                                              ║
║    fuzzy-brain/              ← NOVO (meu trabalho)          ║
║    ├── extractors/     (features visuais)                  ║
║    ├── fuzzy/          (sistema fuzzy - a desenvolver)     ║
║    ├── integration/    (fusão neural-fuzzy)                ║
║    └── utils/          (visualização)                      ║
║                                                              ║
║    artemis/                  ← EXISTENTE (base)             ║
║    ├── neural_speaker/sat/   (SAT treinado ✅)             ║
║    ├── dataset/              (ArtEmis preprocessado ✅)    ║
║    └── ...                                                  ║
║                                                              ║
║  ✅ O QUE JÁ TENHO:                                         ║
║    • Modelo SAT treinado (75% accuracy)                    ║
║    • Dataset completo                                       ║
║    • Extrator de features visuais implementado             ║
║                                                              ║
║  🔄 O QUE VOU DESENVOLVER:                                  ║
║    • Sistema fuzzy (variáveis, regras, inferência)         ║
║    • Integração neural-fuzzy                                ║
║    • Gerador de explicações                                 ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
```

### Fala:
> "Rapidamente sobre implementação: estou criando um novo módulo chamado **fuzzy-brain** separado do código original do ArtEmis, para manter organização.
>
> **O que já tenho pronto**: O modelo SAT treinado com 75% de acurácia, o dataset completo preprocessado, e já implementei o extrator de features visuais.
>
> **O que vou desenvolver agora**: O sistema de lógica fuzzy propriamente dito - variáveis fuzzy, regras, inferência Mamdani - e o módulo de integração que faz a fusão neural-fuzzy e gera as explicações.
>
> Ou seja, estou trabalhando de forma **incremental** em cima de uma base sólida."

**[Transição: 9 minutos acumulados]**

---

## **SLIDE 11: Status Atual** [30 segundos]

### Visual:
```
╔══════════════════════════════════════════════════════════════╗
║  ✅ STATUS ATUAL DO PROJETO (17/11/2024)                   ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  INFRAESTRUTURA (100% ✅)                                   ║
║    ✅ Dataset ArtEmis preprocessado (80k pinturas)          ║
║    ✅ Modelo SAT treinado (epoch 7, val_NLL: 3.393)        ║
║    ✅ Pipeline de avaliação configurado                     ║
║                                                              ║
║  EXTRAÇÃO DE FEATURES (100% ✅)                             ║
║    ✅ 7 features visuais implementadas e testadas           ║
║       • Brilho, saturação, temperatura de cor               ║
║       • Harmonia, complexidade, simetria, textura           ║
║    ✅ 12 unit tests (100% passing)                          ║
║    ✅ Validação com pinturas reais do WikiArt               ║
║                                                              ║
║  SISTEMA FUZZY (EM DESENVOLVIMENTO 🚧)                      ║
║    ✅ Variáveis fuzzy definidas (5 termos/variável)         ║
║    🚧 Regras em implementação (progresso atual)             ║
║    🚧 Inferência Mamdani (próximos dias)                    ║
║                                                              ║
║  CRONOGRAMA ATÉ 01/12 (14 dias):                            ║
║    📅 17-22/11: Sistema fuzzy completo                      ║
║    📅 23-27/11: Integração neural-fuzzy                     ║
║    📅 28-30/11: Avaliação e ajustes finais                  ║
║    📅 01/12: Entrega! 🎯                                    ║
║                                                              ║
║  VIÁVEL: Sim! Base sólida + escopo focado = factível ✅    ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
```

### Fala:
> "Rapidamente sobre onde estamos hoje, 17 de novembro:
>
> A **infraestrutura está 100% pronta** - dataset preprocessado, modelo SAT treinado com validation NLL de 3.393, tudo funcionando.
>
> O **extrator de features visuais está completo** - 7 features implementadas, testadas com 12 unit tests, todos passando. Já validei com pinturas reais do WikiArt e está funcionando perfeitamente.
>
> O **sistema fuzzy está em desenvolvimento** - variáveis já definidas, estou implementando as regras agora.
>
> **Cronograma até 01 de dezembro**: Próximos 5 dias finalizo o sistema fuzzy, depois 5 dias para integração neural-fuzzy, e últimos 3 dias para avaliação e ajustes. É apertado mas **totalmente viável** porque a base já está sólida e o escopo está bem focado."

**[Transição: 9.5 minutos acumulados]**

---

## **SLIDE 12: Resultados Esperados** [30 segundos]

### Visual:
```
╔══════════════════════════════════════════════════════════════╗
║  📈 RESULTADOS ESPERADOS                                    ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  MÉTRICAS QUANTITATIVAS:                                    ║
║    • Acurácia similar ao baseline neural (~75%)             ║
║    • Concordância Neural-Fuzzy > 70%                        ║
║    • Cobertura de regras > 90% dos casos                    ║
║                                                              ║
║  MÉTRICAS QUALITATIVAS:                                     ║
║    • Explicações interpretáveis por humanos                 ║
║    • Justificativas alinhadas com teoria                    ║
║    • Utilidade para aplicações educativas                   ║
║                                                              ║
║  VALIDAÇÃO:                                                 ║
║    • Comparação com anotações humanas                       ║
║    • Estudo com usuários (preferência)                      ║
║    • Análise de casos onde neural e fuzzy divergem          ║
║                                                              ║
║  CONTRIBUIÇÃO CIENTÍFICA:                                   ║
║    • Primeira aplicação de Lógica Fuzzy em                  ║
║      affective image captioning                             ║
║    • Framework reproduzível e extensível                    ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
```

### Fala:
> "Como resultados esperados, em termos quantitativos, espero manter acurácia similar ao baseline neural - em torno de 75% - com alta concordância entre os caminhos neural e fuzzy, acima de 70%.
>
> Qualitativamente, o mais importante são as explicações: elas devem ser interpretáveis por humanos, alinhadas com teoria científica, e úteis para aplicações educativas.
>
> A validação será feita comparando com anotações humanas do dataset, e idealmente com um pequeno estudo de usuários para medir preferência por explicações.
>
> A contribuição científica principal é ser a primeira aplicação de lógica fuzzy especificamente para explicabilidade em affective captioning, com um framework que pode ser estendido para outros domínios."

**[Transição: 9.5 minutos acumulados]**

---

## **SLIDE 13: Conclusão** [30 segundos]

### Visual:
```
╔══════════════════════════════════════════════════════════════╗
║  ✨ CONCLUSÃO                                               ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  RESUMO DO PROJETO:                                         ║
║                                                              ║
║    ✅ ADICIONA Lógica Fuzzy ao ArtEmis existente           ║
║    ✅ Aproveita modelo SAT já treinado (precisão)          ║
║    ✅ Desenvolve camada fuzzy (explicabilidade)            ║
║    ✅ Fundamentado em teoria científica                     ║
║    ✅ Foco em explicabilidade, não em treinar CNN          ║
║                                                              ║
║  POR QUE É RELEVANTE PARA LÓGICA FUZZY?                     ║
║                                                              ║
║    • Demonstra poder da lógica fuzzy em problemas reais     ║
║    • Mostra como complementar (não substituir) DL           ║
║    • Aplica conceitos teóricos em domínio criativo          ║
║    • Prova utilidade de raciocínio gradual e interpretável  ║
║                                                              ║
║  CRONOGRAMA DE ENTREGA (ATÉ 01/12):                         ║
║                                                              ║
║    📅 17-22/11: Sistema fuzzy completo                      ║
║    📅 23-27/11: Integração neural-fuzzy                     ║
║    📅 28-30/11: Avaliação e documentação                    ║
║    🎯 01/12: Entrega final!                                 ║
║                                                              ║
║  📧 Dúvidas? Obrigada pela atenção!                         ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
```

### Fala:
> "Para concluir: este projeto **adiciona** lógica fuzzy ao ArtEmis, aproveitando inteligentemente um modelo neural já treinado e focando no desenvolvimento da camada de explicabilidade.
>
> Este projeto é relevante para Lógica Fuzzy porque demonstra como a lógica fuzzy pode **complementar** deep learning em problemas reais, adicionando interpretabilidade sem perder precisão. Mostra também a aplicação de conceitos teóricos - membership functions, regras fuzzy, inferência Mamdani - em um domínio criativo e subjetivo.
>
> **Importante**: não estou criando tudo do zero. Já tenho modelo SAT treinado, dataset pronto, e extrator de features implementado. Vou focar em desenvolver o sistema fuzzy, fazer a integração, e avaliar os resultados.
>
> Agradeço a atenção e fico à disposição para dúvidas!"

**[FIM - Total: 10 minutos]**

---

# 🎬 **DICAS PARA GRAVAÇÃO DO VÍDEO**

## Preparação:

1. **Ensaie com timer** - Cronometre cada seção
2. **Marque pausas** - Respire entre slides
3. **Prepare transições** - Frases de conexão suaves
4. **Slides prontos** - PowerPoint, Google Slides ou PDF

## Durante a gravação:

1. **Postura** - Olhe para a câmera como se fosse a professora
2. **Ritmo** - Fale claramente, não muito rápido
3. **Entusiasmo** - Mostre que você está animada com o projeto!
4. **Gestos** - Use as mãos para enfatizar pontos importantes
5. **Pausas estratégicas** - Após pontos-chave, pause 1-2 segundos

## Estrutura de cada slide:

```
1. Apresente o título do slide (5s)
2. Contextualize o conteúdo (10-15s)
3. Explique os pontos principais (30-40s)
4. Faça transição para próximo slide (5s)
```

## Frases de transição sugeridas:

- "Agora que entendemos o problema, vamos ao objetivo..."
- "Com isso em mente, vejamos os trabalhos relacionados..."
- "Passando para a metodologia proposta..."
- "Em termos de implementação técnica..."
- "Para finalizar, os resultados esperados..."

---

# 📊 **CHECKLIST PRÉ-GRAVAÇÃO**

- [ ] Slides prontos e revisados
- [ ] Falas ensaiadas pelo menos 2x
- [ ] Timing verificado (7-10 min)
- [ ] Ambiente silencioso
- [ ] Boa iluminação
- [ ] Câmera/microfone testados
- [ ] Tela compartilhada testada (se for screencast)
- [ ] Água por perto (para não secar a garganta)
- [ ] Energia e entusiasmo! 🚀

---

**Boa sorte com a gravação, Paloma! Você vai arrasar! 🎨✨**
