# 🔍 FUZZY EXTRACTOR - Explicação Completa Tim Tim por Tim Tim

## 📊 FLUXOGRAMA COMPLETO DO PROCESSO FUZZY

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                    PIPELINE COMPLETO: IMAGEM → FEATURES FUZZY                    │
└──────────────────────────────────────────────────────────────────────────────────┘

INPUT: painting.jpg (RGB, 512×512 pixels)
   │
   ├─────────────────────────────────────────────────────────────────────────────┐
   │  ETAPA 1: extract_crisp_features(image_path)                                │
   │  Objetivo: Extrair valores numéricos objetivos da imagem                    │
   └─────────────────────────────────────────────────────────────────────────────┘
   │
   ├──> 1.1 Carregar imagem
   │         │
   │         ├─> PIL.Image.open(image_path).convert('RGB')
   │         │   Resultado: Array numpy (512, 512, 3) com valores [0, 255]
   │         │
   │         └─> img_rgb = np.array(image)  # Shape: (H, W, 3)
   │
   ├──> 1.2 Converter para espaço HSV
   │         │
   │         ├─> img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
   │         └─> hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
   │             Resultado: Array (512, 512, 3) com:
   │                        hsv[:,:,0] = Hue (matiz) [0, 179]
   │                        hsv[:,:,1] = Saturation [0, 255]
   │                        hsv[:,:,2] = Value (brilho) [0, 255]
   │
   ├──> 1.3 Extrair Feature 1: BRIGHTNESS
   │         │
   │         ├─> brightness_raw = hsv[:,:,2].mean()  # Média do canal V
   │         │   Exemplo: brightness_raw = 150.2 (em [0, 255])
   │         │
   │         └─> brightness = brightness_raw / 255.0
   │             Resultado: 0.5890 (normalizado para [0, 1])
   │
   ├──> 1.4 Extrair Feature 2: COLOR_TEMPERATURE
   │         │
   │         ├─> r_mean = img_rgb[:,:,0].mean()  # Canal vermelho
   │         │   b_mean = img_rgb[:,:,2].mean()  # Canal azul
   │         │   Exemplo: r_mean = 180, b_mean = 90
   │         │
   │         ├─> temp = (r_mean - b_mean) / 255.0
   │         │   = (180 - 90) / 255.0 = 0.353 (em [-1, 1])
   │         │
   │         └─> color_temperature = (temp + 1) / 2
   │             Resultado: 0.676 (normalizado para [0, 1])
   │
   ├──> 1.5 Extrair Feature 3: SATURATION
   │         │
   │         ├─> saturation_raw = hsv[:,:,1].mean()
   │         │   Exemplo: saturation_raw = 85.4
   │         │
   │         └─> saturation = saturation_raw / 255.0
   │             Resultado: 0.3349
   │
   ├──> 1.6 Extrair Features 4-7: HARMONY, COMPLEXITY, SYMMETRY, ROUGHNESS
   │         │   (cálculos similares omitidos para brevidade)
   │         │
   │         └─> features = {
   │                 'brightness': 0.5890,
   │                 'color_temperature': 0.676,
   │                 'saturation': 0.3349,
   │                 'color_harmony': 0.7234,
   │                 'complexity': 0.4521,
   │                 'symmetry': 0.6789,
   │                 'texture_roughness': 0.3912
   │             }
   │
   └──> OUTPUT ETAPA 1: 7 valores CRISP (precisos) em [0, 1]
        │
        ▼
┌──────────────────────────────────────────────────────────────────────────────────┐
│  ETAPA 2: fuzzify_feature(crisp_value)                                          │
│  Objetivo: Converter valor numérico → graus de pertinência fuzzy                │
└──────────────────────────────────────────────────────────────────────────────────┘
   │
   │  INPUT: brightness = 0.5890 (valor crisp extraído na Etapa 1)
   │
   ├──> 2.1 Definir universo de discurso
   │         │
   │         └─> x = np.arange(0, 1.01, 0.01)
   │             Resultado: [0.00, 0.01, 0.02, ..., 0.99, 1.00]
   │                        101 pontos no eixo X
   │
   ├──> 2.2 Criar 5 funções de pertinência triangulares (trimf)
   │         │
   │         ├─> muito_escuro = trimf(x, [0.0, 0.0, 0.25])
   │         │   escuro       = trimf(x, [0.0, 0.25, 0.5])
   │         │   medio        = trimf(x, [0.25, 0.5, 0.75])
   │         │   claro        = trimf(x, [0.5, 0.75, 1.0])
   │         │   muito_claro  = trimf(x, [0.75, 1.0, 1.0])
   │         │
   │         └─> Cada trimf retorna array [101] com μ(x) para cada ponto x
   │
   ├──────────────────────────────────────────────────────────────────────────────┐
   │  ZOOM: Como funciona trimf([a, b, c]) internamente?                         │
   └──────────────────────────────────────────────────────────────────────────────┘
   │     │
   │     │  INPUT: x (array com 101 pontos), [a, b, c] (parâmetros do triângulo)
   │     │  Exemplo: trimf(x, [0.25, 0.5, 0.75]) para "medio"
   │     │
   │     ├──> Para CADA ponto x_i no array x:
   │     │     │
   │     │     ├─> SE x_i <= a (antes do triângulo):
   │     │     │      μ(x_i) = 0.0
   │     │     │
   │     │     ├─> SE a < x_i <= b (rampa ascendente):
   │     │     │      μ(x_i) = (x_i - a) / (b - a)
   │     │     │      Exemplo: x_i = 0.40, a = 0.25, b = 0.5
   │     │     │               μ(0.40) = (0.40 - 0.25) / (0.5 - 0.25)
   │     │     │                       = 0.15 / 0.25 = 0.6
   │     │     │
   │     │     ├─> SE b < x_i <= c (rampa descendente):
   │     │     │      μ(x_i) = (c - x_i) / (c - b)
   │     │     │      Exemplo: x_i = 0.60, b = 0.5, c = 0.75
   │     │     │               μ(0.60) = (0.75 - 0.60) / (0.75 - 0.5)
   │     │     │                       = 0.15 / 0.25 = 0.6
   │     │     │
   │     │     └─> SE x_i > c (depois do triângulo):
   │     │            μ(x_i) = 0.0
   │     │
   │     └──> OUTPUT: Array [101] com μ(x) para cada ponto
   │                  Exemplo para "medio" [0.25, 0.5, 0.75]:
   │                  [0.0, 0.0, ..., 0.4, 0.8, 1.0, 0.8, 0.4, ..., 0.0]
   │
   ├──> 2.3 Calcular pertinências para brightness = 0.5890
   │         │
   │         ├─> Encontrar índice: idx = int(0.5890 × 100) = 58
   │         │
   │         ├─> Para cada conjunto fuzzy, pegar μ[58]:
   │         │     │
   │         │     ├─> muito_escuro[58] = 0.0   (0.59 > 0.25, FORA do triângulo!)
   │         │     │
   │         │     ├─> escuro[58]       = 0.0   (0.59 > 0.5, FORA do triângulo!)
   │         │     │
   │         │     ├─> medio[58]        = 0.644 (rampa DESC!)
   │         │     │   Cálculo: (c - x) / (c - b)
   │         │     │           = (0.75 - 0.59) / (0.75 - 0.5)
   │         │     │           = 0.16 / 0.25 = 0.644
   │         │     │
   │         │     ├─> claro[58]        = 0.356 (rampa ASC!)
   │         │     │   Cálculo: (x - a) / (b - a)
   │         │     │           = (0.59 - 0.5) / (0.75 - 0.5)
   │         │     │           = 0.09 / 0.25 = 0.356
   │         │     │
   │         │     └─> muito_claro[58]  = 0.0   (0.59 < 0.75, FORA do triângulo!)
   │         │
   │         ├─> 💡 NOTA: medio + claro = 0.644 + 0.356 = 1.0 (overlap!)
   │         │
   │         └─> brightness_fuzzy = {
   │                 'muito_escuro': 0.0,
   │                 'escuro': 0.0,
   │                 'medio': 0.644,
   │                 'claro': 0.356,
   │                 'muito_claro': 0.0
   │             }
   │
   └──> OUTPUT ETAPA 2: 5 graus de pertinência para UMA feature
        │
        ▼
┌──────────────────────────────────────────────────────────────────────────────────┐
│  ETAPA 3: Fuzzificar TODAS as 7 features                                        │
│  Objetivo: Aplicar fuzzificação para todas as features crisp                    │
└──────────────────────────────────────────────────────────────────────────────────┘
   │
   ├──> Para cada feature (7 features):
   │      │
   │      ├─> brightness:        [muito_baixo: 0.0, baixo: 0.6, medio: 0.4, ...]
   │      ├─> color_temperature: [muito_baixo: 0.0, baixo: 0.0, medio: 0.3, ...]
   │      ├─> saturation:        [muito_baixo: 0.2, baixo: 0.8, medio: 0.0, ...]
   │      ├─> color_harmony:     [muito_baixo: 0.0, baixo: 0.0, medio: 0.1, ...]
   │      ├─> complexity:        [muito_baixo: 0.0, baixo: 0.1, medio: 0.9, ...]
   │      ├─> symmetry:          [muito_baixo: 0.0, baixo: 0.0, medio: 0.4, ...]
   │      └─> texture_roughness: [muito_baixo: 0.1, baixo: 0.7, medio: 0.2, ...]
   │
   └──> OUTPUT FINAL: 7 features × 5 termos = 35 valores fuzzy
        │
        │  all_fuzzy = {
        │      'brightness': {muito_baixo: 0.0, baixo: 0.6, medio: 0.4, ...},
        │      'color_temperature': {...},
        │      ...
        │  }
        │
        ▼
┌──────────────────────────────────────────────────────────────────────────────────┐
│  USO FINAL: Regras Fuzzy (V3) ou Concatenação (V2)                              │
└──────────────────────────────────────────────────────────────────────────────────┘
   │
   ├──> V2 (Fuzzy Features): Usa os 7 valores CRISP como input para MLP
   │      features_crisp = [0.59, 0.68, 0.33, 0.72, 0.45, 0.68, 0.39]
   │      ResNet(2048) + RoBERTa(768) + Features(7) = 2823 dims → MLP → 9 emotions
   │
   └──> V3 (Fuzzy Inference): Usa os 35 valores FUZZY em regras Mamdani
        SE brightness É baixo (0.6) E saturation É baixa (0.8)
        ENTÃO sadness É alta (min(0.6, 0.8) = 0.6)
        
        Defuzzificação (centróide) → sadness_crisp = 0.73
        
        9 emoções × inferência fuzzy → [sad: 0.73, awe: 0.21, ...] → MLP → output

```

## 🎯 O QUE É LÓGICA FUZZY?

### Lógica Clássica (Booleana) vs Lógica Fuzzy

```
┌─────────────────────────────────────────────────────────────────┐
│                    LÓGICA CLÁSSICA (Binária)                    │
└─────────────────────────────────────────────────────────────────┘

Pergunta: "A pintura é escura?"

Brilho = 0.49  → NÃO (0)
Brilho = 0.51  → SIM (1)

     Escuro?
       │
   1.0 ├─────────────────────┐
       │                     │
       │                     │
   0.5 ├─────────────────────┤
       │                     │
       │                     │
   0.0 └─────────────────────┘
       0.0   0.5   1.0
           Brilho

⚠️ PROBLEMA: Mudança abrupta em 0.5!
   0.49 = completamente NÃO escuro
   0.51 = completamente escuro
   (mas são quase iguais!)


┌─────────────────────────────────────────────────────────────────┐
│                    LÓGICA FUZZY (Gradual)                       │
└─────────────────────────────────────────────────────────────────┘

Pergunta: "A pintura é escura?"

Brilho = 0.35  → 75% escuro, 25% médio
Brilho = 0.50  → 50% médio, 50% escuro
Brilho = 0.70  → 60% médio, 40% claro

     Grau de Pertinência
       │
   1.0 ├──╱╲────╱╲────╱╲────╱╲────╱╲
       │ ╱  ╲  ╱  ╲  ╱  ╲  ╱  ╲  ╱  ╲
       │╱    ╲╱    ╲╱    ╲╱    ╲╱    ╲
   0.0 └─────────────────────────────────
       muito  escuro médio claro muito
       escuro                     claro
       
       0.0   0.2   0.4   0.6   0.8   1.0
                    Brilho

✅ VANTAGEM: Transição suave!
   0.35 pode ser PARCIALMENTE escuro E PARCIALMENTE médio
```

---

## 🧠 CONCEITOS FUNDAMENTAIS DA LÓGICA FUZZY

### 1. **Variável Linguística**

Variável que usa **palavras** em vez de números:

```
Variável: BRIGHTNESS (brilho)

Valores CLÁSSICOS:    0.0, 0.1, 0.2, ..., 1.0 (números precisos)

Valores FUZZY:        "muito escuro"
                      "escuro"
                      "médio"
                      "claro"
                      "muito claro"
                      (termos linguísticos)
```

### 2. **Conjunto Fuzzy (Fuzzy Set)**

Um conjunto onde cada elemento tem um **grau de pertinência** [0, 1]:

```
Conjunto Clássico "Pinturas Escuras":
  starry_night.jpg    → SIM (1) ou NÃO (0)
  sunflowers.jpg      → SIM (1) ou NÃO (0)

Conjunto Fuzzy "Pinturas Escuras":
  starry_night.jpg    → 0.75 (75% escuro)
  sunflowers.jpg      → 0.10 (10% escuro)
  the_scream.jpg      → 0.60 (60% escuro)
```

### 3. **Função de Pertinência (Membership Function)**

Função matemática que mapeia valor → grau de pertinência.

#### Comparação: Lógica Clássica vs Fuzzy

**LÓGICA CLÁSSICA** (threshold fixo):
```python
def is_escuro_classico(brilho):
    """Retorna 0 ou 1 (tudo ou nada)"""
    if brilho < 0.3:
        return 1.0  # SIM, é escuro
    else:
        return 0.0  # NÃO, não é escuro

# Problema: Transição abrupta!
is_escuro_classico(0.29)  # → 1.0 (100% escuro)
is_escuro_classico(0.31)  # → 0.0 (0% escuro)  ← SALTO!
```

**LÓGICA FUZZY** (função triangular - trimf):
```python
def membership_escuro_fuzzy(brilho):
    """
    Função TRIANGULAR - permite transição suave
    trimf(x, [a, b, c]) onde:
    - a = 0.0  (início)
    - b = 0.25 (pico)
    - c = 0.5  (fim)
    """
    if brilho <= 0.0:
        return 0.0
    elif brilho <= 0.25:
        return brilho / 0.25  # rampa ascendente
    elif brilho <= 0.5:
        return (0.5 - brilho) / 0.25  # rampa descendente
    else:
        return 0.0

# Exemplos - TRANSIÇÃO GRADUAL:
membership_escuro_fuzzy(0.05)  # → 0.20 (20% escuro)
membership_escuro_fuzzy(0.25)  # → 1.00 (100% escuro - PICO)
membership_escuro_fuzzy(0.35)  # → 0.60 (60% escuro - descendo)
membership_escuro_fuzzy(0.50)  # → 0.00 (0% escuro - fim)
```

**Por que 0.35 é "escuro" na lógica fuzzy?**
- O triângulo "escuro" vai de 0.0 até 0.5
- 0.35 está entre o pico (0.25) e o fim (0.5)
- Logo, pertence **parcialmente** ao conjunto "escuro"!

### 4. **Fuzzificação (Fuzzification)**

Converter número **crisp** (preciso) → graus de pertinência **fuzzy** usando **5 triângulos sobrepostos**.

#### 📐 O que significa `trimf([a, b, c])`?

**trimf = Triangular Membership Function** (Função de Pertinência Triangular)

⚠️ **ATENÇÃO**: Os valores `[a, b, c]` são **FIXOS** - você define UMA VEZ e nunca mais muda!

Os 3 números definem os **pontos do triângulo NO EIXO X** (não no eixo Y!):
- **a**: Posição X onde o triângulo **COMEÇA** (base esquerda)
- **b**: Posição X onde está o **PICO** (topo do triângulo)
- **c**: Posição X onde o triângulo **TERMINA** (base direita)

#### 🔺 Exemplo Visual: `escuro = trimf([0.0, 0.25, 0.5])`

```
        EIXO Y                    EIXO Y
    Grau de                    Grau de
  Pertinência μ(x)           Pertinência μ(x)
        ↑                          ↑
    1.0 |      b (0.25, 1.0)   1.0 |      •  ← PICO (100% escuro)
        |      /\                  |     /|\
        |     /  \                 |    / | \
    0.6 |    /   •\            0.6 |   /  |• \  ← input x=0.35 dá μ=0.6
        |   /    ↑ \               |  /   |↑  \
    0.0 |__/_____|__\___       0.0 |_/____|____\_____
        | a    0.35   c            |a     |b    c
        |0.0  0.25  0.5            |0.0  0.25  0.5
        └────────────────→         └──────|───────────→
           EIXO X (Brilho)            EIXO X (Brilho)
                                          |
                                     input x=0.35
```

**IMPORTANTE - Diferença entre EIXO X e EIXO Y:**

| Coisa | Eixo | Valores | É fixo? |
|-------|------|---------|---------|
| **[a, b, c]** | EIXO X | Posições no eixo do brilho | ✅ SIM! Definido uma vez |
| **μ(x)** | EIXO Y | Grau de pertinência [0, 1] | ❌ NÃO! Calculado dinamicamente |

**O que acontece quando você joga DIFERENTES valores de brilho:**

```
Input x=0.00 → μ = 0.0   (na base esquerda 'a')
Input x=0.10 → μ = 0.4   (rampa ascendente)
Input x=0.25 → μ = 1.0   (NO PICO 'b' - 100% escuro!)
Input x=0.35 → μ = 0.6   (rampa descendente)
Input x=0.50 → μ = 0.0   (na base direita 'c')
Input x=0.70 → μ = 0.0   (fora do triângulo)
```

**Fórmula matemática do trimf([a, b, c]):**

```python
def trimf(x, a, b, c):
    if x <= a:
        return 0.0              # Antes do triângulo
    elif a < x <= b:
        return (x - a) / (b - a)  # Rampa ASCENDENTE
    elif b < x <= c:
        return (c - x) / (c - b)  # Rampa DESCENDENTE
    else:  # x > c
        return 0.0              # Depois do triângulo
```

#### ❓ Respondendo sua dúvida: "0.0, 0.0 e 0.25 são valores FIXOS?"

**SIM!** São **FIXOS**. Você define **UMA VEZ** ao criar o sistema fuzzy.

**Exemplo: `muito_escuro = trimf([0.0, 0.0, 0.25])`**

```
Os valores [0.0, 0.0, 0.25] significam:
  a = 0.0  ← Triângulo começa na posição X=0.0
  b = 0.0  ← Pico também está em X=0.0 (triângulo meio degenerado!)
  c = 0.25 ← Triângulo termina na posição X=0.25

EIXO Y                      ⚠️ TRIÂNGULO "RETÂNGULO"!
   ↑
1.0||\                     A e B estão NO MESMO LUGAR (X=0.0)
   || \                    Então só tem rampa DESCENDENTE!
0.6||  \
   ||   \
0.0||____\________________ 
   ||    c               EIXO X
   ab   0.25
  0.0

Isso significa:
  • Se brilho = 0.00 → μ = 1.0 (100% muito_escuro - ESTÁ NO PICO a=b!)
  • Se brilho = 0.10 → μ = 0.6 (60% muito_escuro - rampa desc)
  • Se brilho = 0.25 → μ = 0.0 (0% muito_escuro - fim do triângulo)
  • Se brilho = 0.35 → μ = 0.0 (0% muito_escuro - FORA!)
```

**Por que `a = b = 0.0`?**
Porque queremos dizer: "Brilho ZERO é 100% muito_escuro, e conforme aumenta, vai diminuindo até chegar em 0.25"

Não é triângulo perfeito, é meio "meia rampa"! Mas matematicamente funciona igual.

#### 🎯 Agora a fuzzificação de `brilho = 0.35`:

```
INPUT (crisp):  brilho = 0.35

┌─────────────────┬───────────────────┬──────────────────────────────────┐
│ Termo           │ Triângulo [a,b,c] │ μ(0.35) - Explicação             │
├─────────────────┼───────────────────┼──────────────────────────────────┤
│ muito_escuro    │ [0.00, 0.00, 0.25]│ μ = 0.0                          │
│                 │         /\        │ Por quê? 0.35 > c (0.25)         │
│                 │        /__\       │ Está DEPOIS do fim (c)!          │
│                 │    a,b=0.0  c=0.25│ Portanto: FORA do triângulo      │
├─────────────────┼───────────────────┼──────────────────────────────────┤
│ escuro          │ [0.00, 0.25, 0.50]│ μ = 0.6                          │
│                 │       /\          │ Por quê? 0.25 < 0.35 < 0.50      │
│                 │      /  \         │ Está na rampa DESCENDENTE        │
│                 │     /•   \        │ Fórmula: (c - x) / (c - b)       │
│                 │  a=0.0  b=0.25 c=0.5│ = (0.5 - 0.35) / (0.5 - 0.25) │
│                 │       ↑ 0.35      │ = 0.15 / 0.25 = 0.6              │
├─────────────────┼───────────────────┼──────────────────────────────────┤
│ médio           │ [0.25, 0.50, 0.75]│ μ = 0.4                          │
│                 │        /\         │ Por quê? 0.25 < 0.35 < 0.50      │
│                 │       /• \        │ Está na rampa ASCENDENTE         │
│                 │      /    \       │ Fórmula: (x - a) / (b - a)       │
│                 │   a=0.25  b=0.5 c=0.75│ = (0.35 - 0.25) / (0.5 - 0.25)│
│                 │       ↑ 0.35      │ = 0.10 / 0.25 = 0.4              │
├─────────────────┼───────────────────┼──────────────────────────────────┤
│ claro           │ [0.50, 0.75, 1.00]│ μ = 0.0                          │
│                 │          /\       │ Por quê? 0.35 < a (0.50)         │
│                 │         /  \      │ Está ANTES do início (a)!        │
│                 │   a=0.5  b=0.75 c=1.0│ Portanto: FORA do triângulo   │
├─────────────────┼───────────────────┼──────────────────────────────────┤
│ muito_claro     │ [0.75, 1.00, 1.00]│ μ = 0.0                          │
│                 │            /|     │ Por quê? 0.35 < a (0.75)         │
│                 │           / |     │ Está ANTES do início (a)!        │
│                 │     a=0.75  b,c=1.0│ Portanto: FORA do triângulo     │
└─────────────────┴───────────────────┴──────────────────────────────────┘

RESUMO - Como saber se está DENTRO ou FORA do triângulo:
  • Se x < a (início):  FORA (μ = 0)  ← antes do triângulo
  • Se a ≤ x ≤ b:       DENTRO (rampa ASC)
  • Se b < x ≤ c:       DENTRO (rampa DESC)
  • Se x > c (fim):     FORA (μ = 0)  ← depois do triângulo

OUTPUT (fuzzy):  [muito_escuro: 0.0, escuro: 0.6, médio: 0.4, claro: 0.0, muito_claro: 0.0]

💡 Interpretação: "Brilho 0.35 é 60% escuro e 40% médio"
💡 Nota: 0.6 + 0.4 = 1.0 na região de OVERLAP entre os triângulos!
```

---

## 🔬 IMPLEMENTAÇÃO NO CEREBRUM-ARTIS

### **ETAPA 1: EXTRAÇÃO DE VALORES CRISP** (Quantidades objetivas)

```python
import cv2
import numpy as np
from PIL import Image

def extract_crisp_features(image_path):
    """
    Extrai 7 features numéricas objetivas da imagem
    """
    # 1. Carregar imagem
    image = Image.open(image_path).convert('RGB')
    image_np = np.array(image)  # [height, width, 3]
    
    # 2. Converter para HSV (Hue, Saturation, Value)
    #    HSV é melhor que RGB para análise de cor
    hsv = cv2.cvtColor(image_np, cv2.COLOR_RGB2HSV)
    # hsv[:, :, 0] = matiz (cor: vermelho=0, verde=60, azul=120)
    # hsv[:, :, 1] = saturação (intensidade da cor)
    # hsv[:, :, 2] = valor (brilho)
    
    crisp_features = {}
    
    # ═══════════════════════════════════════════════════════════
    # FEATURE 1: BRIGHTNESS (Brilho médio)
    # ═══════════════════════════════════════════════════════════
    
    # Pega canal V (Value) do HSV
    brightness = hsv[:, :, 2].mean() / 255.0  # Normaliza [0, 255] → [0, 1]
    
    # Exemplo com Starry Night:
    # hsv[:, :, 2] = [[45, 52, 38, ...],   # pixels escuros (céu noturno)
    #                 [220, 198, 234, ...], # pixels claros (estrelas)
    #                 ...]
    # mean() = 89.5
    # 89.5 / 255 = 0.35 (pintura relativamente escura)
    
    crisp_features['brightness'] = brightness
    
    
    # ═══════════════════════════════════════════════════════════
    # FEATURE 2: COLOR_TEMPERATURE (Quente vs Frio)
    # ═══════════════════════════════════════════════════════════
    
    # Cores quentes: vermelho, laranja, amarelo (R alto)
    # Cores frias: azul, verde, roxo (B alto)
    
    r_mean = image_np[:, :, 0].mean()  # Canal Red
    b_mean = image_np[:, :, 2].mean()  # Canal Blue
    
    # Se R > B → quente (positivo)
    # Se B > R → frio (negativo)
    temp = (r_mean - b_mean) / 255.0  # [-1, 1]
    color_temperature = (temp + 1) / 2  # Normaliza para [0, 1]
    
    # Exemplo Starry Night:
    # r_mean = 120 (vermelho/amarelo das estrelas)
    # b_mean = 135 (azul dominante do céu)
    # temp = (120 - 135) / 255 = -0.059 (levemente frio)
    # color_temperature = 0.47 (quase neutro, puxando frio)
    
    crisp_features['color_temperature'] = color_temperature
    
    
    # ═══════════════════════════════════════════════════════════
    # FEATURE 3: SATURATION (Intensidade das cores)
    # ═══════════════════════════════════════════════════════════
    
    saturation = hsv[:, :, 1].mean() / 255.0
    
    # Saturação alta: cores vívidas, puras (Van Gogh, Matisse)
    # Saturação baixa: cores acinzentadas, pastéis (monocromático)
    
    # Exemplo Starry Night:
    # hsv[:, :, 1] = [[180, 200, 165, ...],  # céu azul saturado
    #                 [220, 245, 210, ...],  # estrelas amarelas vívidas
    #                 [90, 110, 95, ...]]    # vila menos saturada
    # mean() = 173.4
    # 173.4 / 255 = 0.68 (cores bastante vívidas)
    
    crisp_features['saturation'] = saturation
    
    
    # ═══════════════════════════════════════════════════════════
    # FEATURE 4: COLOR_HARMONY (Harmonia de cores)
    # ═══════════════════════════════════════════════════════════
    
    # Harmonia = quão "espalhadas" as cores estão no círculo cromático
    # Baixa variação de matiz = harmônico (cores próximas)
    # Alta variação de matiz = diverso (cores complementares)
    
    hue_std = hsv[:, :, 0].std()  # Desvio padrão das matizes
    
    # Normalizar usando função exponencial decrescente
    # std baixo → harmonia alta (1.0)
    # std alto → harmonia baixa (0.0)
    harmony = np.exp(-hue_std / 50.0)
    
    # Exemplo Starry Night:
    # Cores principais: azul (H≈120), amarelo (H≈30)
    # hue_std = 35.2 (diversidade moderada)
    # harmony = e^(-35.2/50) = e^(-0.704) = 0.49 (harmonia média)
    
    crisp_features['color_harmony'] = harmony
    
    
    # ═══════════════════════════════════════════════════════════
    # FEATURE 5: COMPLEXITY (Complexidade visual)
    # ═══════════════════════════════════════════════════════════
    
    # Usa gradientes de Sobel para detectar bordas/mudanças
    # Muitas bordas = complexo (pinceladas visíveis, detalhes)
    # Poucas bordas = simples (áreas lisas, minimalista)
    
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    
    # Filtro Sobel detecta mudanças bruscas de intensidade
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)  # Gradiente horizontal
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)  # Gradiente vertical
    
    # Combinar gradientes
    gradients = np.sqrt(sobel_x**2 + sobel_y**2)
    
    # Desvio padrão dos gradientes = medida de complexidade
    complexity = np.std(gradients) / 100.0  # Normalizar
    complexity = min(complexity, 1.0)  # Clipar em 1.0
    
    # Exemplo Starry Night:
    # Pinceladas swirling criam MUITAS bordas
    # std(gradients) = 72.3
    # complexity = 0.723 → 0.72 (alta complexidade)
    
    crisp_features['complexity'] = complexity
    
    
    # ═══════════════════════════════════════════════════════════
    # FEATURE 6: SYMMETRY (Simetria vertical)
    # ═══════════════════════════════════════════════════════════
    
    # Compara metade esquerda vs metade direita (espelhada)
    
    height, width, _ = image_np.shape
    
    left_half = image_np[:, :width//2]  # Metade esquerda
    right_half = image_np[:, width//2:]  # Metade direita
    right_half_flipped = np.fliplr(right_half)  # Espelha horizontalmente
    
    # Se metades forem iguais → diferença = 0 → simetria = 1.0
    # Se metades forem diferentes → diferença alta → simetria = 0.0
    
    # Garantir mesmas dimensões (se width for ímpar)
    min_width = min(left_half.shape[1], right_half_flipped.shape[1])
    left_half = left_half[:, :min_width]
    right_half_flipped = right_half_flipped[:, :min_width]
    
    # Diferença absoluta média
    diff = np.abs(left_half.astype(float) - right_half_flipped.astype(float)).mean()
    
    # Normalizar: diff=0 → symmetry=1, diff=255 → symmetry=0
    symmetry = 1.0 - (diff / 255.0)
    
    # Exemplo Starry Night:
    # Composição assimétrica (vila à direita, céu swirling não uniforme)
    # diff = 147.8
    # symmetry = 1 - (147.8/255) = 0.42 (baixa simetria)
    
    crisp_features['symmetry'] = symmetry
    
    
    # ═══════════════════════════════════════════════════════════
    # FEATURE 7: TEXTURE_ROUGHNESS (Rugosidade da textura)
    # ═══════════════════════════════════════════════════════════
    
    # Laplaciano detecta mudanças de segunda ordem (textura fina)
    # Textura lisa (sfumato): Laplaciano baixo
    # Textura rugosa (impasto, pinceladas): Laplaciano alto
    
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    
    # Desvio padrão = medida de rugosidade
    roughness = np.std(laplacian) / 50.0  # Normalizar
    roughness = min(roughness, 1.0)  # Clipar em 1.0
    
    # Exemplo Starry Night:
    # Impasto (tinta espessa, pinceladas visíveis)
    # std(laplacian) = 39.1
    # roughness = 0.78 (muito rugoso)
    
    crisp_features['texture_roughness'] = roughness
    
    
    return crisp_features

# ═══════════════════════════════════════════════════════════════
# RESULTADO FINAL (CRISP)
# ═══════════════════════════════════════════════════════════════

crisp = extract_crisp_features("starry_night.jpg")
print(crisp)

# {
#     'brightness': 0.35,           # Escuro (noite)
#     'color_temperature': 0.47,    # Neutro (azul+amarelo)
#     'saturation': 0.68,           # Saturado (cores vívidas)
#     'color_harmony': 0.49,        # Harmonia média
#     'complexity': 0.72,           # Complexo (swirls)
#     'symmetry': 0.42,             # Assimétrico
#     'texture_roughness': 0.78     # Rugoso (impasto)
# }
```

---

### **ETAPA 2: FUZZIFICAÇÃO** (Crisp → Fuzzy)

Agora vamos converter cada valor crisp em **graus de pertinência fuzzy**!

#### 🔺 Função de Pertinência Triangular (trimf)

```python
def trimf(x, abc):
    """
    Triangular Membership Function
    
    Parâmetros:
        x: valor de entrada (crisp)
        abc: [a, b, c] - pontos do triângulo
        
    Retorna:
        grau de pertinência [0, 1]
        
    Funcionamento:
        
          μ(x)
           │
         1 │     ╱╲
           │    ╱  ╲
           │   ╱    ╲
           │  ╱      ╲
         0 └─────────────
           a   b    c
           
        - Se x <= a: pertinência = 0
        - Se a < x < b: pertinência sobe linearmente (rampa)
        - Se x == b: pertinência = 1 (topo do triângulo)
        - Se b < x < c: pertinência desce linearmente
        - Se x >= c: pertinência = 0
    """
    a, b, c = abc
    
    if x <= a or x >= c:
        return 0.0
    elif a < x <= b:
        # Rampa ascendente
        return (x - a) / (b - a)
    elif b < x < c:
        # Rampa descendente
        return (c - x) / (c - b)
    else:
        return 0.0

# Exemplos:
trimf(0.35, [0.0, 0.2, 0.4])  # → 0.25 (na rampa descendente)
trimf(0.35, [0.2, 0.4, 0.6])  # → 0.75 (na rampa ascendente)
trimf(0.35, [0.6, 0.8, 1.0])  # → 0.0 (fora do triângulo)
```

#### 📊 Definição dos Conjuntos Fuzzy

Para **BRIGHTNESS**:

```python
# 5 termos linguísticos, cada um com um triângulo

BRIGHTNESS_SETS = {
    'muito_escuro': [0.0, 0.0, 0.2],  # Pico em 0.0
    'escuro':       [0.1, 0.3, 0.5],  # Pico em 0.3
    'medio':        [0.4, 0.6, 0.8],  # Pico em 0.6
    'claro':        [0.7, 0.9, 1.0],  # Pico em 0.9
    'muito_claro':  [0.9, 1.0, 1.0]   # Pico em 1.0
}

# Visualização:
"""
μ(x)
 1.0 ├╲   ╱╲   ╱╲   ╱╲   ╱
     │ ╲ ╱  ╲ ╱  ╲ ╱  ╲ ╱
     │  ╳    ╳    ╳    ╳
     │ ╱ ╲  ╱ ╲  ╱ ╲  ╱ ╲
 0.0 └──────────────────────
     0.0  0.2  0.4  0.6  0.8  1.0
     muito escuro médio claro muito
     escuro                   claro
"""

def fuzzify_brightness(crisp_value):
    """
    Fuzzifica o valor de brightness
    """
    fuzzy = {}
    
    for term, abc in BRIGHTNESS_SETS.items():
        fuzzy[term] = trimf(crisp_value, abc)
    
    return fuzzy

# Exemplo: Starry Night (brightness = 0.35)
fuzzy_brightness = fuzzify_brightness(0.35)
print(fuzzy_brightness)

# {
#     'muito_escuro': trimf(0.35, [0.0, 0.0, 0.2]) = 0.0
#     'escuro':       trimf(0.35, [0.1, 0.3, 0.5]) = 0.75  ← 75%!
#     'medio':        trimf(0.35, [0.4, 0.6, 0.8]) = 0.25  ← 25%!
#     'claro':        trimf(0.35, [0.7, 0.9, 1.0]) = 0.0
#     'muito_claro':  trimf(0.35, [0.9, 1.0, 1.0]) = 0.0
# }

# Interpretação: 
# "A pintura é 75% ESCURA e 25% MÉDIA (em termos de brilho)"
```

#### 🎨 Cálculo Detalhado com Starry Night

```python
# ═══════════════════════════════════════════════════════════
# BRIGHTNESS = 0.35
# ═══════════════════════════════════════════════════════════

# TERMO 1: muito_escuro [0.0, 0.0, 0.2]
#
#   μ
#   1├╲
#    │ ╲
#    │  ╲
#    │   ╲
#   0└────────
#    0.0  0.2
#         ↑
#      x=0.35 está FORA (x > c)
#
muito_escuro = trimf(0.35, [0.0, 0.0, 0.2])
# x=0.35 >= c=0.2 → return 0.0


# TERMO 2: escuro [0.1, 0.3, 0.5]
#
#   μ
#   1│  ╱╲
#    │ ╱  ╲
#    │╱    ╲
#   0└───────────
#    0.1 0.3 0.5
#          ↑
#       x=0.35 está na RAMPA DESCENDENTE
#
escuro = trimf(0.35, [0.1, 0.3, 0.5])
# b < x < c (0.3 < 0.35 < 0.5)
# return (c - x) / (c - b)
#      = (0.5 - 0.35) / (0.5 - 0.3)
#      = 0.15 / 0.2
#      = 0.75  ✅


# TERMO 3: medio [0.4, 0.6, 0.8]
#
#   μ
#   1│    ╱╲
#    │   ╱  ╲
#    │  ╱    ╲
#   0└──────────
#    0.4 0.6 0.8
#     ↑
#  x=0.35 está na RAMPA ASCENDENTE (mas antes de 'a')
#
medio = trimf(0.35, [0.4, 0.6, 0.8])
# x=0.35 <= a=0.4 → return 0.0
# 
# ESPERA! x < a, então:
# Vamos recalcular...
# x=0.35, a=0.4, b=0.6, c=0.8
# a < x? NÃO (0.4 não é < 0.35)
# x <= a? SIM (0.35 <= 0.4)
# return 0.0  ✅
#
# CORREÇÃO: Olhando o código do projeto real...
# Na verdade existe OVERLAP! Vamos ver:

# Recalculando com definições corretas do código:
BRIGHTNESS_SETS_REAL = {
    'muito_escuro': [0.0, 0.0, 0.25],   # Overlap com escuro
    'escuro':       [0.0, 0.25, 0.5],   # Overlap dos dois lados
    'medio':        [0.25, 0.5, 0.75],  # Overlap dos dois lados
    'claro':        [0.5, 0.75, 1.0],   # Overlap dos dois lados
    'muito_claro':  [0.75, 1.0, 1.0]    # Overlap com claro
}

# Recalculando com x=0.35:

# muito_escuro [0.0, 0.0, 0.25]:
#   b < x < c (0.0 < 0.35, mas 0.35 > 0.25?)
#   x >= c → return 0.0

# escuro [0.0, 0.25, 0.5]:
#   b < x < c (0.25 < 0.35 < 0.5)
#   return (0.5 - 0.35) / (0.5 - 0.25) = 0.15 / 0.25 = 0.6

# medio [0.25, 0.5, 0.75]:
#   a < x <= b (0.25 < 0.35 <= 0.5)
#   return (0.35 - 0.25) / (0.5 - 0.25) = 0.10 / 0.25 = 0.4

# claro [0.5, 0.75, 1.0]:
#   x <= a → return 0.0

# muito_claro [0.75, 1.0, 1.0]:
#   x <= a → return 0.0

# RESULTADO FUZZY para brightness=0.35:
# [0.0, 0.6, 0.4, 0.0, 0.0]
#  muito escuro médio claro muito_claro
#  escuro                           

# Interpretação:
# "60% pertence a ESCURO, 40% pertence a MÉDIO"
```

#### 🔁 Fuzzificação Completa das 7 Features

```python
def fuzzify_all_features(crisp_features):
    """
    Fuzzifica TODAS as 7 features
    """
    # Definir conjuntos fuzzy (valores do cerebrum_artis/fuzzy/variables.py)
    
    FUZZY_SETS = {
        'brightness': {
            'muito_baixo': [0.0, 0.0, 0.25],
            'baixo':       [0.0, 0.25, 0.5],
            'medio':       [0.25, 0.5, 0.75],
            'alto':        [0.5, 0.75, 1.0],
            'muito_alto':  [0.75, 1.0, 1.0]
        },
        'color_temperature': {
            'muito_frio':  [0.0, 0.0, 0.25],
            'frio':        [0.0, 0.25, 0.5],
            'neutro':      [0.25, 0.5, 0.75],
            'quente':      [0.5, 0.75, 1.0],
            'muito_quente':[0.75, 1.0, 1.0]
        },
        'saturation': {
            'muito_baixa': [0.0, 0.0, 0.25],
            'baixa':       [0.0, 0.25, 0.5],
            'media':       [0.25, 0.5, 0.75],
            'alta':        [0.5, 0.75, 1.0],
            'muito_alta':  [0.75, 1.0, 1.0]
        },
        # ... (mesmo padrão para todas as 7 features)
    }
    
    fuzzy_results = {}
    
    for feature_name, crisp_value in crisp_features.items():
        fuzzy_results[feature_name] = {}
        
        for term, abc in FUZZY_SETS[feature_name].items():
            fuzzy_results[feature_name][term] = trimf(crisp_value, abc)
    
    return fuzzy_results


# ═══════════════════════════════════════════════════════════
# APLICAR NO STARRY NIGHT
# ═══════════════════════════════════════════════════════════

crisp = {
    'brightness': 0.35,
    'color_temperature': 0.47,
    'saturation': 0.68,
    'color_harmony': 0.49,
    'complexity': 0.72,
    'symmetry': 0.42,
    'texture_roughness': 0.78
}

fuzzy = fuzzify_all_features(crisp)

print(fuzzy)

# {
#     'brightness': {
#         'muito_baixo': 0.0,
#         'baixo': 0.6,     ← 60% escuro
#         'medio': 0.4,     ← 40% médio
#         'alto': 0.0,
#         'muito_alto': 0.0
#     },
#     'color_temperature': {
#         'muito_frio': 0.0,
#         'frio': 0.12,     ← 12% frio
#         'neutro': 0.88,   ← 88% neutro (azul+amarelo equilibrado)
#         'quente': 0.0,
#         'muito_quente': 0.0
#     },
#     'saturation': {
#         'muito_baixa': 0.0,
#         'baixa': 0.0,
#         'media': 0.28,    ← 28% média
#         'alta': 0.72,     ← 72% alta saturação
#         'muito_alta': 0.0
#     },
#     'color_harmony': {
#         'muito_baixa': 0.0,
#         'baixa': 0.04,
#         'media': 0.96,    ← 96% harmonia média
#         'alta': 0.0,
#         'muito_alta': 0.0
#     },
#     'complexity': {
#         'muito_baixa': 0.0,
#         'baixa': 0.0,
#         'media': 0.12,
#         'alta': 0.88,     ← 88% complexo!
#         'muito_alta': 0.0
#     },
#     'symmetry': {
#         'muito_baixa': 0.0,
#         'baixa': 0.68,    ← 68% baixa simetria (assimétrico)
#         'media': 0.32,
#         'alta': 0.0,
#         'muito_alta': 0.0
#     },
#     'texture_roughness': {
#         'muito_baixa': 0.0,
#         'baixa': 0.0,
#         'media': 0.0,
#         'alta': 0.12,
#         'muito_alta': 0.88  ← 88% muito rugoso! (impasto)
#     }
# }

# Total: 7 features × 5 termos = 35 valores fuzzy
```

---

### **ETAPA 3: NO V2 - USAR APENAS VALORES CRISP** ⚠️

**PLOT TWIST**: O V2 NÃO USA a fuzzificação completa!

No V2, apenas os **7 valores crisp** são passados para o modelo:

```python
# cerebrum_artis/models/v2_fuzzy_features/train_v2_cached.py

class ArtEmisCachedFuzzyDataset(Dataset):
    def __getitem__(self, idx):
        # ...
        
        # Carrega fuzzy features PRÉ-COMPUTADAS
        fuzzy_feats = self.fuzzy_cache[painting_id]
        
        # fuzzy_feats é um array [7] com valores CRISP:
        # [brightness, color_temp, saturation, harmony, complexity, symmetry, roughness]
        
        return {
            'image': image_tensor,
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'fuzzy_features': fuzzy_feats,  # [7] valores crisp
            'label': label
        }

# Modelo recebe:
visual_feats = [B, 2048]     # ResNet
text_feats = [B, 768]        # RoBERTa
fuzzy_feats = [B, 7]         # 7 valores crisp (NÃO fuzzificados!)

combined = torch.cat([visual_feats, text_feats, fuzzy_feats], dim=1)
# [B, 2823]
```

**Por que não usar os 35 valores fuzzy?**

1. **Simplicidade**: 7 dims vs 35 dims (menos parâmetros)
2. **Performance**: V2 já atingiu 70.63% com apenas crisp
3. **Interpretabilidade**: Valores crisp são mais diretos

---

### **ETAPA 4: NO V3/V3.1 - USAR REGRAS FUZZY COMPLETAS** 🎯

Aqui sim entra a **lógica fuzzy COMPLETA**!

#### 🧠 Sistema de Inferência Fuzzy (FIS)

```python
# cerebrum_artis/fuzzy/system.py

class FuzzyInferenceSystem:
    """
    Sistema completo de inferência fuzzy (Mamdani)
    """
    
    def __init__(self):
        # Carregar regras fuzzy
        self.rules = self.load_rules()
        # 18 regras do tipo:
        # "SE brightness é baixo E saturation é alta 
        #  ENTÃO mood é dramático"
    
    def infer(self, crisp_inputs):
        """
        Pipeline completo:
        Crisp → Fuzzificação → Regras → Defuzzificação → Crisp
        """
        
        # 1. FUZZIFICAÇÃO
        fuzzy_inputs = self.fuzzify(crisp_inputs)
        
        # 2. AVALIAÇÃO DAS REGRAS
        fuzzy_outputs = self.evaluate_rules(fuzzy_inputs)
        
        # 3. AGREGAÇÃO
        aggregated = self.aggregate(fuzzy_outputs)
        
        # 4. DEFUZZIFICAÇÃO
        crisp_output = self.defuzzify(aggregated)
        
        return crisp_output
```

#### 📜 Exemplo de Regra Fuzzy

```python
# REGRA 1: Detectar mood dramático

regra_1 = {
    'antecedent': [
        ('brightness', 'baixo'),       # SE brilho é baixo
        ('saturation', 'alta')         # E saturação é alta
    ],
    'consequent': ('mood', 'dramatico'),  # ENTÃO mood é dramático
    'operator': 'AND'
}

# Avaliação com Starry Night:

# 1. Pegar graus de pertinência
brightness_baixo = fuzzy['brightness']['baixo']  # 0.6
saturation_alta = fuzzy['saturation']['alta']     # 0.72

# 2. Operador AND = mínimo
firing_strength = min(0.6, 0.72)  # 0.6

# 3. Aplicar no consequente
# "mood é dramático" com grau 0.6
fuzzy_output['mood']['dramatico'] = 0.6
```

#### ⚖️ Defuzzificação (Fuzzy → Crisp)

Converter grau de pertinência fuzzy de volta para número:

```python
def defuzzify_centroid(fuzzy_output):
    """
    Método do centróide (centro de massa)
    """
    # fuzzy_output['mood'] = {
    #     'calmo': 0.0,
    #     'neutro': 0.2,
    #     'dramatico': 0.6,
    #     'intenso': 0.3,
    #     'caotico': 0.0
    # }
    
    # Valores de referência para cada termo
    centroids = {
        'calmo': 0.1,
        'neutro': 0.3,
        'dramatico': 0.6,
        'intenso': 0.8,
        'caotico': 0.95
    }
    
    # Calcular média ponderada
    numerator = sum(fuzzy_output[term] * centroids[term] 
                    for term in fuzzy_output)
    denominator = sum(fuzzy_output.values())
    
    crisp_mood = numerator / denominator
    
    # (0.0×0.1 + 0.2×0.3 + 0.6×0.6 + 0.3×0.8 + 0.0×0.95) / (0.0+0.2+0.6+0.3+0.0)
    # = (0 + 0.06 + 0.36 + 0.24 + 0) / 1.1
    # = 0.66 / 1.1
    # = 0.6  (mood dramático!)
    
    return crisp_mood
```

---

## 🔄 COMPARAÇÃO: V2 vs V3

### V2 (Fuzzy Features Simples):
```
Imagem → Calcular 7 valores crisp → [7] → Concatenar com ResNet+RoBERTa → MLP
         (brightness=0.35, sat=0.68, ...)
```

### V3 (Fuzzy Inference System Completo):
```
Imagem → Calcular 7 crisp → Fuzzificar → 18 Regras → Defuzzificar → [7 + outputs]
         (0.35, 0.68, ...)   (35 fuzzy)   (mood, energy,  (mood=0.6,
                                           tension, ...)   energy=0.7)
                                                          
         → Concatenar com ResNet+RoBERTa → Gating Adaptativo → MLP
```

**Diferença**:
- V2: Features objetivas diretas
- V3: Raciocínio fuzzy com regras (mais "inteligente", mas não necessariamente melhor performance)

---

## 📊 VISUALIZAÇÃO COMPLETA DO FLUXO

```
┌─────────────────────────────────────────────────────────────────┐
│                     FUZZY EXTRACTOR (V2)                        │
└─────────────────────────────────────────────────────────────────┘

INPUT: starry_night.jpg (imagem PIL)

    │
    ├─→ cv2.cvtColor(RGB → HSV)
    │   hsv[:,:,0] = matiz   [0-180]
    │   hsv[:,:,1] = saturação [0-255]
    │   hsv[:,:,2] = brilho  [0-255]
    │
    ├─→ FEATURE 1: BRIGHTNESS
    │   hsv[:,:,2].mean() / 255 = 89.5/255 = 0.35
    │
    ├─→ FEATURE 2: COLOR_TEMPERATURE
    │   (R_mean - B_mean)/255 normalizado = 0.47
    │
    ├─→ FEATURE 3: SATURATION
    │   hsv[:,:,1].mean() / 255 = 173.4/255 = 0.68
    │
    ├─→ FEATURE 4: COLOR_HARMONY
    │   exp(-std(hue)/50) = exp(-35.2/50) = 0.49
    │
    ├─→ FEATURE 5: COMPLEXITY
    │   std(Sobel gradients)/100 = 72.3/100 = 0.72
    │
    ├─→ FEATURE 6: SYMMETRY
    │   1 - (abs(left-right)/255) = 1 - 0.58 = 0.42
    │
    └─→ FEATURE 7: TEXTURE_ROUGHNESS
        std(Laplacian)/50 = 39.1/50 = 0.78

OUTPUT: torch.tensor([0.35, 0.47, 0.68, 0.49, 0.72, 0.42, 0.78])
        Shape: [7]

    │
    ▼

┌──────────────────────────────────────────────┐
│   CONCATENAÇÃO (no modelo V2)                │
├──────────────────────────────────────────────┤
│ visual_feats [2048]                          │
│ text_feats [768]                             │
│ fuzzy_feats [7]    ← AQUI!                   │
│ ↓                                            │
│ combined [2823]                              │
└──────────────────────────────────────────────┘
```

---

## 🎓 RESUMO CONCEITUAL

### Lógica Fuzzy em 3 passos:

1. **FUZZIFICAÇÃO**: Número preciso → Graus de pertinência
   - `0.35` → `{baixo: 0.6, médio: 0.4}`

2. **INFERÊNCIA**: Aplicar regras linguísticas
   - "SE baixo E alta saturação ENTÃO dramático"

3. **DEFUZZIFICAÇÃO**: Graus de pertinência → Número preciso
   - `{dramático: 0.6, intenso: 0.3}` → `0.6`

### No Cerebrum-Artis:

- **V2**: Usa apenas STEP 1 (features crisp, pula fuzzificação)
- **V3/V3.1**: Usa os 3 steps completos (sistema fuzzy Mamdani)

### Por que Fuzzy Logic?

✅ Modela incerteza ("mais ou menos escuro")  
✅ Raciocínio humano ("SE...ENTÃO")  
✅ Interpretável (regras explícitas)  
✅ Lida com transições suaves (não abrupto como booleano)

---

## 📝 GLOSSÁRIO FINAL

| Termo | Significado |
|-------|-------------|
| **Crisp** | Valor numérico preciso (ex: 0.35) |
| **Fuzzy** | Conjunto de graus de pertinência (ex: {baixo: 0.6, médio: 0.4}) |
| **Fuzzificação** | Conversão crisp → fuzzy |
| **Defuzzificação** | Conversão fuzzy → crisp |
| **Membership Function** | Função que calcula grau de pertinência |
| **trimf** | Função triangular (rampa sobe + rampa desce) |
| **Variável Linguística** | Variável com termos verbais (ex: "escuro", "claro") |
| **Regra Fuzzy** | "SE...ENTÃO" com termos fuzzy |
| **Operador AND** | min(A, B) em lógica fuzzy |
| **Operador OR** | max(A, B) em lógica fuzzy |
| **Centroide** | Método de defuzzificação (centro de massa) |

---

🎉 **Agora você entende TUDO sobre o Fuzzy Extractor!**
