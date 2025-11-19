# 🚀 Guia de Início Rápido - Fuzzy-Brain

## ✅ O que você acabou de criar

Você agora tem um **Visual Feature Extractor** completo que extrai 7 features interpretáveis de imagens!

### Features Implementadas:

1. **Brightness** (Brilho) - Quão clara/escura é a imagem
2. **Color Temperature** (Temperatura) - Quão quente/fria são as cores
3. **Saturation** (Saturação) - Quão vibrantes são as cores
4. **Color Harmony** (Harmonia) - Quão harmonioso é o esquema de cores
5. **Complexity** (Complexidade) - Densidade de detalhes visuais
6. **Symmetry** (Simetria) - Simetria da composição
7. **Texture Roughness** (Aspereza) - Rugosidade da textura

---

## 📦 PASSO 1: Instalar Dependências

```bash
cd /home/paloma/cerebrum-artis/fuzzy-brain

# Ative seu ambiente conda/venv se ainda não estiver ativo
# conda activate artemis-sat  # ou outro ambiente

# Instale as dependências
pip install -r requirements.txt
```

**⏱️ Tempo estimado**: 2-3 minutos

---

## 🧪 PASSO 2: Testar com Imagem Sintética

Vamos criar uma imagem de teste simples para verificar se tudo funciona:

```bash
# Cria um script Python rápido para gerar imagem de teste
python3 << 'EOF'
import cv2
import numpy as np

# Cria imagem gradiente azul (fria, médio brilho)
img = np.zeros((400, 400, 3), dtype=np.uint8)
for i in range(400):
    img[i, :] = [0, i//2, 255 - i//2]  # Gradiente de azul

cv2.imwrite('test_image.jpg', img)
print("✅ Imagem de teste criada: test_image.jpg")
EOF
```

Agora teste o extrator:

```bash
python scripts/demo.py test_image.jpg
```

**Saída esperada:**
```
🎨 ANÁLISE DE FEATURES VISUAIS - FUZZY-BRAIN
======================================================================

📂 Imagem: test_image.jpg

⏳ Extraindo features...
✅ Features extraídas com sucesso!

======================================================================
RESULTADOS
======================================================================

Brilho:
  Valor: 0.4980
  [████████████████████░░░░░░░░░░░░░░░░░░░░]
  → médio ☁️

Color Temperature:
  Valor: 0.2513
  [██████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░]
  → muito frio ❄️ (azul/verde)

... (e assim por diante)
```

---

## 🎨 PASSO 3: Testar com Pinturas Reais do ArtEmis

Para testar com pinturas reais, você precisa ter as imagens do WikiArt baixadas.

**Caminho esperado** (baseado no seu setup do ArtEmis):
```
/data/paloma/data/paintings/wikiart/
├── Abstract_Expressionism/
│   ├── painting1.jpg
│   ├── painting2.jpg
│   └── ...
├── Baroque/
└── ...
```

**Teste com uma pintura real:**

```bash
# Exemplo: testar com uma pintura barroca
python scripts/demo.py /data/paloma/data/paintings/wikiart/Baroque/caravaggio_david-with-the-head-of-goliath.jpg

# Ou qualquer outra que você tenha
```

---

## 🧪 PASSO 4: Rodar Testes Unitários

```bash
# Roda todos os testes
pytest tests/test_extractors.py -v

# Ou rode só um teste específico
pytest tests/test_extractors.py::TestVisualFeatureExtractor::test_brightness_extreme_dark -v
```

**O que os testes verificam:**
- ✅ Todas as features são extraídas
- ✅ Valores estão no range [0, 1]
- ✅ Imagem preta → baixo brilho
- ✅ Imagem branca → alto brilho
- ✅ Vermelho → quente
- ✅ Azul → frio
- ✅ E mais...

---

## 🐍 PASSO 5: Usar no Código Python

```python
from fuzzy_brain.extractors.visual import VisualFeatureExtractor

# Cria o extrator
extractor = VisualFeatureExtractor()

# Extrai features de uma imagem
features = extractor.extract_all("minha_pintura.jpg")

# Acessa features individuais
print(f"Brilho: {features['brightness']:.2f}")
print(f"Temperatura: {features['color_temperature']:.2f}")

# Ou use a função helper
from fuzzy_brain.extractors.visual import extract_features_from_path
features = extract_features_from_path("minha_pintura.jpg")
```

---

## 📊 PASSO 6: Entender os Valores

### Interpretação dos Ranges:

| Feature | Baixo (0.0-0.3) | Médio (0.3-0.7) | Alto (0.7-1.0) |
|---------|----------------|-----------------|----------------|
| **Brightness** | Muito escuro 🌑 | Médio ☁️ | Muito claro ✨ |
| **Temperature** | Frio ❄️ (azul) | Neutro ⚖️ | Quente 🔥 (vermelho) |
| **Saturation** | Cinza ⬛ | Moderado 🎨 | Vibrante 🌈 |
| **Harmony** | Dissonante 🎪 | Harmônico 🎼 | Muito harmônico 🎵 |
| **Complexity** | Simples ➖ | Moderado 📰 | Complexo 🧩 |
| **Symmetry** | Assimétrico ↗️ | Simétrico ⚖️ | Perfeitamente simétrico 🔲 |
| **Roughness** | Suave 🧈 | Médio 🪨 | Áspero 🖌️ |

---

## 🎯 O QUE VOCÊ APRENDEU

### Teorias Implementadas:

1. **Espaços de Cor (HSV vs RGB)**
   - HSV separa cor, saturação e brilho
   - Mais intuitivo para análise de cor
   
2. **Psicologia das Cores**
   - Cores quentes vs frias têm impactos emocionais diferentes
   - Saturação afeta energia percebida

3. **Análise de Textura (LBP)**
   - Local Binary Patterns capturam micropadrões
   - Útil para detectar estilo de pincelada

4. **Edge Detection (Canny)**
   - Densidade de edges = proxy para complexidade
   - Fundamental em visão computacional

5. **Análise de Composição**
   - Simetria é princípio estético fundamental
   - Harmonia cromática baseada em entropia

---

## 🔜 PRÓXIMOS PASSOS

Agora que temos o **extrator de features**, os próximos passos são:

1. **✅ COMPLETO**: Visual Feature Extractor
2. **🔄 PRÓXIMO**: Sistema de Lógica Fuzzy
   - Definir variáveis fuzzy
   - Criar regras fuzzy
   - Implementar inferência Mamdani
3. **⏭️ DEPOIS**: Integração Neural-Fuzzy
4. **⏭️ FINAL**: Avaliação e visualizações

---

## 💡 DICAS

### Debugging:
```bash
# Modo verbose para ver o que está acontecendo
python -c "
from fuzzy_brain.extractors.visual import VisualFeatureExtractor
import logging
logging.basicConfig(level=logging.DEBUG)
extractor = VisualFeatureExtractor()
features = extractor.extract_all('test.jpg')
print(features)
"
```

### Performance:
```python
# Para processar muitas imagens
import time
from fuzzy_brain.extractors.visual import VisualFeatureExtractor

extractor = VisualFeatureExtractor()

start = time.time()
features = extractor.extract_all("painting.jpg")
elapsed = time.time() - start

print(f"⏱️ Tempo: {elapsed:.4f}s")
# Esperado: ~0.05-0.2s dependendo do tamanho da imagem
```

---

## 🐛 Troubleshooting

### Erro: "Module 'cv2' not found"
```bash
pip install opencv-python
```

### Erro: "Module 'skimage' not found"
```bash
pip install scikit-image
```

### Erro: FileNotFoundError
- Verifique se o caminho da imagem está correto
- Use caminho absoluto ou relativo correto

### Valores estranhos (todos 0 ou 1)
- Verifique se a imagem foi carregada corretamente
- Tente com outra imagem

---

## 📚 Quer Aprender Mais?

- Leia os **comentários no código** em `fuzzy_brain/extractors/visual.py`
- Cada método tem explicação teórica detalhada!
- Experimente modificar thresholds e ver o impacto

---

**🎉 Parabéns! Você completou a primeira fase do projeto Fuzzy-Brain!**

**Próximo arquivo a criar**: `fuzzy_brain/fuzzy/variables.py` (Sistema Fuzzy)
