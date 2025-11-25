"""
============================================================================
VISUAL FEATURE EXTRACTOR
============================================================================
Extrai features visuais INTERPRETÁVEIS de imagens para alimentar o sistema fuzzy.

TEORIA:
-------
Features visuais são propriedades mensuráveis da imagem que podem ser
extraídas DIRETAMENTE dos pixels, sem necessidade de deep learning.

Estas features são a BASE para o raciocínio fuzzy porque:
1. São interpretáveis (humanos entendem "brilho" e "saturação")
2. Relacionam-se com emoções (cores escuras → tristeza)
3. Validadas por psicologia das cores

FEATURES IMPLEMENTADAS:
-----------------------
1. BRIGHTNESS (Brilho): Quão clara/escura é a imagem
   - Teoria: Cores escuras associadas a emoções negativas (tristeza, medo)
   - Cálculo: Média do canal V (Value) no espaço HSV
   - Range: [0, 1] onde 0=preto, 1=branco

2. COLOR TEMPERATURE (Temperatura de Cor): Quão quente/fria é a paleta
   - Teoria: Cores quentes (vermelho/amarelo) → energia, raiva
              Cores frias (azul/verde) → calma, tristeza
   - Cálculo: Ratio de cores quentes vs frias
   - Range: [0, 1] onde 0=frio, 1=quente

3. SATURATION (Saturação): Quão vibrantes são as cores
   - Teoria: Alta saturação → energia, excitação
              Baixa saturação → melancolia, nostalgia
   - Cálculo: Média do canal S (Saturation) no espaço HSV
   - Range: [0, 1] onde 0=cinza, 1=vibrante

4. COLOR HARMONY (Harmonia de Cores): Quão harmonioso é o esquema de cores
   - Teoria: Harmonia → paz, admiração
              Dissonância → tensão, desconforto
   - Cálculo: Baseado em entropia da distribuição de matizes
   - Range: [0, 1] onde 0=dissonante, 1=harmônico

5. COMPLEXITY (Complexidade Visual): Densidade de informação na imagem
   - Teoria: Alta complexidade → admiração ou confusão
              Baixa complexidade → calma, monotonia
   - Cálculo: Densidade de edges (Canny edge detection)
   - Range: [0, 1]

6. SYMMETRY (Simetria): Quão simétrica é a composição
   - Teoria: Simetria → ordem, beleza, admiração
              Assimetria → dinamismo, tensão
   - Cálculo: Correlação entre imagem e sua versão espelhada
   - Range: [0, 1] onde 0=assimétrico, 1=simétrico

7. TEXTURE ROUGHNESS (Aspereza de Textura): Rugosidade da pincelada
   - Teoria: Textura áspera → energia, raiva
              Textura suave → calma, contentamento
   - Cálculo: Variância de Local Binary Patterns (LBP)
   - Range: [0, 1]

REFERÊNCIAS CIENTÍFICAS:
------------------------
- Valdez & Mehrabian (1994): Effects of color on emotions
- Palmer & Schloss (2010): An ecological valence theory
- Itten (1970): The Art of Color
============================================================================
"""

import numpy as np
from typing import Dict, Tuple
from scipy.stats import entropy
from skimage import feature
from pathlib import Path
from PIL import Image
import warnings

# Flag global para fallback PIL
# Força PIL se cv2 der segfault
USE_PIL_FALLBACK = True

try:
    if not USE_PIL_FALLBACK:
        import cv2
        # Testa se cv2 funciona
        _test_img = np.zeros((10, 10, 3), dtype=np.uint8)
        _test_gray = cv2.cvtColor(_test_img, cv2.COLOR_BGR2GRAY)
except Exception as e:
    USE_PIL_FALLBACK = True
    warnings.warn(
        f"cv2 não está funcionando corretamente ({e})! "
        "Usando fallback PIL (algumas features podem ser aproximadas)."
    )


class VisualFeatureExtractor:
    """
    Extrator de features visuais interpretáveis de imagens.
    
    Esta classe implementa métodos para extrair propriedades visuais básicas
    de imagens (cor, textura, composição) que serão usadas como INPUT para
    o sistema de lógica fuzzy.
    
    Attributes:
        None (stateless - processa cada imagem independentemente)
    
    Examples:
        >>> extractor = VisualFeatureExtractor()
        >>> features = extractor.extract_all("path/to/painting.jpg")
        >>> print(features['brightness'])  # 0.25 (escuro)
        >>> print(features['color_temperature'])  # 0.3 (frio)
    """
    
    def __init__(self):
        """Inicializa o extrator (stateless, sem parâmetros)."""
        pass
    
    def extract_all(self, image_path: str) -> Dict[str, float]:
        """
        Extrai TODAS as features visuais de uma imagem.
        
        Este é o método principal que você vai usar! Ele chama todos os
        métodos individuais e retorna um dicionário completo.
        
        Args:
            image_path: Caminho para a imagem (str ou Path)
        
        Returns:
            Dict com todas as features normalizadas em [0, 1]:
            {
                'brightness': float,          # Brilho médio
                'color_temperature': float,   # Quão quente/fria
                'saturation': float,          # Vivacidade das cores
                'color_harmony': float,       # Harmonia cromática
                'complexity': float,          # Densidade de informação
                'symmetry': float,            # Simetria da composição
                'texture_roughness': float    # Aspereza da textura
            }
        
        Raises:
            FileNotFoundError: Se a imagem não existe
            ValueError: Se a imagem está corrompida
        """
        # Carrega a imagem usando PIL (fallback seguro)
        if USE_PIL_FALLBACK:
            return self._extract_all_pil(image_path)
        
        # Tenta cv2 primeiro (mais rápido)
        try:
            img_bgr = cv2.imread(str(image_path))
            if img_bgr is None:
                # cv2 falhou, tenta PIL
                return self._extract_all_pil(image_path)
            
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
            img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
            
        except Exception as e:
            # cv2 deu erro, fallback para PIL
            warnings.warn(f"cv2 falhou ({e}), usando PIL fallback")
            return self._extract_all_pil(image_path)
        
        # Extrai todas as features
        features = {
            'brightness': self._compute_brightness(img_hsv),
            'color_temperature': self._compute_color_temperature(img_rgb),
            'saturation': self._compute_saturation(img_hsv),
            'color_harmony': self._compute_color_harmony(img_hsv),
            'complexity': self._compute_complexity(img_gray),
            'symmetry': self._compute_symmetry(img_rgb),
            'texture_roughness': self._compute_texture_roughness(img_gray)
        }
        
        return features
    
    def _extract_all_pil(self, image_path: str) -> Dict[str, float]:
        """
        Versão alternativa usando PIL (quando cv2 não funciona).
        
        Args:
            image_path: Caminho para a imagem
        
        Returns:
            Dicionário com 7 features [0, 1]
        """
        # Carrega com PIL
        img_pil = Image.open(image_path).convert('RGB')
        img_rgb = np.array(img_pil, dtype=np.float32) / 255.0  # [0, 1]
        
        # Converte RGB → HSV manualmente
        img_hsv = self._rgb_to_hsv(img_rgb)
        
        # Grayscale
        img_gray = np.dot(img_rgb[..., :3], [0.299, 0.587, 0.114])
        
        # Extrai features (métodos adaptados)
        features = {
            'brightness': float(np.mean(img_hsv[:, :, 2])),  # V channel
            'color_temperature': self._compute_color_temperature(img_rgb),
            'saturation': float(np.mean(img_hsv[:, :, 1])),  # S channel
            'color_harmony': self._compute_color_harmony_pil(img_hsv),
            'complexity': self._compute_complexity(img_gray),
            'symmetry': self._compute_symmetry(img_rgb),
            'texture_roughness': self._compute_texture_roughness(img_gray)
        }
        
        return features
    
    @staticmethod
    def _rgb_to_hsv(rgb: np.ndarray) -> np.ndarray:
        """
        Converte RGB [0,1] para HSV [0,1].
        
        Args:
            rgb: Array (H, W, 3) em [0, 1]
        
        Returns:
            hsv: Array (H, W, 3) em [0, 1]
        """
        r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
        
        maxc = np.maximum(np.maximum(r, g), b)
        minc = np.minimum(np.minimum(r, g), b)
        v = maxc
        
        deltac = maxc - minc
        s = deltac / (maxc + 1e-10)
        s[maxc == 0] = 0
        
        # Hue
        h = np.zeros_like(maxc)
        
        rc = (maxc - r) / (deltac + 1e-10)
        gc = (maxc - g) / (deltac + 1e-10)
        bc = (maxc - b) / (deltac + 1e-10)
        
        mask_r = (r == maxc)
        mask_g = (g == maxc)
        mask_b = (b == maxc)
        
        h[mask_r] = bc[mask_r] - gc[mask_r]
        h[mask_g] = 2.0 + rc[mask_g] - bc[mask_g]
        h[mask_b] = 4.0 + gc[mask_b] - rc[mask_b]
        
        h = (h / 6.0) % 1.0
        h[deltac == 0] = 0
        
        return np.stack([h, s, v], axis=-1)
    
    def _compute_color_harmony_pil(self, img_hsv: np.ndarray) -> float:
        """
        Versão simplificada de harmony para PIL (sem cv2.calcHist).
        
        Args:
            img_hsv: HSV em [0, 1]
        
        Returns:
            Harmonia [0, 1]
        """
        # Histogram manual do canal Hue
        h_channel = img_hsv[:, :, 2].flatten()
        hist, _ = np.histogram(h_channel, bins=12, range=(0, 1))
        hist = hist / (hist.sum() + 1e-10)
        
        # Entropia (quanto menor, mais harmônico)
        h_entropy = entropy(hist + 1e-10)
        max_entropy = np.log(12)  # 12 bins
        
        # Inverte: alta entropia = baixa harmonia
        harmony = 1.0 - (h_entropy / max_entropy)
        
        return float(np.clip(harmony, 0.0, 1.0))
    
    # ========================================================================
    # MÉTODOS PRIVADOS - Cada um calcula UMA feature específica
    # ========================================================================
    
    def _compute_brightness(self, img_hsv: np.ndarray) -> float:
        """
        Calcula o BRILHO médio da imagem.
        
        TEORIA:
        -------
        O espaço de cor HSV separa:
        - H (Hue): Matiz (qual cor)
        - S (Saturation): Saturação (quão pura)
        - V (Value): Brilho (quão clara)
        
        O canal V representa diretamente o brilho, independente da cor!
        
        PSICOLOGIA:
        -----------
        - Baixo brilho (0.0-0.3): Associado a tristeza, medo, mistério
        - Médio brilho (0.3-0.7): Neutro, equilíbrio
        - Alto brilho (0.7-1.0): Alegria, pureza, esperança
        
        Args:
            img_hsv: Imagem no espaço HSV (height, width, 3)
        
        Returns:
            Brilho normalizado [0, 1]
        """
        # Canal V está na posição 2 do HSV
        v_channel = img_hsv[:, :, 2]
        
        # OpenCV usa V em [0, 255], normalizamos para [0, 1]
        brightness = np.mean(v_channel) / 255.0
        
        return float(brightness)
    
    def _compute_color_temperature(self, img_rgb: np.ndarray) -> float:
        """
        Calcula a TEMPERATURA DE COR (quão quente/fria é a paleta).
        
        TEORIA:
        -------
        Cores quentes: Vermelho, Laranja, Amarelo (wavelength longa)
        Cores frias: Azul, Verde, Violeta (wavelength curta)
        
        PSICOLOGIA:
        -----------
        - Cores quentes: Energia, paixão, raiva, excitação
        - Cores frias: Calma, tristeza, serenidade, medo
        
        CÁLCULO:
        --------
        Usamos um ratio simplificado:
        - Warm score = média(R) + 0.5 * média(G)  # Vermelho e amarelo
        - Cool score = média(B) + 0.5 * média(G)  # Azul e verde
        - Temperature = warm / (warm + cool)
        
        Args:
            img_rgb: Imagem no espaço RGB (height, width, 3)
        
        Returns:
            Temperatura normalizada [0, 1] onde 0=frio, 1=quente
        """
        r, g, b = img_rgb[:, :, 0], img_rgb[:, :, 1], img_rgb[:, :, 2]
        
        # Warm: vermelho + parte do verde (amarelo = R+G)
        warm_score = np.mean(r) + 0.5 * np.mean(g)
        
        # Cool: azul + parte do verde (ciano = G+B)
        cool_score = np.mean(b) + 0.5 * np.mean(g)
        
        # Normaliza para [0, 1]
        # Adiciona epsilon (1e-6) para evitar divisão por zero
        temperature = warm_score / (warm_score + cool_score + 1e-6)
        
        return float(temperature)
    
    def _compute_saturation(self, img_hsv: np.ndarray) -> float:
        """
        Calcula a SATURAÇÃO média das cores.
        
        TEORIA:
        -------
        Saturação mede quão "pura" é a cor:
        - Alta saturação (1.0): Cor pura, vibrante
        - Baixa saturação (0.0): Cor acinzentada, desbotada
        
        PSICOLOGIA:
        -----------
        - Alta saturação: Energia, alegria, excitação
        - Baixa saturação: Nostalgia, melancolia, sobriedade
        
        Args:
            img_hsv: Imagem no espaço HSV
        
        Returns:
            Saturação normalizada [0, 1]
        """
        # Canal S está na posição 1 do HSV
        s_channel = img_hsv[:, :, 1]
        
        # OpenCV usa S em [0, 255], normalizamos
        saturation = np.mean(s_channel) / 255.0
        
        return float(saturation)
    
    def _compute_color_harmony(self, img_hsv: np.ndarray) -> float:
        """
        Calcula a HARMONIA DE CORES usando entropia.
        
        TEORIA:
        -------
        Harmonia de cores é um conceito da teoria da cor que mede quão
        "agradável" é a combinação de cores em uma imagem.
        
        Usamos ENTROPIA da distribuição de matizes (Hue):
        - Baixa entropia: Poucas cores dominantes → HARMONIOSO
        - Alta entropia: Muitas cores diferentes → DISSONANTE
        
        PSICOLOGIA:
        -----------
        - Alta harmonia: Paz, contentamento, admiração
        - Baixa harmonia: Tensão, desconforto, caos
        
        CÁLCULO:
        --------
        1. Cria histograma de matizes (12 bins = roda de cores)
        2. Calcula entropia: H = -Σ(p * log(p))
        3. Inverte: harmony = 1 - (entropy / max_entropy)
        
        Args:
            img_hsv: Imagem no espaço HSV
        
        Returns:
            Harmonia normalizada [0, 1] onde 0=dissonante, 1=harmônico
        """
        # Canal H (Hue/Matiz) está na posição 0
        h_channel = img_hsv[:, :, 0]
        
        # OpenCV usa H em [0, 180] (não 360!)
        # Cria histograma com 12 bins (roda de cores tradicional)
        hist, _ = np.histogram(h_channel, bins=12, range=(0, 180))
        
        # Normaliza para probabilidades
        hist = hist / (np.sum(hist) + 1e-6)
        
        # Calcula entropia de Shannon
        color_entropy = entropy(hist + 1e-6)  # +epsilon evita log(0)
        
        # Entropia máxima = log(12) (distribuição uniforme)
        max_entropy = np.log(12)
        
        # Inverte: queremos que alta harmonia = baixa entropia
        harmony = 1.0 - (color_entropy / max_entropy)
        
        return float(harmony)
    
    def _compute_complexity(self, img_gray: np.ndarray) -> float:
        """
        Calcula a COMPLEXIDADE VISUAL usando edge detection.
        
        TEORIA:
        -------
        Complexidade visual é a quantidade de "informação" em uma imagem.
        Imagens com muitos detalhes/edges têm alta complexidade.
        
        Usamos Canny Edge Detection:
        - Detecta mudanças abruptas de intensidade (edges)
        - Densidade de edges = proxy para complexidade
        
        PSICOLOGIA:
        -----------
        - Alta complexidade: Admiração (pintura detalhada) ou confusão
        - Baixa complexidade: Calma, minimalismo, tédio
        
        Args:
            img_gray: Imagem em escala de cinza
        
        Returns:
            Complexidade normalizada [0, 1]
        """
        if USE_PIL_FALLBACK:
            # Versão PIL: usa gradiente de Sobel
            from skimage.filters import sobel
            edges = sobel(img_gray)
            # Threshold para binarizar
            edge_mask = edges > 0.1
            complexity = np.sum(edge_mask) / edge_mask.size
        else:
            # Canny Edge Detection com thresholds padrão
            # threshold1=100, threshold2=200
            edges = cv2.Canny(img_gray, 100, 200)
            
            # Calcula proporção de pixels que são edges
            # edges > 0 cria máscara booleana
            complexity = np.sum(edges > 0) / edges.size
        
        return float(complexity)
    
    def _compute_symmetry(self, img_rgb: np.ndarray) -> float:
        """
        Calcula a SIMETRIA vertical da composição.
        
        TEORIA:
        -------
        Simetria é uma propriedade fundamental da estética.
        Testamos simetria VERTICAL (mais comum em pinturas).
        
        Método: Comparar lado esquerdo com lado direito espelhado.
        
        PSICOLOGIA:
        -----------
        - Alta simetria: Ordem, beleza, admiração, harmonia
        - Baixa simetria: Dinamismo, tensão, interesse visual
        
        CÁLCULO:
        --------
        1. Divide imagem ao meio verticalmente
        2. Espelha lado direito
        3. Calcula diferença absoluta média
        4. Converte para score de similaridade
        
        Args:
            img_rgb: Imagem RGB
        
        Returns:
            Simetria normalizada [0, 1] onde 0=assimétrico, 1=simétrico
        """
        h, w = img_rgb.shape[:2]
        
        # Divide ao meio
        left_half = img_rgb[:, :w//2]
        right_half = img_rgb[:, w//2:]
        
        # Espelha o lado direito (flip horizontal)
        right_half_flipped = np.fliplr(right_half)
        
        # Garante mesma largura (se w for ímpar)
        min_width = min(left_half.shape[1], right_half_flipped.shape[1])
        left_half = left_half[:, :min_width]
        right_half_flipped = right_half_flipped[:, :min_width]
        
        # Calcula diferença absoluta média (Mean Absolute Error)
        diff = np.abs(left_half.astype(float) - right_half_flipped.astype(float))
        mae = np.mean(diff)
        
        # Converte diferença em similaridade
        # MAE está em [0, 255], então symmetry em [0, 1]
        symmetry = 1.0 - (mae / 255.0)
        
        return float(symmetry)
    
    def _compute_texture_roughness(self, img_gray: np.ndarray) -> float:
        """
        Calcula a ASPEREZA DE TEXTURA usando Local Binary Patterns (LBP).
        
        TEORIA:
        -------
        LBP (Local Binary Pattern) é um descritor de textura que captura
        padrões locais de variação de intensidade.
        
        Alta variância de LBP = textura áspera/irregular
        Baixa variância de LBP = textura suave/uniforme
        
        PSICOLOGIA (aplicado a pinturas):
        ----------------------------------
        - Textura áspera: Pinceladas visíveis, energia, expressionismo
        - Textura suave: Sfumato, calma, realismo
        
        CÁLCULO:
        --------
        1. Calcula LBP para cada pixel (compara com 8 vizinhos)
        2. Mede variância do mapa LBP
        3. Normaliza empiricamente
        
        Args:
            img_gray: Imagem em escala de cinza
        
        Returns:
            Aspereza normalizada [0, 1] onde 0=suave, 1=áspero
        """
        # LBP com raio=1 e 8 pontos (padrão)
        # method='uniform' reduz dimensionalidade
        lbp = feature.local_binary_pattern(
            img_gray, 
            P=8,           # 8 vizinhos
            R=1,           # Raio de 1 pixel
            method='uniform'
        )
        
        # Calcula variância do mapa LBP
        roughness = np.std(lbp) / 10.0  # Normalização empírica
        
        # Clipa para [0, 1] (valores muito altos são raros)
        roughness = np.clip(roughness, 0.0, 1.0)
        
        return float(roughness)


# ============================================================================
# FUNÇÕES AUXILIARES PARA FACILITAR USO
# ============================================================================

def extract_features_from_path(image_path: str) -> Dict[str, float]:
    """
    Função de conveniência para extrair features de uma imagem.
    
    Args:
        image_path: Caminho para a imagem
    
    Returns:
        Dict com todas as features
    
    Example:
        >>> features = extract_features_from_path("painting.jpg")
        >>> print(f"Brilho: {features['brightness']:.2f}")
    """
    extractor = VisualFeatureExtractor()
    return extractor.extract_all(image_path)


if __name__ == "__main__":
    # Código de teste quando executado diretamente
    import sys
    
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        print(f"\n🎨 Analisando: {image_path}\n")
        
        features = extract_features_from_path(image_path)
        
        print("=" * 60)
        print("FEATURES VISUAIS EXTRAÍDAS")
        print("=" * 60)
        for name, value in features.items():
            print(f"{name:20s}: {value:.4f}")
        print("=" * 60)
    else:
        print("Uso: python visual.py <caminho_da_imagem>")
