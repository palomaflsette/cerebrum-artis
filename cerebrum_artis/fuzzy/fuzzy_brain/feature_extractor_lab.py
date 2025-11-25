"""
Feature Extractor usando LAB Color Space (Perceptualmente Uniforme)

Extrai 7 features visuais otimizadas para análise afetiva:
1. Brightness (L*)
2. Color Temperature (a*)
3. Saturation (Chroma C*)
4. Color Harmony (entropia no plano a*b*)
5. Complexity (gradientes LAB)
6. Symmetry (simetria em L*)
7. Texture Roughness (variância de L*)
"""

import cv2
import numpy as np
from typing import Dict, Union
from pathlib import Path
from PIL import Image


class LABFeatureExtractor:
    """
    Extrai features visuais usando o espaço LAB perceptualmente uniforme.
    
    Vantagens do LAB sobre RGB:
    - L* = Luminosidade perceptual (0-100)
    - a* = Verde←→Vermelho (-127 a +127)
    - b* = Azul←→Amarelo (-127 a +127)
    - Distâncias euclidianas = diferenças perceptuais reais
    """
    
    def __init__(self, target_size=(256, 256)):
        """
        Inicializa extrator.
        
        Args:
            target_size: Tamanho para redimensionar imagens (H, W)
        """
        self.target_size = target_size
    
    def extract(self, image: Union[str, Path, np.ndarray]) -> Dict[str, float]:
        """
        Extrai 7 features de uma imagem.
        
        Args:
            image: Caminho da imagem ou array numpy RGB
        
        Returns:
            Dict com 7 features normalizadas [0, 1]
        """
        # Carrega imagem
        if isinstance(image, (str, Path)):
            img = Image.open(image).convert('RGB')
            img = np.array(img)
        else:
            img = image
        
        # Redimensiona
        if img.shape[:2] != self.target_size:
            img = cv2.resize(img, self.target_size[::-1])  # cv2 usa (W, H)
        
        # Converte RGB [0, 255] → LAB usando cv2
        # cv2 espera BGR, então convertemos RGB → BGR → LAB
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
        
        # Normaliza para ranges corretos:
        # cv2: L [0, 255], a/b [0, 255]
        # CIE: L [0, 100], a/b [-127, 127]
        lab[:,:,0] = lab[:,:,0] * (100.0 / 255.0)  # L: [0, 255] → [0, 100]
        lab[:,:,1] = lab[:,:,1] - 128.0            # a: [0, 255] → [-128, 127]
        lab[:,:,2] = lab[:,:,2] - 128.0            # b: [0, 255] → [-128, 127]
        
        L, a, b = lab[:,:,0], lab[:,:,1], lab[:,:,2]
        
        # Extrai features
        features = {
            'brightness': self._brightness(L),
            'color_temperature': self._color_temperature(a),
            'saturation': self._saturation(a, b),
            'color_harmony': self._color_harmony(a, b),
            'complexity': self._complexity(L, a, b),
            'symmetry': self._symmetry(L),
            'texture_roughness': self._texture_roughness(L)
        }
        
        return features
    
    def _brightness(self, L: np.ndarray) -> float:
        """
        Luminosidade média perceptual.
        
        L* é linearmente relacionado à percepção de brilho.
        
        Args:
            L: Canal L* (0-100)
        
        Returns:
            Brightness normalizado [0, 1]
        """
        return float(L.mean() / 100.0)
    
    def _color_temperature(self, a: np.ndarray) -> float:
        """
        Temperatura de cor baseada no eixo a*.
        
        a* > 0 = vermelho (quente)
        a* < 0 = verde (frio)
        
        Args:
            a: Canal a* (-127 a +127)
        
        Returns:
            Temperature normalizada [0, 1]
            0 = frio (verde), 1 = quente (vermelho)
        """
        a_mean = a.mean()
        # Normaliza [-127, 127] → [0, 1]
        temp = (a_mean + 127.0) / 254.0
        return float(np.clip(temp, 0, 1))
    
    def _saturation(self, a: np.ndarray, b: np.ndarray) -> float:
        """
        Saturação cromática (Chroma C*).
        
        C* = √(a*² + b*²) = distância do eixo acromático (cinza)
        
        Args:
            a: Canal a*
            b: Canal b*
        
        Returns:
            Saturation normalizada [0, 1]
        """
        chroma = np.sqrt(a**2 + b**2)
        # C* máximo teórico ≈ 181 (cores saturadas)
        # Na prática, artworks raramente excedem 100
        saturation = chroma.mean() / 100.0
        return float(np.clip(saturation, 0, 1))
    
    def _color_harmony(self, a: np.ndarray, b: np.ndarray) -> float:
        """
        Harmonia cromática via entropia no círculo a*b*.
        
        Cores harmônicas tendem a distribuir-se uniformemente no
        círculo cromático (complementares, tríades, etc.)
        
        Args:
            a: Canal a*
            b: Canal b*
        
        Returns:
            Harmony [0, 1]
            1 = cores uniformemente distribuídas (harmônico)
            0 = cores concentradas (monocromático/dissonante)
        """
        # Calcula ângulo hue no plano a*b*
        hue = np.arctan2(b, a)  # [-π, π]
        hue_deg = np.degrees(hue) % 360  # [0, 360]
        
        # Histogram circular (12 bins = 30° cada)
        # 12 bins captura: complementares, tríades, tetrádicas
        hist, _ = np.histogram(hue_deg.flatten(), bins=12, range=(0, 360))
        hist_norm = hist / (hist.sum() + 1e-10)
        
        # Entropia de Shannon
        # Alta entropia = distribuição uniforme = harmônico
        entropy = -np.sum(hist_norm * np.log(hist_norm + 1e-10))
        max_entropy = np.log(12)  # Distribuição uniforme perfeita
        
        harmony = entropy / max_entropy
        return float(np.clip(harmony, 0, 1))
    
    def _complexity(self, L: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
        """
        Complexidade visual via gradientes em LAB.
        
        Imagens complexas têm alta variação espacial em todos os canais.
        
        Args:
            L, a, b: Canais LAB
        
        Returns:
            Complexity normalizada [0, 1]
        """
        # Gradientes Sobel em cada canal
        grad_L = np.abs(cv2.Sobel(L, cv2.CV_64F, 1, 0, ksize=3)) + \
                 np.abs(cv2.Sobel(L, cv2.CV_64F, 0, 1, ksize=3))
        
        grad_a = np.abs(cv2.Sobel(a, cv2.CV_64F, 1, 0, ksize=3)) + \
                 np.abs(cv2.Sobel(a, cv2.CV_64F, 0, 1, ksize=3))
        
        grad_b = np.abs(cv2.Sobel(b, cv2.CV_64F, 1, 0, ksize=3)) + \
                 np.abs(cv2.Sobel(b, cv2.CV_64F, 0, 1, ksize=3))
        
        # Média dos gradientes (normalizado empiricamente)
        # Valores típicos: 10-50 para L, 5-30 para a/b
        complexity = (grad_L.mean() / 100.0 + 
                     grad_a.mean() / 60.0 + 
                     grad_b.mean() / 60.0) / 3.0
        
        return float(np.clip(complexity, 0, 1))
    
    def _symmetry(self, L: np.ndarray) -> float:
        """
        Simetria horizontal baseada em luminosidade.
        
        Analisa similaridade entre metades esquerda/direita.
        
        Args:
            L: Canal L*
        
        Returns:
            Symmetry [0, 1]
            1 = perfeitamente simétrico
            0 = completamente assimétrico
        """
        h, w = L.shape
        
        # Divide horizontalmente
        left_half = L[:, :w//2]
        right_half = np.fliplr(L[:, w//2:])
        
        # Garante mesmo tamanho (w ímpar)
        min_w = min(left_half.shape[1], right_half.shape[1])
        left_half = left_half[:, :min_w]
        right_half = right_half[:, :min_w]
        
        # Diferença média absoluta
        diff = np.abs(left_half - right_half).mean()
        
        # Normaliza (L* max = 100)
        symmetry = 1.0 - (diff / 100.0)
        return float(np.clip(symmetry, 0, 1))
    
    def _texture_roughness(self, L: np.ndarray) -> float:
        """
        Rugosidade de textura via variância local de luminosidade.
        
        Texturas rugosas têm alta variação local em L*.
        
        Args:
            L: Canal L*
        
        Returns:
            Roughness [0, 1]
            1 = muito rugoso (alta variação)
            0 = suave (baixa variação)
        """
        # Desvio padrão de L* (medida global)
        # Valores típicos: 5-30
        L_std = L.std()
        
        # Normaliza
        roughness = L_std / 50.0  # Std máximo empírico
        return float(np.clip(roughness, 0, 1))
    
    def extract_batch(self, images: list) -> list:
        """
        Extrai features de múltiplas imagens.
        
        Args:
            images: Lista de caminhos ou arrays
        
        Returns:
            Lista de dicts com features
        """
        return [self.extract(img) for img in images]
    
    def __repr__(self):
        return f"LABFeatureExtractor(target_size={self.target_size})"


# ============================================================================
# TESTE RÁPIDO
# ============================================================================

if __name__ == "__main__":
    import sys
    from pathlib import Path
    
    print("="*70)
    print("TESTE: LAB Feature Extractor")
    print("="*70)
    
    # Cria imagem de teste (gradiente vermelho→verde)
    test_img = np.zeros((256, 256, 3), dtype=np.uint8)
    for i in range(256):
        test_img[:, i, 0] = 255 - i  # R decresce
        test_img[:, i, 1] = i         # G cresce
    
    extractor = LABFeatureExtractor()
    features = extractor.extract(test_img)
    
    print("\n📊 Features extraídas (imagem de teste):")
    print("-" * 70)
    for name, value in features.items():
        bar = "█" * int(value * 30)
        print(f"{name:20s}: {value:.4f} {bar}")
    
    print("\n" + "="*70)
    print("✅ LAB Feature Extractor funcionando!")
    print("="*70)
