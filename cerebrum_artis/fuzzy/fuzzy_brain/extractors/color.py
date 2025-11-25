"""
Color Feature Extractors

Implements the 7 fuzzy color-based feature extractors:
1. WarmthExtractor - Extracts warmth (warm tones: reds, oranges, yellows)
2. ColdnessExtractor - Extracts coldness (cool tones: blues, greens, purples)
3. SaturationExtractor - Extracts color saturation
4. MutednessExtractor - Extracts mutedness (desaturated colors)
5. BrightnessExtractor - Extracts brightness/luminosity
6. DarknessExtractor - Extracts darkness (dark tones)
7. HarmonyExtractor - Extracts chromatic harmony
"""

import numpy as np
from typing import Union
from PIL import Image


class ColorExtractor:
    """Base class for color feature extractors."""
    
    def __init__(self):
        self.name = self.__class__.__name__
    
    def extract(self, image: Union[np.ndarray, Image.Image]) -> float:
        """
        Extract feature from image.
        
        Args:
            image: Input image (numpy array or PIL Image)
        
        Returns:
            Feature value (float between 0 and 1)
        """
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        if image.dtype == np.uint8:
            image = image.astype(np.float32) / 255.0
        
        return self._compute(image)
    
    def _compute(self, image: np.ndarray) -> float:
        """Implement in subclass."""
        raise NotImplementedError


class WarmthExtractor(ColorExtractor):
    """Extracts warmth from image (warm colors: reds, oranges, yellows)."""
    
    def _compute(self, image: np.ndarray) -> float:
        """
        Compute warmth based on red/yellow dominance.
        
        Warmth increases with:
        - Higher red channel values
        - Red > Blue (warm vs cool)
        """
        r, g, b = image[:, :, 0], image[:, :, 1], image[:, :, 2]
        
        # Warmth: red dominance and red > blue
        warmth = (r + g/2) / 2  # Red and yellow tones
        warm_vs_cool = np.maximum(0, r - b)  # Warm (red) vs cool (blue)
        
        warmth_score = (warmth.mean() + warm_vs_cool.mean()) / 2
        return float(np.clip(warmth_score, 0, 1))


class ColdnessExtractor(ColorExtractor):
    """Extracts coldness from image (cool colors: blues, greens, purples)."""
    
    def _compute(self, image: np.ndarray) -> float:
        """
        Compute coldness based on blue/cyan dominance.
        
        Coldness increases with:
        - Higher blue channel values
        - Blue > Red (cool vs warm)
        """
        r, g, b = image[:, :, 0], image[:, :, 1], image[:, :, 2]
        
        # Coldness: blue dominance and blue > red
        coldness = (b + g/2) / 2  # Blue and cyan tones
        cool_vs_warm = np.maximum(0, b - r)  # Cool (blue) vs warm (red)
        
        coldness_score = (coldness.mean() + cool_vs_warm.mean()) / 2
        return float(np.clip(coldness_score, 0, 1))


class SaturationExtractor(ColorExtractor):
    """Extracts color saturation from image."""
    
    def _compute(self, image: np.ndarray) -> float:
        """
        Compute saturation as color intensity variance.
        
        High saturation = vivid colors (large difference between max and min channels)
        Low saturation = grayish colors (small difference)
        """
        max_rgb = image.max(axis=2)
        min_rgb = image.min(axis=2)
        
        # Saturation = (max - min) / max (avoid division by zero)
        saturation = np.where(max_rgb > 0, (max_rgb - min_rgb) / max_rgb, 0)
        
        return float(saturation.mean())


class MutednessExtractor(ColorExtractor):
    """Extracts mutedness (desaturation) from image."""
    
    def _compute(self, image: np.ndarray) -> float:
        """
        Compute mutedness as inverse of saturation.
        
        High mutedness = desaturated, grayish colors
        Low mutedness = vivid, saturated colors
        """
        max_rgb = image.max(axis=2)
        min_rgb = image.min(axis=2)
        
        # Mutedness = 1 - saturation
        saturation = np.where(max_rgb > 0, (max_rgb - min_rgb) / max_rgb, 0)
        mutedness = 1 - saturation
        
        return float(mutedness.mean())


class BrightnessExtractor(ColorExtractor):
    """Extracts brightness/luminosity from image."""
    
    def _compute(self, image: np.ndarray) -> float:
        """
        Compute brightness as average luminosity.
        
        Uses standard luminosity formula: 0.299*R + 0.587*G + 0.114*B
        """
        r, g, b = image[:, :, 0], image[:, :, 1], image[:, :, 2]
        
        # Standard luminosity weights (perceived brightness)
        luminosity = 0.299 * r + 0.587 * g + 0.114 * b
        
        return float(luminosity.mean())


class DarknessExtractor(ColorExtractor):
    """Extracts darkness (inverse of brightness) from image."""
    
    def _compute(self, image: np.ndarray) -> float:
        """
        Compute darkness as inverse of brightness.
        
        High darkness = dark image
        Low darkness = bright image
        """
        r, g, b = image[:, :, 0], image[:, :, 1], image[:, :, 2]
        
        # Darkness = 1 - brightness
        luminosity = 0.299 * r + 0.587 * g + 0.114 * b
        darkness = 1 - luminosity
        
        return float(darkness.mean())


class HarmonyExtractor(ColorExtractor):
    """Extracts chromatic harmony from image."""
    
    def _compute(self, image: np.ndarray) -> float:
        """
        Compute harmony based on color coherence.
        
        High harmony = low variance in hue (colors are similar/related)
        Low harmony = high variance in hue (colors are disparate)
        
        Uses RGB to HSV conversion for hue analysis.
        """
        r, g, b = image[:, :, 0], image[:, :, 1], image[:, :, 2]
        
        # Convert to HSV (approximate)
        max_rgb = np.maximum(np.maximum(r, g), b)
        min_rgb = np.minimum(np.minimum(r, g), b)
        delta = max_rgb - min_rgb
        
        # Compute hue (0-360 degrees, normalized to 0-1)
        hue = np.zeros_like(r)
        
        # Red is max
        mask = (max_rgb == r) & (delta > 0)
        hue[mask] = ((g[mask] - b[mask]) / delta[mask]) % 6
        
        # Green is max
        mask = (max_rgb == g) & (delta > 0)
        hue[mask] = ((b[mask] - r[mask]) / delta[mask]) + 2
        
        # Blue is max
        mask = (max_rgb == b) & (delta > 0)
        hue[mask] = ((r[mask] - g[mask]) / delta[mask]) + 4
        
        hue = hue / 6.0  # Normalize to 0-1
        
        # Harmony = 1 - variance in hue (low variance = high harmony)
        # Only consider saturated pixels (avoid grayscale noise)
        saturation = np.where(max_rgb > 0, delta / max_rgb, 0)
        saturated_pixels = saturation > 0.1
        
        if saturated_pixels.sum() > 0:
            hue_variance = hue[saturated_pixels].var()
            harmony = 1 - np.clip(hue_variance * 4, 0, 1)  # Scale variance
        else:
            harmony = 0.5  # Neutral harmony for grayscale images
        
        return float(harmony)


# Convenience function to extract all features at once
def extract_all_features(image: Union[np.ndarray, Image.Image]) -> dict:
    """
    Extract all 7 color features from an image.
    
    Args:
        image: Input image
    
    Returns:
        Dictionary with feature names and values
    """
    extractors = {
        'warmth': WarmthExtractor(),
        'coldness': ColdnessExtractor(),
        'saturation': SaturationExtractor(),
        'mutedness': MutednessExtractor(),
        'brightness': BrightnessExtractor(),
        'darkness': DarknessExtractor(),
        'harmony': HarmonyExtractor(),
    }
    
    return {name: extractor.extract(image) for name, extractor in extractors.items()}
