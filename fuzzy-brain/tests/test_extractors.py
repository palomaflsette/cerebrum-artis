"""
Testes unitários para o Visual Feature Extractor.

Este arquivo testa se todas as features estão sendo extraídas corretamente
e se os valores estão no range esperado [0, 1].
"""

import pytest
import numpy as np
import cv2
from pathlib import Path
import tempfile
import os

from fuzzy_brain.extractors.visual import VisualFeatureExtractor, extract_features_from_path


class TestVisualFeatureExtractor:
    """Testes para o VisualFeatureExtractor."""
    
    @pytest.fixture
    def extractor(self):
        """Cria uma instância do extrator."""
        return VisualFeatureExtractor()
    
    @pytest.fixture
    def temp_image_path(self):
        """Cria uma imagem temporária para testes."""
        # Cria imagem sintética RGB (100x100, cor uniforme azul)
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        img[:, :] = [100, 149, 237]  # Cornflower blue
        
        # Salva temporariamente
        temp_file = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
        cv2.imwrite(temp_file.name, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        
        yield temp_file.name
        
        # Cleanup
        os.unlink(temp_file.name)
    
    def test_extractor_initialization(self, extractor):
        """Testa se o extrator inicializa corretamente."""
        assert extractor is not None
    
    def test_extract_all_returns_dict(self, extractor, temp_image_path):
        """Testa se extract_all retorna um dicionário."""
        features = extractor.extract_all(temp_image_path)
        assert isinstance(features, dict)
    
    def test_extract_all_has_all_features(self, extractor, temp_image_path):
        """Testa se todas as features esperadas estão presentes."""
        features = extractor.extract_all(temp_image_path)
        
        expected_keys = {
            'brightness',
            'color_temperature',
            'saturation',
            'color_harmony',
            'complexity',
            'symmetry',
            'texture_roughness'
        }
        
        assert set(features.keys()) == expected_keys
    
    def test_all_values_in_valid_range(self, extractor, temp_image_path):
        """Testa se todos os valores estão no range [0, 1]."""
        features = extractor.extract_all(temp_image_path)
        
        for key, value in features.items():
            assert 0.0 <= value <= 1.0, \
                f"Feature '{key}' fora do range: {value}"
    
    def test_brightness_extreme_dark(self, extractor):
        """Testa brilho em imagem totalmente preta."""
        # Cria imagem preta
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        temp_file = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
        cv2.imwrite(temp_file.name, img)
        
        try:
            features = extractor.extract_all(temp_file.name)
            # Preto deve ter brilho muito baixo (próximo de 0)
            assert features['brightness'] < 0.1
        finally:
            os.unlink(temp_file.name)
    
    def test_brightness_extreme_white(self, extractor):
        """Testa brilho em imagem totalmente branca."""
        # Cria imagem branca
        img = np.ones((100, 100, 3), dtype=np.uint8) * 255
        temp_file = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
        cv2.imwrite(temp_file.name, img)
        
        try:
            features = extractor.extract_all(temp_file.name)
            # Branco deve ter brilho muito alto (próximo de 1)
            assert features['brightness'] > 0.9
        finally:
            os.unlink(temp_file.name)
    
    def test_color_temperature_warm(self, extractor):
        """Testa temperatura de cor quente (vermelho)."""
        # Imagem vermelha
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        img[:, :] = [255, 0, 0]  # Vermelho puro (RGB)
        
        temp_file = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
        cv2.imwrite(temp_file.name, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        
        try:
            features = extractor.extract_all(temp_file.name)
            # Vermelho é cor quente, deve ter temperatura > 0.5
            assert features['color_temperature'] > 0.5
        finally:
            os.unlink(temp_file.name)
    
    def test_color_temperature_cool(self, extractor):
        """Testa temperatura de cor fria (azul)."""
        # Imagem azul
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        img[:, :] = [0, 0, 255]  # Azul puro (RGB)
        
        temp_file = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
        cv2.imwrite(temp_file.name, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        
        try:
            features = extractor.extract_all(temp_file.name)
            # Azul é cor fria, deve ter temperatura < 0.5
            assert features['color_temperature'] < 0.5
        finally:
            os.unlink(temp_file.name)
    
    def test_saturation_gray(self, extractor):
        """Testa saturação em imagem em tons de cinza."""
        # Imagem cinza (sem saturação)
        img = np.ones((100, 100, 3), dtype=np.uint8) * 128
        
        temp_file = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
        cv2.imwrite(temp_file.name, img)
        
        try:
            features = extractor.extract_all(temp_file.name)
            # Cinza não tem saturação
            assert features['saturation'] < 0.1
        finally:
            os.unlink(temp_file.name)
    
    def test_symmetry_perfect(self, extractor):
        """Testa simetria em imagem perfeitamente simétrica."""
        # Cria imagem simétrica (gradiente espelhado)
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        for i in range(50):
            img[:, i] = i * 5
            img[:, 99-i] = i * 5
        
        temp_file = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
        cv2.imwrite(temp_file.name, img)
        
        try:
            features = extractor.extract_all(temp_file.name)
            # Deve ter alta simetria
            assert features['symmetry'] > 0.7
        finally:
            os.unlink(temp_file.name)
    
    def test_extract_features_from_path_helper(self, temp_image_path):
        """Testa a função auxiliar extract_features_from_path."""
        features = extract_features_from_path(temp_image_path)
        assert isinstance(features, dict)
        assert len(features) == 7
    
    def test_invalid_image_path(self, extractor):
        """Testa comportamento com caminho inválido."""
        with pytest.raises(FileNotFoundError):
            extractor.extract_all("caminho/inexistente.jpg")


if __name__ == "__main__":
    # Permite rodar testes diretamente
    pytest.main([__file__, "-v"])
