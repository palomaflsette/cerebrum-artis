"""
Comparação RGB vs LAB Feature Extraction

Compara as 7 features extraídas usando RGB vs LAB em imagens de teste.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List
import matplotlib.pyplot as plt
import seaborn as sns

from fuzzy_brain.extractors.visual import VisualFeatureExtractor
from fuzzy_brain.feature_extractor_lab import LABFeatureExtractor


class RGBFeatureExtractorWrapper:
    """Wrapper para VisualFeatureExtractor usar mesma interface que LAB."""
    def __init__(self):
        self.extractor = VisualFeatureExtractor()
    
    def extract(self, image):
        """Extrai features aceitando array numpy."""
        import tempfile
        from PIL import Image
        import numpy as np
        
        # Se for array, salva temporariamente
        if isinstance(image, np.ndarray):
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as f:
                img_pil = Image.fromarray(image)
                img_pil.save(f.name)
                features = self.extractor.extract_all(f.name)
            import os
            os.unlink(f.name)
            return features
        else:
            return self.extractor.extract_all(image)


class RGBvsLABComparison:
    """
    Compara extração de features RGB vs LAB.
    """
    
    def __init__(self):
        self.rgb_extractor = RGBFeatureExtractorWrapper()
        self.lab_extractor = LABFeatureExtractor()
        self.feature_names = [
            'brightness',
            'color_temperature',
            'saturation',
            'color_harmony',
            'complexity',
            'symmetry',
            'texture_roughness'
        ]
    
    def compare_single_image(self, image) -> Dict:
        """
        Compara features RGB vs LAB em uma imagem.
        
        Args:
            image: Caminho ou array numpy
        
        Returns:
            Dict com features RGB e LAB
        """
        rgb_features = self.rgb_extractor.extract(image)
        lab_features = self.lab_extractor.extract(image)
        
        comparison = {
            'rgb': rgb_features,
            'lab': lab_features,
            'diff': {}
        }
        
        # Calcula diferenças
        for name in self.feature_names:
            comparison['diff'][name] = abs(
                rgb_features[name] - lab_features[name]
            )
        
        return comparison
    
    def compare_batch(self, images: List, labels: List[str] = None) -> pd.DataFrame:
        """
        Compara features em múltiplas imagens.
        
        Args:
            images: Lista de imagens
            labels: Labels opcionais para cada imagem
        
        Returns:
            DataFrame com comparação
        """
        results = []
        
        for i, img in enumerate(images):
            comp = self.compare_single_image(img)
            
            for name in self.feature_names:
                results.append({
                    'image_id': labels[i] if labels else f'img_{i}',
                    'feature': name,
                    'rgb_value': comp['rgb'][name],
                    'lab_value': comp['lab'][name],
                    'abs_diff': comp['diff'][name],
                    'rel_diff': comp['diff'][name] / (comp['rgb'][name] + 1e-10)
                })
        
        return pd.DataFrame(results)
    
    def visualize_comparison(self, df: pd.DataFrame, save_path: str = None):
        """
        Visualiza comparação RGB vs LAB.
        
        Args:
            df: DataFrame do compare_batch
            save_path: Caminho para salvar figura
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('RGB vs LAB Feature Extraction Comparison', 
                     fontsize=16, fontweight='bold')
        
        # 1. Scatter: RGB vs LAB values
        ax = axes[0, 0]
        for feature in self.feature_names:
            subset = df[df['feature'] == feature]
            ax.scatter(subset['rgb_value'], subset['lab_value'], 
                      label=feature, alpha=0.6, s=50)
        
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='Perfect match')
        ax.set_xlabel('RGB Value', fontsize=12)
        ax.set_ylabel('LAB Value', fontsize=12)
        ax.set_title('RGB vs LAB Feature Values', fontsize=14)
        ax.legend(loc='upper left', fontsize=8)
        ax.grid(True, alpha=0.3)
        
        # 2. Boxplot: Absolute differences per feature
        ax = axes[0, 1]
        df_pivot = df.pivot_table(
            values='abs_diff', 
            index='image_id', 
            columns='feature'
        )
        df_pivot.boxplot(ax=ax, rot=45)
        ax.set_ylabel('Absolute Difference', fontsize=12)
        ax.set_title('Distribution of Differences (RGB - LAB)', fontsize=14)
        ax.grid(True, alpha=0.3)
        
        # 3. Heatmap: Mean differences
        ax = axes[1, 0]
        mean_diffs = df.groupby('feature')['abs_diff'].mean().sort_values()
        colors = ['green' if x < 0.1 else 'orange' if x < 0.2 else 'red' 
                  for x in mean_diffs.values]
        ax.barh(mean_diffs.index, mean_diffs.values, color=colors)
        ax.set_xlabel('Mean Absolute Difference', fontsize=12)
        ax.set_title('Average Difference per Feature', fontsize=14)
        ax.axvline(0.1, color='green', linestyle='--', alpha=0.5, label='Low (<0.1)')
        ax.axvline(0.2, color='orange', linestyle='--', alpha=0.5, label='Medium (<0.2)')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='x')
        
        # 4. Correlation matrix
        ax = axes[1, 1]
        correlation_data = []
        for feature in self.feature_names:
            subset = df[df['feature'] == feature]
            corr = np.corrcoef(subset['rgb_value'], subset['lab_value'])[0, 1]
            correlation_data.append(corr)
        
        ax.barh(self.feature_names, correlation_data, color='steelblue')
        ax.set_xlabel('Correlation (RGB vs LAB)', fontsize=12)
        ax.set_title('Feature Correlation RGB↔LAB', fontsize=14)
        ax.axvline(0.8, color='green', linestyle='--', alpha=0.5, label='High (>0.8)')
        ax.axvline(0.5, color='orange', linestyle='--', alpha=0.5, label='Medium (>0.5)')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='x')
        ax.set_xlim([0, 1])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"💾 Figura salva em: {save_path}")
        
        plt.show()
    
    def print_summary(self, df: pd.DataFrame):
        """
        Imprime resumo estatístico da comparação.
        
        Args:
            df: DataFrame do compare_batch
        """
        print("="*70)
        print("COMPARAÇÃO RGB vs LAB - RESUMO ESTATÍSTICO")
        print("="*70)
        
        summary = df.groupby('feature').agg({
            'abs_diff': ['mean', 'std', 'min', 'max'],
            'rel_diff': 'mean'
        }).round(4)
        
        print("\n📊 Diferenças Absolutas por Feature:")
        print(summary)
        
        print("\n🔍 Correlações RGB ↔ LAB:")
        for feature in self.feature_names:
            subset = df[df['feature'] == feature]
            corr = np.corrcoef(subset['rgb_value'], subset['lab_value'])[0, 1]
            status = "✅" if corr > 0.8 else "⚠️" if corr > 0.5 else "❌"
            print(f"  {status} {feature:20s}: {corr:.4f}")
        
        print("\n" + "="*70)


# ============================================================================
# TESTE COM IMAGENS SINTÉTICAS
# ============================================================================

def create_test_images() -> List[np.ndarray]:
    """
    Cria 6 imagens de teste com características diferentes.
    
    Returns:
        Lista de arrays numpy (256, 256, 3)
    """
    images = []
    
    # 1. Gradiente vermelho (quente)
    img = np.zeros((256, 256, 3), dtype=np.uint8)
    for i in range(256):
        img[:, i, 0] = i
    images.append(img)
    
    # 2. Gradiente azul (frio)
    img = np.zeros((256, 256, 3), dtype=np.uint8)
    for i in range(256):
        img[:, i, 2] = i
    images.append(img)
    
    # 3. Alta saturação (cores puras)
    img = np.zeros((256, 256, 3), dtype=np.uint8)
    img[:, :85, 0] = 255    # Vermelho
    img[:, 85:170, 1] = 255  # Verde
    img[:, 170:, 2] = 255    # Azul
    images.append(img)
    
    # 4. Baixa saturação (tons de cinza)
    img = np.ones((256, 256, 3), dtype=np.uint8) * 128
    for i in range(256):
        img[i, :, :] = i
    images.append(img)
    
    # 5. Complexo (tabuleiro xadrez)
    img = np.zeros((256, 256, 3), dtype=np.uint8)
    for i in range(0, 256, 32):
        for j in range(0, 256, 32):
            if (i + j) % 64 == 0:
                img[i:i+32, j:j+32] = 255
    images.append(img)
    
    # 6. Simétrico (círculo central)
    img = np.zeros((256, 256, 3), dtype=np.uint8)
    y, x = np.ogrid[:256, :256]
    mask = (x - 128)**2 + (y - 128)**2 <= 64**2
    img[mask] = [255, 255, 255]
    images.append(img)
    
    return images


if __name__ == "__main__":
    print("="*70)
    print("TESTE: Comparação RGB vs LAB")
    print("="*70)
    
    # Cria comparador
    comparator = RGBvsLABComparison()
    
    # Cria imagens de teste
    test_images = create_test_images()
    labels = [
        'Gradiente Vermelho',
        'Gradiente Azul',
        'Alta Saturação',
        'Baixa Saturação',
        'Complexo (Xadrez)',
        'Simétrico (Círculo)'
    ]
    
    # Executa comparação
    print("\n🔍 Comparando features em 6 imagens de teste...")
    df = comparator.compare_batch(test_images, labels)
    
    # Mostra resumo
    comparator.print_summary(df)
    
    # Salva resultados
    output_dir = Path(__file__).parent.parent / 'results'
    output_dir.mkdir(exist_ok=True)
    
    csv_path = output_dir / 'rgb_vs_lab_comparison.csv'
    df.to_csv(csv_path, index=False)
    print(f"\n💾 Resultados salvos em: {csv_path}")
    
    # Visualiza (opcional)
    try:
        fig_path = output_dir / 'rgb_vs_lab_comparison.png'
        comparator.visualize_comparison(df, save_path=str(fig_path))
    except Exception as e:
        print(f"\n⚠️  Visualização falhou (precisa de display): {e}")
    
    print("\n✅ Comparação concluída!")
    print("="*70)
