#!/usr/bin/env python3
"""
Pre-compute fuzzy features for ALL images
==========================================
Extrai features fuzzy UMA VEZ e salva em disco.
Treinamento depois só carrega (instantâneo).
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import pickle

sys.path.insert(0,"../../fuzzy-brain")

from fuzzy_brain.extractors.visual import VisualFeatureExtractor

def extract_features_for_image(img_path):
    """Extract fuzzy features for one image (parallelizable)"""
    try:
        extractor = VisualFeatureExtractor()
        features = extractor.extract_all(str(img_path))
        
        # Convert to numpy array (7 features)
        feature_vector = np.array([
            features['brightness'],
            features['color_temperature'],
            features['saturation'],
            features['color_harmony'],
            features['complexity'],
            features['symmetry'],
            features['texture_roughness']
        ], dtype=np.float32)
        
        return img_path.stem, feature_vector
    except Exception as e:
        print(f"ERROR on {img_path.stem}: {e}")
        return img_path.stem, None

def main():
    # Paths
    image_dir = Path('/data/paloma/data/paintings/wikiart')
    output_file = Path('/data/paloma/fuzzy_features_cache.pkl')
    
    print("=" * 80)
    print("🧠 PRE-COMPUTING FUZZY FEATURES")
    print("=" * 80)
    print(f"📂 Image dir: {image_dir}")
    print(f"💾 Output: {output_file}")
    print()
    
    # Index all images
    print("🔍 Indexing images...")
    image_paths = list(image_dir.rglob('*.jpg'))
    print(f"✅ Found {len(image_paths)} images")
    print()
    
    # Process in parallel
    print("⚡ Extracting features in parallel (using ALL CPU cores)...")
    fuzzy_cache = {}
    
    with ProcessPoolExecutor(max_workers=16) as executor:
        futures = {executor.submit(extract_features_for_image, img_path): img_path 
                   for img_path in image_paths}
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing"):
            img_key, features = future.result()
            if features is not None:
                fuzzy_cache[img_key] = features
    
    print(f"\n✅ Extracted features for {len(fuzzy_cache)} images")
    print(f"💾 Saving to {output_file}...")
    
    with open(output_file, 'wb') as f:
        pickle.dump(fuzzy_cache, f, protocol=4)
    
    print(f"✅ Saved! File size: {output_file.stat().st_size / 1024 / 1024:.2f} MB")
    print()
    print("🎉 Done! Now you can train v3 FAST!")
    print("=" * 80)

if __name__ == '__main__':
    main()
