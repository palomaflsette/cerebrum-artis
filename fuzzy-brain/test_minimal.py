#!/usr/bin/env python3
"""Teste minimalista para debugar"""
import sys
import cv2

print("Python version:", sys.version)
print("OpenCV version:", cv2.__version__)

# Testar com apenas uma imagem
img_path = "/data/paloma/data/paintings/wikiart/Impressionism/claude-monet_snow-at-argenteuil-02.jpg"
print(f"\nTentando carregar: {img_path}")

img = cv2.imread(img_path)
if img is None:
    print("❌ ERRO: Não conseguiu carregar!")
else:
    print(f"✅ OK! Shape: {img.shape}")
    print(f"   Dtype: {img.dtype}")
