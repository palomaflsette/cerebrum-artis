#!/bin/bash
# Train Deep-Mind v3: Neural + Fuzzy Features as Input

echo "======================================================================"
echo "🧠 TRAINING DEEP-MIND V3: Neural + Fuzzy Features"
echo "======================================================================"
echo ""

cd /home/paloma/cerebrum-artis/deep-mind/v3_fuzzy_features

# Fix libstdc++ (usa versão do conda, mais nova)
export LD_LIBRARY_PATH=/data/paloma/miniconda3/lib:$LD_LIBRARY_PATH

# Use GPU 0 for v3
export CUDA_VISIBLE_DEVICES=0

# Run with direct python from cerebrum-artis env
# All hyperparameters are configured in train_v3_cached.py
/data/paloma/venvs/cerebrum-artis/bin/python train_v3_cached.py

echo ""
echo "✅ Training completed!"
