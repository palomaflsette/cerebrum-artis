#!/bin/bash
# Train Deep-Mind v4: Neural + Fuzzy with Intelligent Gating

echo "======================================================================"
echo "🧠 TRAINING DEEP-MIND V4: Intelligent Fuzzy Gating"
echo "======================================================================"
echo ""

cd /home/paloma/cerebrum-artis/deep-mind/v4_fuzzy_gating

# Fix libstdc++ (usa versão do conda, mais nova)
export LD_LIBRARY_PATH=/data/paloma/miniconda3/lib:$LD_LIBRARY_PATH

# Use GPU 1 for v4
export CUDA_VISIBLE_DEVICES=1

# Create log file with timestamp
LOG_DIR="/home/paloma/cerebrum-artis/deep-mind/logs"
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$LOG_DIR/training_v4_$TIMESTAMP.log"

# Run with direct python from cerebrum-artis env
# All hyperparameters are configured in train_v3.py
/data/paloma/venvs/cerebrum-artis/bin/python train_v3.py 2>&1 | tee "$LOG_FILE"

echo ""
echo "✅ Training completed!"
