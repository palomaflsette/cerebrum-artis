#!/bin/bash
# Launch V4.1 Training on GPU 2

export CUDA_VISIBLE_DEVICES=2

cd /home/paloma/cerebrum-artis/deep-mind/v4.1_integrated_gating

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "       🚀 LAUNCHING V4.1 INTEGRATED GATING (GPU 2)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "GPU: 2 (CUDA_VISIBLE_DEVICES=2)"
echo "Working dir: $(pwd)"
echo "Python: /data/paloma/venvs/cerebrum-artis/bin/python"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

/data/paloma/venvs/cerebrum-artis/bin/python train_v3_1.py
