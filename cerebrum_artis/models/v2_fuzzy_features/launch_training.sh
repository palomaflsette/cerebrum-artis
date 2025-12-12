#!/bin/bash
# Launch Deep-Mind v3 training with fuzzy features

cd /home/paloma/cerebrum-artis/deep-mind/v3_fuzzy_features

export LD_LIBRARY_PATH=/data/paloma/miniconda3/lib:$LD_LIBRARY_PATH

echo "🚀 Launching Deep-Mind v3 Training (Multimodal + Fuzzy Features)"
echo "================================================================"
echo ""
echo "This will train overnight. Check progress:"
echo "  tail -f training_v3.log"
echo ""

nohup /data/paloma/venvs/cerebrum-artis/bin/python train_v2.py > training_v3.log 2>&1 &

PID=$!
echo "✅ Training started with PID: $PID"
echo ""
echo "Monitor with:"
echo "  tail -f training_v3.log"
echo "  ps aux | grep $PID"
echo "  nvidia-smi"
echo ""
