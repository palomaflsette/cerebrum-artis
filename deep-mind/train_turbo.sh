#!/bin/bash
# MODO TURBO: 4x GTX 1080 Ti full power!
# Sem perguntas, vai direto! 🔥

set -e

# Fix libstdc++ (usa versão do conda, mais nova)
export LD_LIBRARY_PATH=/data/paloma/miniconda3/lib:$LD_LIBRARY_PATH

cd /home/paloma/cerebrum-artis/deep-mind

echo "═══════════════════════════════════════════════"
echo "🔥 MODO TURBO: 4x GTX 1080 Ti ACTIVATED! 🔥"
echo "═══════════════════════════════════════════════"
echo ""
echo "📊 Configuração TURBO-FP16:"
echo "   - GPUs: 4x NVIDIA GTX 1080 Ti"
echo "   - Batch size: 128 por GPU = 512 total"
echo "   - Mixed Precision: FP16 (~2x mais rápido!)"
echo "   - Accumulation: 2 steps (batch efetivo 1024)"
echo "   - Workers: 8"
echo "   - Epochs: 15 (early stop patience: 5)"
echo ""
echo "⚡ MODO MÁXIMA VELOCIDADE!"
echo "⏱️  Tempo estimado: ~6-8 horas (ou menos com early stop)"
echo ""
echo "Iniciando em 3 segundos..."
sleep 3

/data/paloma/venvs/cerebrum-artis/bin/python train_emotion_classifier.py \
    --epochs 15 \
    --early-stop-patience 5 \
    --batch-size 128 \
    --lr 3e-5 \
    --weight-decay 5e-5 \
    --dropout 0.3 \
    --num-workers 8 \
    --multi-gpu \
    --mixed-precision \
    --accumulation-steps 2 \
    --save-every 2 \
    --val-split 0.1 \
    --test-split 0.1

echo ""
echo "═══════════════════════════════════════════════"
echo "✅ TREINO MULTI-GPU CONCLUÍDO!"
echo "═══════════════════════════════════════════════"
