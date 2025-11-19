#!/bin/bash
# Script de treino otimizado para 4x GTX 1080 Ti (11GB cada)
# 40 CPUs, 125GB RAM
# Dataset: 692k samples

set -e

cd /home/paloma/cerebrum-artis/deep-mind

echo "🔥 CONFIGURAÇÃO OTIMIZADA PARA SEU HARDWARE:"
echo "   - 4x GTX 1080 Ti (11GB)"
echo "   - 40 CPUs"
echo "   - 125GB RAM"
echo ""

# ============================================
# OPÇÃO 1: SINGLE GPU (Recomendado para começar)
# ============================================
echo "🚀 OPÇÃO 1: Single GPU (GPU 0)"
echo "   Batch size: 96 (max para 11GB GTX 1080 Ti)"
echo "   Workers: 12 (ótimo para 40 CPUs)"
echo "   Tempo estimado: ~6-7 horas"
echo "   Acurácia esperada: 78-83%"
echo ""

read -p "Deseja iniciar treino SINGLE GPU? (s/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Ss]$ ]]; then
    /data/paloma/venvs/cerebrum-artis/bin/python train_emotion_classifier.py \
        --epochs 20 \
        --batch-size 96 \
        --lr 3e-5 \
        --weight-decay 1e-4 \
        --dropout 0.2 \
        --num-workers 12 \
        --gpu 0 \
        --save-every 3 \
        --val-split 0.1 \
        --test-split 0.1
    
    echo "✅ Treino concluído!"
    exit 0
fi

# ============================================
# OPÇÃO 2: MULTI GPU (DataParallel - 4x GTX 1080 Ti)
# ============================================
echo ""
echo "🚀 OPÇÃO 2: Multi-GPU (4x GPUs, DataParallel) ⚡"
echo "   Batch size: 96 → 384 efetivo (96 x 4 GPUs)"
echo "   Workers: 16"
echo "   Tempo estimado: ~1.5-2 horas (4x speedup!)"
echo "   Acurácia esperada: 78-83%"
echo ""

read -p "Deseja iniciar treino MULTI-GPU? (s/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Ss]$ ]]; then
    /data/paloma/venvs/cerebrum-artis/bin/python train_emotion_classifier.py \
        --epochs 20 \
        --batch-size 96 \
        --lr 3e-5 \
        --weight-decay 1e-4 \
        --dropout 0.2 \
        --num-workers 16 \
        --multi-gpu \
        --save-every 2 \
        --val-split 0.1 \
        --test-split 0.1
    
    echo "✅ Treino Multi-GPU concluído!"
    exit 0
fi

echo "❌ Treino cancelado pelo usuário."
