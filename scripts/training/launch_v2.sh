#!/bin/bash

# Activate conda environment
source /data/paloma/miniconda3/etc/profile.d/conda.sh
conda activate artemis-sat

# Set GPU
export CUDA_VISIBLE_DEVICES=1

# Run training
cd /home/paloma/cerebrum-artis
python scripts/training/train_v2_improved.py
