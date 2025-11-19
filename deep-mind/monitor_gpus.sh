#!/bin/bash
# Monitor de GPU em tempo real durante treino
# Atualiza a cada 2 segundos

watch -n 2 'nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw --format=csv,noheader,nounits | awk -F", " "{printf \"GPU %s: %s | Uso: %3s%% | VRAM: %5s/%5sMB | Temp: %2s°C | Power: %sW\n\", \$1, \$2, \$3, \$4, \$5, \$6, \$7}"'
