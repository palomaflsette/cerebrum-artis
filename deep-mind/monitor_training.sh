#!/bin/bash
# Monitor completo: GPUs + Treino + Sistema

clear

echo "════════════════════════════════════════════════════════════════"
echo "🔥 DEEP-MIND TRAINING MONITOR"
echo "════════════════════════════════════════════════════════════════"
echo ""

# Função para monitorar
monitor() {
    while true; do
        clear
        echo "════════════════════════════════════════════════════════════════"
        echo "🔥 DEEP-MIND TRAINING MONITOR - $(date '+%H:%M:%S')"
        echo "════════════════════════════════════════════════════════════════"
        echo ""
        
        # === GPUs ===
        echo "🖥️  GPU STATUS:"
        nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw \
            --format=csv,noheader,nounits | \
            awk -F", " '{
                util = $3
                vram_used = $4
                vram_total = $5
                temp = $6
                power = $7
                
                # Cores baseado em utilização
                if (util >= 90) color="\033[1;32m"  # Verde forte
                else if (util >= 50) color="\033[1;33m"  # Amarelo
                else color="\033[1;31m"  # Vermelho
                
                printf "  GPU %s: %-20s | %sUso: %3s%%%s | VRAM: %5s/%5sMB (%2d%%) | Temp: %2s°C | %3sW\n",
                    $1, $2, color, util, "\033[0m", vram_used, vram_total, 
                    int(vram_used/vram_total*100), temp, power
            }'
        echo ""
        
        # === Processo de treino ===
        echo "🚀 TRAINING PROCESS:"
        if pgrep -f train_emotion_classifier.py > /dev/null; then
            PID=$(pgrep -f train_emotion_classifier.py)
            echo "  ✅ Running (PID: $PID)"
            
            # CPU e RAM do processo
            ps -p $PID -o %cpu,%mem,rss,etime --no-headers | \
                awk '{printf "  CPU: %5.1f%% | RAM: %5.1fGB | Runtime: %s\n", $1, $3/1024/1024, $4}'
            
            # Última linha do log (se existir)
            LOG_DIR="/home/paloma/cerebrum-artis/deep-mind/checkpoints"
            LATEST_LOG=$(ls -t $LOG_DIR/*/tb_log* 2>/dev/null | head -1)
            if [ -n "$LATEST_LOG" ]; then
                echo "  Log: $LATEST_LOG"
            fi
        else
            echo "  ❌ Not running"
        fi
        echo ""
        
        # === Sistema ===
        echo "💻 SYSTEM:"
        free -h | grep "Mem:" | awk '{printf "  RAM: %s / %s (%.1f%% usado)\n", $3, $2, ($3/$2)*100}'
        
        # CPU total
        top -bn1 | grep "Cpu(s)" | awk '{printf "  CPU: %.1f%% usado\n", 100-$8}'
        
        # Disco (só home)
        df -h /home | tail -1 | awk '{printf "  Disk /home: %s / %s (%s usado)\n", $3, $2, $5}'
        echo ""
        
        # === Checkpoints ===
        echo "💾 CHECKPOINTS:"
        CKPT_DIR="/home/paloma/cerebrum-artis/deep-mind/checkpoints"
        if [ -d "$CKPT_DIR" ]; then
            LATEST_DIR=$(ls -td $CKPT_DIR/multimodal_* 2>/dev/null | head -1)
            if [ -n "$LATEST_DIR" ]; then
                echo "  Dir: $(basename $LATEST_DIR)"
                CKPTS=$(ls $LATEST_DIR/*.pt 2>/dev/null | wc -l)
                if [ $CKPTS -gt 0 ]; then
                    echo "  Saved: $CKPTS checkpoint(s)"
                    ls -lh $LATEST_DIR/*.pt 2>/dev/null | tail -3 | \
                        awk '{printf "    %s  %s  %s\n", $9, $5, $6" "$7}'
                fi
            fi
        fi
        echo ""
        
        echo "════════════════════════════════════════════════════════════════"
        echo "Atualiza a cada 2s | Ctrl+C para sair"
        
        sleep 2
    done
}

monitor
