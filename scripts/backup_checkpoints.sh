#!/bin/bash
# ============================================================================
# BACKUP DE CHECKPOINTS TREINADOS
# ============================================================================
# Salva checkpoints importantes de /data/paloma para outputs/checkpoints
# com timestamp para evitar sobrescrita acidental
# ============================================================================

set -e  # Para se der erro

# Cores para output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}🔒 BACKUP DE CHECKPOINTS - Cerebrum Artis${NC}"
echo "========================================================================"

# Diretórios
SOURCE_DIR="/data/paloma/deep-mind-checkpoints"
BACKUP_DIR="/home/paloma/cerebrum-artis/outputs/checkpoints"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Cria estrutura de backup
mkdir -p "$BACKUP_DIR"

# Lista de modelos importantes para backup
MODELS=(
    "v2_fuzzy_features"
    "v3_adaptive_gating"
    "v3_1_integrated"
    "fuzzy_gating_v4"
)

echo ""
echo "📦 Fazendo backup dos checkpoints BEST..."
echo ""

total_size=0

for model in "${MODELS[@]}"; do
    src="$SOURCE_DIR/$model"
    
    if [ ! -d "$src" ]; then
        echo -e "${YELLOW}⚠️  $model: diretório não encontrado, pulando...${NC}"
        continue
    fi
    
    # Cria pasta de destino com timestamp
    dest="$BACKUP_DIR/${model}_backup_${TIMESTAMP}"
    mkdir -p "$dest"
    
    # Copia SOMENTE checkpoint_best.pt (mais importante)
    if [ -f "$src/checkpoint_best.pt" ]; then
        echo "✅ Copiando $model/checkpoint_best.pt..."
        cp "$src/checkpoint_best.pt" "$dest/"
        
        # Calcula tamanho
        size=$(du -sh "$dest/checkpoint_best.pt" | cut -f1)
        total_size=$((total_size + $(du -s "$dest/checkpoint_best.pt" | cut -f1)))
        
        echo "   └─ Salvo em: $dest/checkpoint_best.pt ($size)"
    else
        echo -e "${YELLOW}⚠️  $model: checkpoint_best.pt não encontrado${NC}"
    fi
    
    # OPCIONAL: Copia também training_log.txt se existir
    if [ -f "$src/training_log.txt" ]; then
        cp "$src/training_log.txt" "$dest/"
        echo "   └─ Log de treino também salvo"
    fi
    
    echo ""
done

# Converte total_size para human-readable
total_mb=$((total_size / 1024))
echo "========================================================================"
echo -e "${GREEN}✅ Backup concluído!${NC}"
echo "📊 Total backup: ~${total_mb} MB"
echo "📁 Localização: $BACKUP_DIR"
echo ""
echo "🔍 Estrutura criada:"
ls -lh "$BACKUP_DIR" | tail -n +2 | awk '{print "   " $9 " (" $5 ")"}'
echo ""
echo -e "${YELLOW}💡 IMPORTANTE:${NC} Os treinos futuros NÃO vão sobrescrever esses backups!"
echo "   Eles estão em outputs/checkpoints/ (separado de /data/paloma/)"
echo "========================================================================"
