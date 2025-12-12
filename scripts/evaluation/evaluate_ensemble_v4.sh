#!/bin/bash
# Quick evaluation of V4 Ensemble

set -e

echo "═══════════════════════════════════════════════"
echo "🔬 V4 ENSEMBLE EVALUATION"
echo "═══════════════════════════════════════════════"
echo ""

# Activate environment
source /data/paloma/venvs/artemis-sat/bin/activate

cd /home/paloma/cerebrum-artis

echo "📊 Option 1: Validation set (50/50 weights)"
echo "   python cerebrum_artis/models/ensemble/evaluate_v4.py --split val"
echo ""
echo "📊 Option 2: Validation set (optimized weights)"
echo "   python cerebrum_artis/models/ensemble/evaluate_v4.py --split val --optimize-weights"
echo ""
echo "📊 Option 3: Test set (50/50 weights)"
echo "   python cerebrum_artis/models/ensemble/evaluate_v4.py --split test"
echo ""
echo "Which option? (1/2/3): "
read option

case $option in
    1)
        echo "Running validation with 50/50 weights..."
        python cerebrum_artis/models/ensemble/evaluate_v4.py --split val
        ;;
    2)
        echo "Running validation with optimized weights..."
        python cerebrum_artis/models/ensemble/evaluate_v4.py --split val --optimize-weights
        ;;
    3)
        echo "Running test with 50/50 weights..."
        python cerebrum_artis/models/ensemble/evaluate_v4.py --split test
        ;;
    *)
        echo "Invalid option"
        exit 1
        ;;
esac

echo ""
echo "═══════════════════════════════════════════════"
echo "✅ EVALUATION COMPLETE!"
echo "═══════════════════════════════════════════════"
