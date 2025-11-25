# Scripts

> Executable scripts for training, evaluation, and demonstration.

## Directory Structure

```
scripts/
├── training/           # Model training scripts
├── evaluation/         # Model evaluation and testing
└── demo/              # Demo applications
```

## Training Scripts

### train_v3.py

Train the V3 Fuzzy Features model (recommended).

```bash
python scripts/training/train_v3.py \
    --config configs/training/v3_config.json \
    --data_path /path/to/artemis/data \
    --checkpoint_dir /path/to/checkpoints \
    --num_epochs 20 \
    --batch_size 32 \
    --lr 1e-4
```

**Arguments:**
- `--config`: Path to configuration JSON file
- `--data_path`: Path to ArtEmis dataset
- `--checkpoint_dir`: Directory to save checkpoints
- `--num_epochs`: Number of training epochs (default: 20)
- `--batch_size`: Batch size (default: 32)
- `--lr`: Learning rate (default: 1e-4)
- `--resume`: Path to checkpoint to resume from

### train_v4.py

Train the V4 Fuzzy Gating model.

```bash
python scripts/training/train_v4.py \
    --config configs/training/v4_config.json \
    --data_path /path/to/artemis/data
```

### train_v4_1.py

Train the V4.1 Integrated Gating model.

```bash
python scripts/training/train_v4_1.py \
    --config configs/training/v4_1_config.json \
    --data_path /path/to/artemis/data
```

## Evaluation Scripts

### ensemble_test.py

Evaluate ensemble combinations of multiple models.

```bash
python scripts/evaluation/ensemble_test.py \
    --models v3 v4 v4_1 \
    --checkpoint_dir /path/to/checkpoints \
    --data_path /path/to/artemis/data \
    --strategies simple voting optimized
```

**Strategies:**
- `simple`: Simple average of probabilities
- `voting`: Hard voting (majority wins)
- `weighted`: Performance-weighted average
- `optimized`: Grid search for optimal weights

**Output:**
```
Ensemble Results:
  Simple Average:        71.26%
  Hard Voting:          71.13%
  Performance-Weighted: 71.27%
  Optimized:           71.47% ⭐ (best)
```

## Demo Scripts

### demo_percepto.py

Interactive demo of the PerceptoEmocional agent.

```bash
python scripts/demo/demo_percepto.py \
    --image_path path/to/painting.jpg \
    --caption "A vibrant sunset over mountains"
```

**Interactive Mode:**
```bash
python scripts/demo/demo_percepto.py --interactive
```

## Utility Scripts

### Quick Evaluation

Evaluate a single model quickly:

```bash
python scripts/evaluation/eval_models.py \
    --model v3 \
    --checkpoint checkpoints/v3_fuzzy_features/checkpoint_best.pt \
    --test_split validation
```

### Batch Processing

Process multiple images:

```bash
python scripts/evaluation/batch_process.py \
    --input_dir /path/to/images \
    --output_file results.json \
    --model ensemble
```

## Development

### Running in Debug Mode

```bash
python scripts/training/train_v3.py \
    --debug \
    --log_level DEBUG \
    --num_epochs 1 \
    --batch_size 4
```

### Monitoring Training

Use TensorBoard to monitor training:

```bash
tensorboard --logdir logs/
```

---

**Last Updated**: November 25, 2025
