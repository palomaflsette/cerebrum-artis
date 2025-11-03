import torch.distributed as dist
from torch.utils.data import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

import os, argparse, torch, torch.nn as nn, torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.amp import autocast  # new API to avoid deprecation warnings
from torch.cuda.amp import GradScaler
from PIL import Image
from datasets.artemis import ArtemisDataset
from datasets.collate import create_loader
from vocab import load_vocab, PAD, SOS, EOS
from models.sat_baseline import SATBaseline
from config import PROJECT_CSV_DIR

def setup_distributed(args):
    """Prepara DDP a partir das variáveis que o torchrun passa."""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ["WORLD_SIZE"])
        args.local_rank = int(os.environ["LOCAL_RANK"])
        dist.init_process_group(backend="nccl", init_method="env://")
        torch.cuda.set_device(args.local_rank)
        args.distributed = True
    else:
        # modo 1 GPU normal
        args.rank = 0
        args.world_size = 1
        args.local_rank = 0
        args.distributed = False


def parse_args():
    p = argparse.ArgumentParser(description="Train SAT baseline (CNN+LSTM)")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=4, help="micro-batch size per step")
    p.add_argument("--accum-steps", type=int, default=8, help="gradient accumulation steps")
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--enc-lr", type=float, default=1e-5, help="encoder LR when unfrozen")
    p.add_argument("--warmup-epochs", type=int, default=2)
    p.add_argument("--label-smoothing", type=float, default=0.1)
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--num-workers", type=int, default=None)
    p.add_argument("--unfreeze-encoder", action="store_true")
    p.add_argument("--dist", action="store_true", help="forçar modo distribuído (torchrun)")

    return p.parse_args()


def main():
    args = parse_args()
    setup_distributed(args)
    if args.distributed:
        device = torch.device(f"cuda:{args.local_rank}")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    # --- Config
    device = "cuda" if torch.cuda.is_available() else "cpu"
    EPOCHS = args.epochs
    MICRO_BATCH_SIZE = args.batch_size
    ACCUM_STEPS = args.accum_steps
    LEARNING_RATE = args.lr
    ENC_LR = args.enc_lr
    WARMUP_EPOCHS = args.warmup_epochs
    LABEL_SMOOTHING = args.label_smoothing

    # --- Vocab
    # Avoid PIL DecompressionBomb warnings for very large images; we resize later anyway
    Image.MAX_IMAGE_PIXELS = None
    voc = load_vocab(os.path.join(PROJECT_CSV_DIR, "vocab.json"))
    pad_idx = voc["stoi"][PAD]
    vocab_size = len(voc["itos"])

    # --- Data
    train_ds = ArtemisDataset(split="train", caption_col="utterance", image_size=args.image_size)
    val_ds   = ArtemisDataset(split="val",   caption_col="utterance", image_size=args.image_size)
    # Choose a sensible number of workers; if not provided, auto select
    cpu_count = os.cpu_count() or 1
    if args.num_workers is None:
        NUM_WORKERS = max(0, min(8, cpu_count - 1))
    else:
        NUM_WORKERS = args.num_workers
    train_dl = create_loader(train_ds, batch_size=MICRO_BATCH_SIZE, shuffle=True,  num_workers=NUM_WORKERS)
    val_dl   = create_loader(val_ds,   batch_size=MICRO_BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    # Se o dataset for muito pequeno para o ACCUM_STEPS desejado, ajuste para garantir pelo menos 1 passo de otimização por época
    if len(train_dl) < ACCUM_STEPS:
        print(f"[note] len(train_dl)={len(train_dl)} < ACCUM_STEPS={ACCUM_STEPS}; reduzindo ACCUM_STEPS para {len(train_dl)}")
        ACCUM_STEPS = max(1, len(train_dl))

    # --- Model, Loss, Optimizer, Scheduler
    model = SATBaseline(vocab_size=vocab_size, pad_idx=pad_idx, freeze_backbone=not args.unfreeze_encoder).to(device)
    crit  = nn.CrossEntropyLoss(ignore_index=pad_idx, label_smoothing=LABEL_SMOOTHING)
    if args.unfreeze_encoder:
        # different lrs for encoder/decoder
        for p in model.encoder.parameters():
            p.requires_grad = True
        enc_params = [p for p in model.encoder.parameters() if p.requires_grad]
        dec_params = [p for p in model.decoder.parameters() if p.requires_grad]
        opt = optim.AdamW([
            {"params": enc_params, "lr": ENC_LR},
            {"params": dec_params, "lr": LEARNING_RATE},
        ])
    else:
        opt = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE)
    
    # Scheduler with warmup handled manually in the loop
    scheduler = CosineAnnealingLR(opt, T_max=EPOCHS - WARMUP_EPOCHS, eta_min=1e-6)

    if device == "cuda":
        torch.backends.cudnn.benchmark = True  # speed-up for fixed input sizes
    scaler = GradScaler(enabled=(device == "cuda"))
    best_val_loss = float('inf')
    os.makedirs("results", exist_ok=True)

    import math
    steps_per_epoch = math.ceil(len(train_dl) / ACCUM_STEPS) if len(train_dl) else 0
    print(f"Iniciando treino por {EPOCHS} épocas em {device.upper()} | batches={len(train_dl)} | accum={ACCUM_STEPS} | opt-steps/época~{steps_per_epoch} ...")
    for ep in range(1, EPOCHS+1):
        # --- Warmup phase
        if ep <= WARMUP_EPOCHS:
            # Linear warmup
            lr = LEARNING_RATE * (ep / WARMUP_EPOCHS)
            for param_group in opt.param_groups:
                param_group['lr'] = lr
        
        # --- Training loop
        model.train()
        tot_train_loss = 0
        last_step = 0
        for step, (imgs, _, caps) in enumerate(train_dl, start=1):
            imgs, caps = imgs.to(device), caps.to(device)
            with autocast(device_type="cuda", enabled=(device == "cuda")):
                logits = model(imgs, caps)
                # Align dims for loss: (B,T,V) -> (B*T,V) and targets (B*T)
                loss = crit(logits.reshape(-1, logits.size(-1)), caps.reshape(-1))
                # Normalize by ACCUM_STEPS so total grad matches original large batch
                loss = loss / ACCUM_STEPS

            # Accumulate gradients
            scaler.scale(loss).backward()

            if step % ACCUM_STEPS == 0:
                # gradient clipping para estabilidade
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad()
            last_step = step
            
            # Track un-normalized loss for logging
            tot_train_loss += loss.item() * ACCUM_STEPS
        
        # Flush dos gradientes restantes (quando o número de batches não é múltiplo de ACCUM_STEPS)
        if last_step and (last_step % ACCUM_STEPS != 0):
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(opt)
            scaler.update()
            opt.zero_grad()

        avg_train_loss = tot_train_loss / max(1, len(train_dl))
        current_lr = opt.param_groups[0]['lr']
        print(f"[Época {ep:02d}/{EPOCHS}] LR: {current_lr:.6f} | Loss Treino: {avg_train_loss:.4f}", end=" | ")

        # --- Validation loop
        model.eval()
        with torch.no_grad():
            tot_val_loss = 0
            for imgs, _, caps in val_dl:
                imgs, caps = imgs.to(device), caps.to(device)
                with autocast(device_type="cuda", enabled=(device == "cuda")):
                    logits = model(imgs, caps)
                    loss = crit(logits.reshape(-1, logits.size(-1)), caps.reshape(-1))
                tot_val_loss += loss.item()
        
        avg_val_loss = tot_val_loss / len(val_dl)
        print(f"Loss Val: {avg_val_loss:.4f}")

        # --- Scheduler step after warmup
        if ep > WARMUP_EPOCHS:
            scheduler.step()

        # --- Checkpointing
        # Salva melhor modelo (por val loss) e o último
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), "results/baseline_sat_best.pt")
            print(f"  [best] salvo: results/baseline_sat_best.pt")
        torch.save(model.state_dict(), "results/baseline_sat_last.pt")

    # --- Save final model
    # Mantém compatibilidade com o nome antigo
    torch.save(model.state_dict(), "results/baseline_sat.pt")
    print("\n[ok] Checkpoints salvos em results/ (best.pt, last.pt e baseline_sat.pt)")

if __name__ == "__main__":
    main()
