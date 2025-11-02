import os, torch, torch.nn as nn, torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.amp import autocast  # new API to avoid deprecation warnings
from torch.cuda.amp import GradScaler
from PIL import Image
from datasets.artemis import ArtemisDataset
from datasets.collate import create_loader
from vocab import load_vocab, PAD, SOS, EOS
from models.sat_baseline import SATBaseline
from config import PROJECT_CSV_DIR

def main():
    # --- Config
    device = "cuda" if torch.cuda.is_available() else "cpu"
    EPOCHS = 20
    # Use micro-batches to fit on small GPUs (e.g., 2GB VRAM on MX550)
    MICRO_BATCH_SIZE = 4  # adjust if you still get OOM: try 2
    ACCUM_STEPS = 8       # effective batch = MICRO_BATCH_SIZE * ACCUM_STEPS (here, 32)
    LEARNING_RATE = 5e-4
    WARMUP_EPOCHS = 2
    LABEL_SMOOTHING = 0.1

    # --- Vocab
    # Avoid PIL DecompressionBomb warnings for very large images; we resize later anyway
    Image.MAX_IMAGE_PIXELS = None
    voc = load_vocab(os.path.join(PROJECT_CSV_DIR, "vocab.json"))
    pad_idx = voc["stoi"][PAD]
    vocab_size = len(voc["itos"])

    # --- Data
    train_ds = ArtemisDataset(split="train", caption_col="utterance")
    val_ds   = ArtemisDataset(split="val",   caption_col="utterance")
    # Choose a sensible number of workers for Windows; keep modest to avoid spawn overhead
    cpu_count = os.cpu_count() or 1
    NUM_WORKERS = max(0, min(4, cpu_count - 1))
    train_dl = create_loader(train_ds, batch_size=MICRO_BATCH_SIZE, shuffle=True,  num_workers=NUM_WORKERS)
    val_dl   = create_loader(val_ds,   batch_size=MICRO_BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    # --- Model, Loss, Optimizer, Scheduler
    model = SATBaseline(vocab_size=vocab_size, pad_idx=pad_idx).to(device)
    crit  = nn.CrossEntropyLoss(ignore_index=pad_idx, label_smoothing=LABEL_SMOOTHING)
    opt   = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE)
    
    # Scheduler with warmup handled manually in the loop
    scheduler = CosineAnnealingLR(opt, T_max=EPOCHS - WARMUP_EPOCHS, eta_min=1e-6)

    if device == "cuda":
        torch.backends.cudnn.benchmark = True  # speed-up for fixed input sizes
    scaler = GradScaler(enabled=(device == "cuda"))
    best_val_loss = float('inf')
    os.makedirs("results", exist_ok=True)

    print(f"Iniciando treino por {EPOCHS} épocas em {device.upper()}...")
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
            
            # Track un-normalized loss for logging
            tot_train_loss += loss.item() * ACCUM_STEPS
        
        avg_train_loss = tot_train_loss / len(train_dl)
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
