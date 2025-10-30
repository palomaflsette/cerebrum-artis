import os, torch, torch.nn as nn, torch.optim as optim
from datasets.artemis import ArtemisDataset
from datasets.collate import create_loader
from vocab import load_vocab, PAD, SOS, EOS
from models.sat_baseline import SATBaseline
from config import PROJECT_CSV_DIR

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    voc = load_vocab(os.path.join(PROJECT_CSV_DIR, "vocab.json"))
    pad_idx = voc["stoi"][PAD]
    vocab_size = len(voc["itos"])

    train_ds = ArtemisDataset(split="train", caption_col="utterance")
    val_ds   = ArtemisDataset(split="val",   caption_col="utterance")
    train_dl = create_loader(train_ds, batch_size=32, shuffle=True)
    val_dl   = create_loader(val_ds,   batch_size=32, shuffle=False)

    model = SATBaseline(vocab_size=vocab_size, pad_idx=pad_idx).to(device)
    crit  = nn.CrossEntropyLoss(ignore_index=pad_idx)
    opt   = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)

    EPOCHS = 3
    for ep in range(1, EPOCHS+1):
        model.train()
        tot = 0
        for imgs, _, caps in train_dl:
            imgs, caps = imgs.to(device), caps.to(device)
            logits = model(imgs, caps)
            # alinhar dims: (B,T,V) -> (B*T,V) e targets (B*T)
            loss = crit(logits.reshape(-1, logits.size(-1)), caps.reshape(-1))
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item()
        print(f"[ep {ep}] train loss: {tot/len(train_dl):.3f}")

        # validação rápida
        model.eval()
        with torch.no_grad():
            totv=0
            for imgs, _, caps in val_dl:
                imgs, caps = imgs.to(device), caps.to(device)
                logits = model(imgs, caps)
                loss = crit(logits.reshape(-1, logits.size(-1)), caps.reshape(-1))
                totv += loss.item()
        print(f"[ep {ep}]   val loss: {totv/len(val_dl):.3f}")

    torch.save(model.state_dict(), "results/baseline_sat.pt")
    print("[ok] checkpoint salvo em results/baseline_sat.pt")

if __name__ == "__main__":
    main()
