import os, torch, random, pandas as pd
from config import PROJECT_SPLITS_DIR, PROJECT_CSV_DIR
from datasets.artemis import ArtemisDataset
from models.sat_baseline import SATBaseline as CaptionerSAT    
from vocab import load_vocab, PAD, SOS, EOS, UNK
from torchvision.utils import save_image

CKPT = "results/baseline_sat.pt" 
N     = 6
MAX_LEN = 40

def main():
    voc  = load_vocab(os.path.join(PROJECT_CSV_DIR, "vocab.json"))
    itos = voc["itos"]
    pad_id = itos.index(PAD)
    sos_id = itos.index(SOS)
    eos_id = itos.index(EOS)
    unk_id = itos.index(UNK)

    ds = ArtemisDataset(split="val", image_size=224, max_len=MAX_LEN)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = CaptionerSAT(vocab_size=len(itos), pad_idx=pad_id)  
    model.load_state_dict(torch.load(CKPT, map_location=device))
    model.to(device).eval()

    os.makedirs("samples", exist_ok=True)
    rows = []

    from torchvision.utils import save_image
    idxs = random.sample(range(len(ds)), k=min(N, len(ds)))

    for j, i in enumerate(idxs):
        img, _, cap_ids = ds[i] 
        gt_tokens = [itos[int(t)] for t in cap_ids.tolist() if t not in (pad_id, sos_id, eos_id)]
        gt = " ".join(gt_tokens)

        if hasattr(model, "generate"):
            pred_ids = model.generate(
                img.unsqueeze(0).to(device),
                max_len=MAX_LEN,
                sos_id=sos_id,
                eos_id=eos_id
            )[0]
        else:
            with torch.no_grad():
                x = torch.tensor([[sos_id]], device=device)
                state = model.init_state(img.unsqueeze(0).to(device))
                pred_ids = []
                for _ in range(MAX_LEN):
                    logits, state = model.step(x, state)
                    nxt = logits.argmax(-1).item()
                    if nxt == eos_id: break
                    pred_ids.append(nxt)
                    x = torch.tensor([[nxt]], device=device)

        pred = " ".join(itos[t] if 0 <= t < len(itos) else UNK for t in pred_ids)

        save_image(img, f"samples/img_{j}.png")
        rows.append((f"samples/img_{j}.png", pred, gt))

    pd.DataFrame(rows, columns=["img","pred","gt"]).to_csv("samples/predicoes.csv", index=False)
    print("→ veja `samples/` e `samples/predicoes.csv`")

if __name__ == "__main__":
    main()