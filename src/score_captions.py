import os
import math
import pandas as pd
import torch
from tqdm import tqdm

from datasets.artemis import ArtemisDataset
from vocab import load_vocab, PAD, SOS, EOS, UNK
from models.sat_baseline import SATBaseline as CaptionerSAT
from config import PROJECT_CSV_DIR

# Metrics
from sacrebleu import corpus_bleu
from rouge_score import rouge_scorer


def ids_to_text(ids, itos, unk=UNK):
    toks = []
    for t in ids:
        if 0 <= t < len(itos):
            toks.append(itos[t])
        else:
            toks.append(unk)
    return " ".join(toks)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    voc  = load_vocab(os.path.join(PROJECT_CSV_DIR, "vocab.json"))
    itos = voc["itos"]
    pad_id = itos.index(PAD)
    sos_id = itos.index(SOS)
    eos_id = itos.index(EOS)

    # Data
    ds = ArtemisDataset(split="val", image_size=224, max_len=40)

    # Model
    ckpt = os.environ.get("CKPT", "results/baseline_sat_best.pt")
    beam_size = int(os.environ.get("BEAM_SIZE", 3))
    lp_alpha  = float(os.environ.get("LEN_PEN", 0.6))
    no_rep_ng = int(os.environ.get("NO_REPEAT", 2))
    min_len   = int(os.environ.get("MIN_LEN", 3))

    model = CaptionerSAT(vocab_size=len(itos), pad_idx=pad_id)
    state = torch.load(ckpt, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.to(device).eval()

    refs = []
    hyps = []
    rows = []

    for i in tqdm(range(len(ds)), desc="Scoring val"):
        img, _, cap_ids = ds[i]
        gt = [itos[int(t)] for t in cap_ids.tolist() if t not in (pad_id, itos.index(SOS), eos_id)]
        gt_text = " ".join(gt)

        with torch.no_grad():
            pred_ids = model.generate(
                img.unsqueeze(0).to(device),
                max_len=40,
                sos_id=sos_id,
                eos_id=eos_id,
                beam_size=beam_size,
                length_penalty_alpha=lp_alpha,
                no_repeat_ngram_size=no_rep_ng,
                min_len=min_len,
            )[0]
        pred_text = ids_to_text(pred_ids, itos)

        hyps.append(pred_text)
        refs.append([gt_text])  # sacrebleu expects list of references per sample
        rows.append({"id": i, "pred": pred_text, "gt": gt_text})

    # Metrics
    bleu = corpus_bleu(hyps, list(zip(*refs)))  # sacrebleu API: list of ref streams
    rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    rouge_scores = [rouge.score(r[0], h)["rougeL"].fmeasure for h, r in zip(hyps, refs)]
    rougeL_f = sum(rouge_scores) / max(1, len(rouge_scores))

    os.makedirs("results", exist_ok=True)
    pd.DataFrame(rows).to_csv("results/predicoes_scored.csv", index=False)
    with open("results/scores.txt", "w", encoding="utf-8") as f:
        f.write(f"BLEU = {bleu.score:.2f}\n")
        f.write(f"ROUGE_L_F = {rougeL_f*100:.2f}\n")
        f.write(f"Details: {bleu}\n")

    print("Saved:")
    print(" - results/predicoes_scored.csv")
    print(" - results/scores.txt")


if __name__ == "__main__":
    main()
