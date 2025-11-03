import os, argparse
import math
import pandas as pd
import torch
from packaging import version
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


def parse_args():
    p = argparse.ArgumentParser(description="Score captions on validation split")
    p.add_argument("--ckpt", type=str, default=os.environ.get("CKPT", "results/baseline_sat_best.pt"))
    p.add_argument("--beam-size", type=int, default=int(os.environ.get("BEAM_SIZE", 1)))
    p.add_argument("--len-pen", type=float, default=float(os.environ.get("LEN_PEN", 0.6)))
    p.add_argument("--no-repeat", type=int, default=int(os.environ.get("NO_REPEAT", 0)))
    p.add_argument("--min-len", type=int, default=int(os.environ.get("MIN_LEN", 0)))
    p.add_argument("--sampling", action="store_true")
    p.add_argument("--top-k", type=int, default=int(os.environ.get("TOP_K", 0)))
    p.add_argument("--top-p", type=float, default=float(os.environ.get("TOP_P", 1.0)))
    p.add_argument("--temperature", type=float, default=float(os.environ.get("TEMPERATURE", 1.0)))
    p.add_argument("--max-len", type=int, default=40)
    return p.parse_args()

def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    voc  = load_vocab(os.path.join(PROJECT_CSV_DIR, "vocab.json"))
    itos = voc["itos"]
    pad_id = itos.index(PAD)
    sos_id = itos.index(SOS)
    eos_id = itos.index(EOS)

    # Data
    ds = ArtemisDataset(split="val", image_size=224, max_len=40)

    # Model
    ckpt = args.ckpt
    beam_size = args.beam_size
    lp_alpha  = args.len_pen
    no_rep_ng = args.no_repeat
    min_len   = args.min_len

    model = CaptionerSAT(vocab_size=len(itos), pad_idx=pad_id)
    # Carregar checkpoint com compatibilidade ampla
    kwargs = {}
    if version.parse(torch.__version__) >= version.parse("2.4.0"):
        kwargs["weights_only"] = True
    state = torch.load(ckpt, map_location="cpu", **kwargs)
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError:
        # Fallback não estrito em caso de pequenas diferenças
        model.load_state_dict(state, strict=False)
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
                max_len=args.max_len,
                sos_id=sos_id,
                eos_id=eos_id,
                beam_size=beam_size,
                length_penalty_alpha=lp_alpha,
                no_repeat_ngram_size=no_rep_ng,
                min_len=min_len,
                banned_token_ids=[itos.index(UNK)] if UNK in itos else None,
                sampling=args.sampling,
                top_k=args.top_k,
                top_p=args.top_p,
                temperature=args.temperature,
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
