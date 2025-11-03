import os, argparse
import pandas as pd
from vocab import build_vocab, save_vocab
from config import PROJECT_SPLITS_DIR, PROJECT_CSV_DIR

def parse_args():
    p = argparse.ArgumentParser(description="Construir vocabulário a partir do split de treino")
    p.add_argument("--cap-col", type=str, default="utterance", help="Coluna de texto no CSV")
    p.add_argument("--min-freq", type=int, default=1, help="Frequência mínima para entrar no vocabulário")
    p.add_argument("--input", type=str, default=None, help="Caminho opcional para um CSV específico (default: artemis_train.csv)")
    p.add_argument("--out", type=str, default=None, help="Caminho opcional para salvar vocab.json")
    return p.parse_args()

def main():
    args = parse_args()

    train_csv = args.input or os.path.join(PROJECT_SPLITS_DIR, "artemis_train.csv")
    df = pd.read_csv(train_csv)

    if args.cap_col not in df.columns:
        raise SystemExit(f"Coluna '{args.cap_col}' não existe em {train_csv}. Colunas: {list(df.columns)}")

    voc = build_vocab(df[args.cap_col], min_freq=args.min_freq)
    out_path = args.out or os.path.join(PROJECT_CSV_DIR, "vocab.json")
    save_vocab(voc, out_path)
    print(f"[ok] vocab -> {out_path} | tamanho={len(voc['itos'])} | min_freq={args.min_freq}")

if __name__ == "__main__":
    main()
