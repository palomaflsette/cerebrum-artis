import os, pandas as pd
from vocab import build_vocab, save_vocab
from config import PROJECT_SPLITS_DIR, PROJECT_CSV_DIR

CAP_COL = "utterance" 

train_csv = os.path.join(PROJECT_SPLITS_DIR, "artemis_train.csv")
df = pd.read_csv(train_csv)

if CAP_COL not in df.columns:
    raise SystemExit(f"Coluna '{CAP_COL}' não existe em {train_csv}. Colunas: {list(df.columns)}")

voc = build_vocab(df[CAP_COL], min_freq=5)
out_path = os.path.join(PROJECT_CSV_DIR, "vocab.json")
save_vocab(voc, out_path)
print(f"[ok] vocab -> {out_path} | tamanho={len(voc['itos'])}")
