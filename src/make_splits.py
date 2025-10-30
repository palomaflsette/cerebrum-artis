import os
import sys
import math
import random
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import GroupShuffleSplit
from collections import Counter

from config import (
    ARTEMIS_CSV_DIR, ARTEMIS_IMAGES_DIR,
    PROJECT_CSV_DIR, PROJECT_SPLITS_DIR,
    SEED, SUBSET_TARGET, SPLIT_TRAIN, SPLIT_VAL, SPLIT_TEST
)

random.seed(SEED)
np.random.seed(SEED)
 
CSV_PATH = os.path.join(ARTEMIS_CSV_DIR, "artemis_dataset_release_v0.csv")
df = pd.read_csv(CSV_PATH, delimiter=";")

# Esperados
# 'emotion'  -> classe alvo
# 'painting' -> nome do arquivo (ou caminho relativo)
# 'art_style', 'artist', ... (opcionais)
# 'painting_id' -> id único por obra (se inexistente, derivamos de 'painting' sem o sufixo)

required_cols = ["emotion", "painting"]
for c in required_cols:
    if c not in df.columns:
        raise KeyError(f"Coluna ausente no CSV: {c}")

def derive_id(p):
    base = os.path.splitext(os.path.basename(str(p)))[0]
    parts = base.split("_")
    if parts and parts[-1].isdigit():
        parts = parts[:-1]
    return "_".join(parts) if parts else base

if "painting_id" not in df.columns:
    df["painting_id"] = df["painting"].apply(derive_id)

def abs_path(p):
    return os.path.join(ARTEMIS_IMAGES_DIR, os.path.normpath(str(p)))

df["abs_path"] = df["painting"].apply(abs_path)

df["emotion"] = df["emotion"].astype(str).str.strip()


counts = df["emotion"].value_counts()
print("[info] Distribuição original:")
print(counts)

n_classes = df["emotion"].nunique()
per_class = max(1, SUBSET_TARGET // n_classes)

balanced_frames = []
for emotion, grp in df.groupby("emotion", sort=False):

    grp_sample = grp.drop_duplicates(subset=["painting_id"])
    take = min(len(grp_sample), per_class)
    balanced = grp_sample.sample(n=take, random_state=SEED)
    balanced_frames.append(balanced)

df_bal = pd.concat(balanced_frames, ignore_index=True)

print("\n[info] Subset balanceado (1 por obra):")
print(df_bal["emotion"].value_counts())

groups = df_bal["painting_id"].values
gss = GroupShuffleSplit(n_splits=1, train_size=SPLIT_TRAIN, random_state=SEED)
train_idx, rest_idx = next(gss.split(df_bal, groups=groups))

df_train = df_bal.iloc[train_idx]
df_rest  = df_bal.iloc[rest_idx]

rest_groups = df_rest["painting_id"].values
val_size = SPLIT_VAL / (SPLIT_VAL + SPLIT_TEST)
gss2 = GroupShuffleSplit(n_splits=1, train_size=val_size, random_state=SEED)
val_idx, test_idx = next(gss2.split(df_rest, groups=rest_groups))

df_val  = df_rest.iloc[val_idx]
df_test = df_rest.iloc[test_idx]


assert set(df_train["painting_id"]).isdisjoint(df_val["painting_id"])
assert set(df_train["painting_id"]).isdisjoint(df_test["painting_id"])
assert set(df_val["painting_id"]).isdisjoint(df_test["painting_id"])

print("\n[info] Tamanhos:")
print("train:", len(df_train), "val:", len(df_val), "test:", len(df_test))

print("\n[info] Distribuição por split:")
for name, dfx in [("train", df_train), ("val", df_val), ("test", df_test)]:
    print(f"\n{name}:")
    print(dfx["emotion"].value_counts())

train_csv = os.path.join(PROJECT_SPLITS_DIR, "artemis_train.csv")
val_csv   = os.path.join(PROJECT_SPLITS_DIR, "artemis_val.csv")
test_csv  = os.path.join(PROJECT_SPLITS_DIR, "artemis_test.csv")

df_train.to_csv(train_csv, index=False)
df_val.to_csv(val_csv, index=False)
df_test.to_csv(test_csv, index=False)

subset_csv = os.path.join(PROJECT_CSV_DIR, "artemis_subset_balanced.csv")
df_bal.to_csv(subset_csv, index=False)

print(f"\n[ok] Salvo:\n- {train_csv}\n- {val_csv}\n- {test_csv}\n- {subset_csv}")
