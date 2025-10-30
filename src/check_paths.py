import os
import sys

import pandas as pd
from config import ARTEMIS_IMAGES_DIR, PROJECT_SPLITS_DIR
from datasets.artemis import EXTS

def resolve(row):
    style = str(row["art_style"])
    name  = str(row["painting"])
    base  = os.path.join(ARTEMIS_IMAGES_DIR, style, name)
    if os.path.splitext(base)[1].lower() in EXTS and os.path.isfile(base):
        return base
    root, ext = os.path.splitext(base)
    for e in EXTS:
        cand = root + e
        if os.path.isfile(cand):
            return cand
    return ""

for split in ["train","val","test"]:
    csv_path = os.path.join(PROJECT_SPLITS_DIR, f"artemis_{split}.csv")
    df = pd.read_csv(csv_path)
    ok = df.apply(resolve, axis=1).ne("").mean()
    print(f"{split}: {ok*100:.1f}% com caminho válido")
