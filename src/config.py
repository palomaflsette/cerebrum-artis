import os
from dotenv import load_dotenv

load_dotenv()

def _norm(p: str) -> str:
    return os.path.normpath(p.strip().strip('"').strip("'"))

ARTEMIS_IMAGES_DIR = _norm(os.getenv("ARTEMIS_IMAGES_DIR", ""))
ARTEMIS_CSV_DIR    = _norm(os.getenv("ARTEMIS_CSV_DIR", ""))

PROJECT_CSV_DIR    = _norm(os.getenv("PROJECT_CSV_DIR", "data/csv"))
PROJECT_SPLITS_DIR = _norm(os.getenv("PROJECT_SPLITS_DIR", "data/splits"))

SEED           = int(os.getenv("SEED", "42"))
SUBSET_TARGET  = int(os.getenv("SUBSET_TARGET", "24000"))

SPLIT_TRAIN = float(os.getenv("SPLIT_TRAIN", "0.8"))
SPLIT_VAL   = float(os.getenv("SPLIT_VAL",   "0.1"))
SPLIT_TEST  = float(os.getenv("SPLIT_TEST",  "0.1"))

for k, v in {
    "ARTEMIS_IMAGES_DIR": ARTEMIS_IMAGES_DIR,
    "ARTEMIS_CSV_DIR": ARTEMIS_CSV_DIR,
}.items():
    if not os.path.isdir(v):
        raise FileNotFoundError(f"[config] Folder not found: {k} = {v}")

os.makedirs(PROJECT_CSV_DIR, exist_ok=True)
os.makedirs(PROJECT_SPLITS_DIR, exist_ok=True)
