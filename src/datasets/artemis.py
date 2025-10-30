# src/datasets/artemis.py
import os, torch, pandas as pd, glob
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T
from vocab import load_vocab, tokenize, PAD, SOS, EOS, UNK
from labels import EMO2ID, normalize_emotion
from config import ARTEMIS_IMAGES_DIR, PROJECT_SPLITS_DIR, PROJECT_CSV_DIR

EXTS = [".jpg", ".jpeg", ".png", ".webp", ".bmp"]

class ArtemisDataset(Dataset):
    def __init__(self, split="train", image_size=224, max_len=40,
                 caption_col="utterance", log_missing=True):
        csv_path = os.path.join(PROJECT_SPLITS_DIR, f"artemis_{split}.csv")
        self.df = pd.read_csv(csv_path)
        self.caption_col = caption_col
        self.max_len = max_len
        self.log_missing = log_missing
        self.missing = []

        self.vocab = load_vocab(os.path.join(PROJECT_CSV_DIR, "vocab.json"))
        self.stoi = self.vocab["stoi"]

        self.tf = T.Compose([
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
        ])

        # Colunas esperadas
        assert "painting" in self.df.columns, "CSV precisa ter coluna 'painting'"
        assert "emotion"  in self.df.columns, "CSV precisa ter coluna 'emotion'"

        # coluna de estilo (se existir)
        self.style_col = None
        for c in ["art_style", "style", "genre", "movement"]:
            if c in self.df.columns:
                self.style_col = c
                break

    def _encode_caption(self, text):
        toks = [SOS] + [t for t in tokenize(str(text))][:self.max_len-2] + [EOS]
        ids = [self.stoi.get(t, self.stoi[UNK]) for t in toks]
        if len(ids) < self.max_len:
            ids += [self.stoi[PAD]] * (self.max_len - len(ids))
        return torch.tensor(ids, dtype=torch.long)

    def _resolve_path(self, painting_name, style):
        """
        Resolve caminho assumindo que 'style' é a pasta e 'painting' pode
        ou não ter extensão.
        """
        base = os.path.join(ARTEMIS_IMAGES_DIR, str(style), str(painting_name))

        # se já vem com extensão e existe, retorna
        if os.path.splitext(base)[1].lower() in EXTS and os.path.isfile(base):
            return base

        # se não tem extensão, tente as comuns
        root, ext = os.path.splitext(base)
        if ext == "" or not os.path.isfile(base):
            for e in EXTS:
                cand = root + e
                if os.path.isfile(cand):
                    return cand

        # não achou
        return None


    def __len__(self): 
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        style = row[self.style_col] if self.style_col else None
        img_path = self._resolve_path(row["painting"], style=style)

        if style is None:
            raise KeyError("CSV precisa ter coluna 'art_style' (pasta).")
        img_path = self._resolve_path(row["painting"], style)

        if img_path is None:
            if self.log_missing:
                self.missing.append(str(row["painting"]))
            # cria imagem dummy preta para não quebrar o batch
            x = torch.zeros(3, 224, 224)
        else:
            img = Image.open(img_path).convert("RGB")
            x = self.tf(img)

        emo = normalize_emotion(row["emotion"])
        y_emo = torch.tensor(EMO2ID[emo], dtype=torch.long)
        cap = row[self.caption_col] if self.caption_col in row else ""
        y_cap = self._encode_caption(cap)

        if self.log_missing and (len(self.missing) and (len(self.missing) % 1000 == 0) and i == len(self.df)-1):
            # salva periodicamente ao final
            pass

        return x, y_emo, y_cap

    def save_missing_log(self, split):
        if self.log_missing and self.missing:
            out = os.path.join(PROJECT_SPLITS_DIR, f"missing_{split}.txt")
            with open(out, "w", encoding="utf-8") as f:
                f.write("\n".join(self.missing))
            print(f"[warn] {len(self.missing)} imagens não encontradas. Log: {out}")
