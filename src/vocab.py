import json, re
from collections import Counter

PAD, SOS, EOS, UNK = "<pad>", "<sos>", "<eos>", "<unk>"

def tokenize(txt: str):
    return re.findall(r"[A-Za-zÀ-ÿ']+|\d+|[.,!?;:()\-]", txt.lower())

def build_vocab(caption_series, min_freq=5):
    cnt = Counter()
    for t in caption_series.astype(str):
        cnt.update(tokenize(t))
    itos = [PAD, SOS, EOS, UNK]
    for w, f in cnt.most_common():
        if f >= min_freq and w not in itos:
            itos.append(w)
    stoi = {w:i for i,w in enumerate(itos)}
    return {"itos": itos, "stoi": stoi}

def save_vocab(voc, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"itos": voc["itos"]}, f, ensure_ascii=False)

def load_vocab(path):
    with open(path, "r", encoding="utf-8") as f:
        itos = json.load(f)["itos"]
    stoi = {w:i for i,w in enumerate(itos)}
    return {"itos": itos, "stoi": stoi}
