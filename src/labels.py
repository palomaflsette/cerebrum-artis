EMOTIONS = [
    "awe", "contentment", "amusement", "excitement",
    "sadness", "fear", "anger", "disgust", "something else"
]
EMO2ID = {e:i for i, e in enumerate(EMOTIONS)}
ID2EMO = {i:e for e,i in EMO2ID.items()}

def normalize_emotion(s: str) -> str:
    e = str(s).strip().lower().replace("_", " ").replace("-", " ")
    # normalizações comuns
    if e in {"other", "others", "unknown"}:
        e = "something else"
    return e
