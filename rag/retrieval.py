import torch
import faiss, json, numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer

ROOT   = Path(__file__).resolve().parent.parent        # …/buddhagpt
DATA   = ROOT / "data/processed"                       # …/buddhagpt/data/processed
IDX    = DATA / "cbeta_faiss.index"
META   = DATA / "cbeta_passages.json"
MODEL = "thenlper/gte-large-zh"

_sentence_model = None
_index          = None
_passages       = None

def _load():
    global _sentence_model, _index, _passages
    if _sentence_model is None:
        _sentence_model = SentenceTransformer(MODEL, device="cuda" if torch.cuda.is_available() else "cpu")
    if _index is None:
        _index = faiss.read_index(str(IDX))
    if _passages is None:
        _passages = json.load(META.open())
    return _sentence_model, _index, _passages

def get_top(query: str, k: int = 5):
    model, index, passages = _load()
    qv = model.encode([query])[0].astype("float32")
    D, I = index.search(np.expand_dims(qv, 0), k)
    results = []
    for rank, idx in enumerate(I[0]):
        results.append({
            "rank":  rank + 1,
            "id":    passages["id"][idx],
            "score": float(D[0][rank]),
            "text":  passages["text"][idx]
        })
    return results

if __name__ == "__main__":
    import sys, textwrap
    if len(sys.argv) < 2:
        print("Usage: python -m buddhagpt.retrieval \"<question>\" [k]")
        sys.exit(0)

    query = sys.argv[1]
    k     = int(sys.argv[2]) if len(sys.argv) > 2 else 5

    for hit in get_top(query, k):
        print(f"\n[{hit['id']}]  (score={hit['score']:.3f})\n"
              f"{textwrap.shorten(hit['text'], width=120, placeholder='…')}")

