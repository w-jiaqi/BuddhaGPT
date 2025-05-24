"""
Build FAISS index over CBETA scripture chunks (traditional Chinese).
    data/processed/cbeta_faiss.index
    data/processed/cbeta_passages.json
"""
import json, pathlib, numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import faiss

DATA_FILE = pathlib.Path("data/processed/cbeta_zh_trad.jsonl")
INDEX_FILE = pathlib.Path("data/processed/cbeta_faiss.index")
PASSAGE_FILE = pathlib.Path("data/processed/cbeta_passages.json")

MODEL_NAME = "thenlper/gte-large-zh"
BATCH_SIZE = 64

def main():
    print("Loading model…")
    model = SentenceTransformer(MODEL_NAME)

    ids, texts, embeddings = [], [], []

    with DATA_FILE.open() as f:
        batch_ids, batch_texts = [], []
        for line in tqdm(f, desc="Reading & batching"):
            record = json.loads(line)
            ids.append(record["id"])
            texts.append(record["trad"])

            batch_ids.append(record["id"])
            batch_texts.append(record["trad"])

            if len(batch_ids) == BATCH_SIZE:
                vecs = model.encode(batch_texts, convert_to_numpy=True)
                embeddings.append(vecs.astype("float32"))
                batch_ids.clear(), batch_texts.clear()

        # last batch
        if batch_ids:
            vecs = model.encode(batch_texts, convert_to_numpy=True)
            embeddings.append(vecs.astype("float32"))

    X = np.vstack(embeddings)
    print(f"Encoded {len(X)} passages.")

    print("Building index…")
    index = faiss.IndexHNSWFlat(X.shape[1], 64)
    index.add(X)
    faiss.write_index(index, str(INDEX_FILE))

    print("Saving metadata…")
    with PASSAGE_FILE.open("w") as f:
        json.dump({"id": ids, "text": texts}, f, ensure_ascii=False)

    print("✅ Done.")

if __name__ == "__main__":
    main()
