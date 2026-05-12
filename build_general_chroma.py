
import os
import shutil
from pathlib import Path

import pandas as pd
import chromadb
from sentence_transformers import SentenceTransformer

BASE_DIR = Path(__file__).resolve().parent
CHUNKS_PATH = BASE_DIR / "data" / "chunks.parquet"
CHROMA_PATH = BASE_DIR / "chroma_db"
# COLLECTION_NAME = "agrigenius_20260209_185402"
COLLECTION_NAME = "agrigenius_20260511_201239"
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"

if not CHUNKS_PATH.exists():
    raise FileNotFoundError(f"chunks.parquet not found at: {CHUNKS_PATH}")

print("Loading chunks from:", CHUNKS_PATH)
chunks_df = pd.read_parquet(CHUNKS_PATH)
chunks_df["chunk_text"] = chunks_df["chunk_text"].fillna("").astype(str)
chunks_df = chunks_df[chunks_df["chunk_text"].str.strip() != ""].copy()

print("Total chunks:", len(chunks_df))

# Remove old DB if it exists
if CHROMA_PATH.exists():
    print("Removing old chroma_db...")
    shutil.rmtree(CHROMA_PATH)

print("Creating new Chroma DB at:", CHROMA_PATH)
client = chromadb.PersistentClient(path=str(CHROMA_PATH))
collection = client.create_collection(name=COLLECTION_NAME)

docs = chunks_df["chunk_text"].astype(str).tolist()
ids = chunks_df["chunk_id"].astype(str).tolist()

metadatas = []
for _, row in chunks_df.iterrows():
    metadatas.append({
        "source_type": str(row.get("source_type", "")),
        "source_name": str(row.get("source_name", "")),
        "file_name": str(row.get("file_name", "")),
        "chunk_index_in_file": int(row.get("chunk_index_in_file", 0)),
        "chunk_words": int(row.get("chunk_words", 0)),
        "chunk_chars": int(row.get("chunk_chars", 0)),
    })

print("Loading embedding model:", EMBED_MODEL_NAME)
model = SentenceTransformer(EMBED_MODEL_NAME)

batch_size = 64
for start in range(0, len(docs), batch_size):
    end = min(start + batch_size, len(docs))

    batch_docs = docs[start:end]
    batch_ids = ids[start:end]
    batch_metas = metadatas[start:end]

    batch_embeddings = model.encode(batch_docs, show_progress_bar=False).tolist()

    collection.add(
        ids=batch_ids,
        documents=batch_docs,
        metadatas=batch_metas,
        embeddings=batch_embeddings
    )

    print(f"Indexed {end}/{len(docs)}")

print("\nDone.")
print("Collection:", COLLECTION_NAME)
print("Total vectors:", collection.count())
print("Saved at:", CHROMA_PATH)