"""Export the ChromaDB collection (ids, documents, metadata, embeddings) to CSV for viewing.

Usage:
    python inspect_db.py                 # export everything to data/collection_export.csv
    python inspect_db.py --limit 20      # export only the first 20 rows
    python inspect_db.py --show 3        # also print the first 3 rows in the terminal

Open the CSV in VS Code (Rainbow CSV colours the columns).
"""
import argparse
import csv
import json
import os

import chromadb

DB_DIR = "./data/db/"
COLLECTION = "clinic_sops"
OUT_FILE = "./data/collection_export.csv"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--show", type=int, default=0)
    args = parser.parse_args()

    if not os.path.isdir(DB_DIR):
        raise SystemExit(f"No database at {DB_DIR}. Run generate_embeddings.py first.")

    client = chromadb.PersistentClient(path=DB_DIR)
    collection = client.get_collection(COLLECTION)
    print(f"Collection '{COLLECTION}': {collection.count()} records")

    data = collection.get(
        limit=args.limit,
        include=["documents", "metadatas", "embeddings"],
    )

    metadata_keys = sorted({k for m in data["metadatas"] for k in (m or {})})
    header = ["id", "document_text", *metadata_keys, "embedding_dim", "embedding"]

    with open(OUT_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for doc_id, text, meta, emb in zip(
            data["ids"], data["documents"], data["metadatas"], data["embeddings"]
        ):
            meta = meta or {}
            emb_list = [round(float(x), 6) for x in emb]
            writer.writerow([
                doc_id,
                text,
                *[meta.get(k, "") for k in metadata_keys],
                len(emb_list),
                json.dumps(emb_list),
            ])

    print(f"Wrote {len(data['ids'])} rows to {OUT_FILE}")

    for i in range(min(args.show, len(data["ids"]))):
        emb = data["embeddings"][i]
        print("\n" + "-" * 80)
        print(f"id:        {data['ids'][i]}")
        print(f"metadata:  {data['metadatas'][i]}")
        print(f"text:      {data['documents'][i][:300]!r}...")
        print(f"embedding: dim={len(emb)} first 8={[round(float(x), 4) for x in emb[:8]]}")


if __name__ == "__main__":
    main()
