#!/usr/bin/env python3
"""MSCE — Step 5: pre-compute skill abstract embeddings + write index.

Input:
    --skill-bank   skill_bank.jsonl from crystallize
Output:
    --output       skill_index.jsonl  (same records + 'embedding' field)
    --emb-npy      skill_embeddings.npy  (N, embedding_dim matrix in skill_id order)
    --emb-ids      skill_ids.json        (list of skill_ids in same order)
"""
from __future__ import annotations
import argparse, json, os, time, sys
from pathlib import Path
import numpy as np

try:
    from .clients import embedding_batch as provider_embedding_batch
except ImportError:  # pragma: no cover - allows direct script execution
    from clients import embedding_batch as provider_embedding_batch


def embed_batch(texts, batch=16):
    return provider_embedding_batch(texts, batch_size=batch)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skill-bank", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--emb-npy", required=True)
    ap.add_argument("--emb-ids", required=True)
    args = ap.parse_args()

    skills = [json.loads(l) for l in open(args.skill_bank)]
    print(f"Loaded {len(skills)} skills")

    texts = [s.get("abstract") or s.get("embedding_text") or s.get("summary", "")
             for s in skills]
    print("Embedding abstracts with bge-m3 ...")
    embs = embed_batch(texts)
    print(f"  embeddings: {embs.shape}")

    ids = [s["skill_id"] for s in skills]
    np.save(args.emb_npy, embs)
    with open(args.emb_ids, "w") as f:
        json.dump(ids, f)

    with open(args.output, "w") as fo:
        for s, e in zip(skills, embs):
            s["embedding"] = e.tolist()
            fo.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(f"Wrote {args.output}, {args.emb_npy}, {args.emb_ids}")


if __name__ == "__main__":
    sys.exit(main())
