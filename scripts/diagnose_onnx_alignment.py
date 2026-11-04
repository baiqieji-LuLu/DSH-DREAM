#!/usr/bin/env python
"""Diagnose ONNX BGE vs sentence-transformers on one text.

Usage:
  python scripts/diagnose_onnx_alignment.py \
    --onnx model.onnx --tokenizer tokenizer.json --model-dir bge-dir \
    --text "你好"
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float32).reshape(-1)
    b = np.asarray(b, dtype=np.float32).reshape(-1)
    return float(
        np.dot(a / np.linalg.norm(a), b / np.linalg.norm(b))
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", required=True, type=Path)
    parser.add_argument("--tokenizer", required=True, type=Path)
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--text", default="你好")
    args = parser.parse_args()

    from tokenizers import Tokenizer
    from transformers import AutoTokenizer

    from agent_memory.embedding.onnx_bge import OnnxBgeEmbedding

    raw_tokenizer = Tokenizer.from_file(str(args.tokenizer))
    raw = raw_tokenizer.encode(args.text)
    hf = AutoTokenizer.from_pretrained(str(args.model_dir))
    hf_ids = hf(args.text, return_attention_mask=True)

    print("tokenizers ids       :", raw.ids)
    print("tokenizers attention :", raw.attention_mask)
    print("HF ids               :", hf_ids["input_ids"])
    print("HF attention         :", hf_ids["attention_mask"])
    print("ids equal            :", raw.ids == hf_ids["input_ids"])

    from sentence_transformers import SentenceTransformer

    st = SentenceTransformer(str(args.model_dir))
    onnx = OnnxBgeEmbedding(args.onnx, args.tokenizer, pooling="cls")

    a_doc = onnx.embed_documents([args.text])[0]
    a_query = onnx.embed_query(args.text)
    s_doc = st.encode([args.text], normalize_embeddings=True)[0]
    s_query = st.encode(
        ["为这个句子生成表示以用于检索相关文章：" + args.text],
        normalize_embeddings=True,
    )[0]
    print("onnx doc vs st doc   :", cosine(a_doc, s_doc))
    print("onnx qry vs st qry   :", cosine(a_query, s_query))
    print("onnx doc vs onnx qry :", cosine(a_doc, a_query))
    print("st doc vs st qry     :", cosine(s_doc, s_query))

    # Compare the raw ONNX session with the ST backbone on the exact same ids.
    import torch
    from transformers import AutoModel

    enc = raw
    feed = {
        "input_ids": np.asarray([enc.ids], dtype=np.int64),
        "attention_mask": np.asarray([enc.attention_mask], dtype=np.int64),
        "token_type_ids": np.asarray([enc.type_ids], dtype=np.int64),
    }
    session = onnx._session
    raw_out = session.run(None, feed)[0][0, 0]
    backbone = AutoModel.from_pretrained(str(args.model_dir)).eval()
    with torch.no_grad():
        hidden = backbone(
            input_ids=torch.tensor(feed["input_ids"]),
            attention_mask=torch.tensor(feed["attention_mask"]),
            token_type_ids=torch.tensor(feed["token_type_ids"]),
        ).last_hidden_state[0, 0].numpy()
    print("ONNX raw vs HF backbone:", cosine(raw_out, hidden))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
