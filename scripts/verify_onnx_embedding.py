#!/usr/bin/env python
"""Compare ONNX BGE embeddings with sentence-transformers on 100 texts.

Usage:
  python scripts/verify_onnx_embedding.py \
    --onnx path/to/model.onnx \
    --tokenizer path/to/tokenizer.json \
    --model-dir path/to/bge-small-zh-v1.5

Exit code 0 only when every cosine similarity is > 0.999.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", required=True, type=Path)
    parser.add_argument("--tokenizer", required=True, type=Path)
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--threshold", type=float, default=0.999)
    parser.add_argument("--pooling", choices=("cls", "mean"), default="cls")
    args = parser.parse_args()

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("sentence-transformers is required for comparison", file=sys.stderr)
        return 2

    try:
        from agent_memory.embedding.onnx_bge import OnnxBgeEmbedding
    except ImportError:
        print("run from the repository root with PYTHONPATH=src", file=sys.stderr)
        return 2

    onnx = OnnxBgeEmbedding(
        args.onnx,
        args.tokenizer,
        pooling=args.pooling,
    )
    reference = SentenceTransformer(str(args.model_dir))

    texts = [
        f"这是第{i}条测试文本，包含中文、English and numbers {i}。"
        for i in range(max(1, args.limit))
    ]
    actual = onnx.embed_documents(texts)
    expected = reference.encode(
        texts,
        normalize_embeddings=True,
        batch_size=32,
        show_progress_bar=False,
    )
    similarities = np.sum(actual * expected, axis=1)
    print(f"shape: {actual.shape} vs {expected.shape}")
    print(
        "cosine: "
        f"min={float(similarities.min()):.8f} "
        f"mean={float(similarities.mean()):.8f} "
        f"max={float(similarities.max()):.8f}"
    )
    failures = int((similarities <= args.threshold).sum())
    print(f"threshold {args.threshold}: failures={failures}/{len(similarities)}")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
