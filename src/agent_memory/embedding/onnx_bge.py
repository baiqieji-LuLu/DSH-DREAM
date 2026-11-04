"""BGE embeddings through ONNX Runtime and tokenizers.

This backend is intentionally independent of torch/sentence-transformers:
only ``onnxruntime`` and ``tokenizers`` are required at runtime.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

_DEFAULT_QUERY_INSTRUCTION = "为这个句子生成表示以用于检索相关文章："


class OnnxBgeEmbedding:
    """ONNX Runtime implementation of BGE query/document embeddings."""

    model_name = "BAAI/bge-small-zh-v1.5"
    dimension = 512

    def __init__(
        self,
        model_path: str | Path,
        tokenizer_path: str | Path | None = None,
        *,
        dimension: int = 512,
        max_length: int = 512,
        pooling: str = "cls",
        query_instruction: str = _DEFAULT_QUERY_INSTRUCTION,
        providers: Sequence[str] | None = None,
    ) -> None:
        try:
            import onnxruntime as ort
            from tokenizers import Tokenizer
        except ImportError as exc:  # pragma: no cover - exercised by install smoke test
            raise RuntimeError(
                "OnnxBgeEmbedding requires 'onnxruntime' and 'tokenizers'. "
                "Install them with: pip install onnxruntime tokenizers"
            ) from exc

        self.model_name = Path(model_path).stem
        self.dimension = int(dimension)
        self.max_length = int(max_length)
        self.pooling = pooling.strip().lower()
        if self.pooling not in {"cls", "mean"}:
            raise ValueError("pooling must be 'cls' or 'mean'")
        self.query_instruction = query_instruction
        self._tokenizer = Tokenizer.from_file(
            str(tokenizer_path or Path(model_path).parent / "tokenizer.json")
        )
        self._tokenizer.enable_truncation(max_length=self.max_length)
        self._tokenizer.enable_padding(
            pad_id=0,
            pad_token="[PAD]",
            length=None,
        )
        available = ort.get_available_providers()
        selected = [
            provider
            for provider in (providers or ["CPUExecutionProvider"])
            if provider in available
        ] or ["CPUExecutionProvider"]
        self._session = ort.InferenceSession(
            str(model_path),
            providers=selected,
        )
        self._input_names = {item.name for item in self._session.get_inputs()}

    def _encode(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)
        encodings = self._tokenizer.encode_batch(list(texts))
        input_ids = np.asarray([item.ids for item in encodings], dtype=np.int64)
        attention = np.asarray(
            [item.attention_mask for item in encodings],
            dtype=np.int64,
        )
        feeds: dict[str, Any] = {"input_ids": input_ids}
        if "attention_mask" in self._input_names:
            feeds["attention_mask"] = attention
        if "token_type_ids" in self._input_names:
            feeds["token_type_ids"] = np.asarray(
                [item.type_ids for item in encodings],
                dtype=np.int64,
            )
        output = self._session.run(None, feeds)[0]
        hidden = np.asarray(output, dtype=np.float32)
        if hidden.ndim == 2:
            vectors = hidden
        elif self.pooling == "cls":
            # BGE's sentence-transformers pooling config is CLS, not mean.
            vectors = hidden[:, 0]
        else:
            mask = attention.astype(np.float32)[..., None]
            summed = (hidden * mask).sum(axis=1)
            counts = np.clip(mask.sum(axis=1), 1.0, None)
            vectors = summed / counts
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return (vectors / norms).astype(np.float32, copy=False)

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        return self._encode([str(text) for text in texts])

    def embed_query(self, text: str) -> np.ndarray:
        query = f"{self.query_instruction}{text}"
        return self._encode([query])[0]
