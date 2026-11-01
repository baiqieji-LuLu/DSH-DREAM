"""End-to-end test for the Host turn bridge writing through Python."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import uuid
from pathlib import Path

import chromadb

ROOT = Path(__file__).resolve().parents[1]
BRIDGE = ROOT / "dsh_plugin" / "turn_bridge.py"
TEST_TMP = ROOT / ".test-tmp"


def test_turn_bridge_writes_one_timestamped_turn() -> None:
    TEST_TMP.mkdir(exist_ok=True)
    db_path = TEST_TMP / f"bridge-{uuid.uuid4().hex}"
    config_path = TEST_TMP / f"bridge-config-{uuid.uuid4().hex}.json"
    config = {
        "enable_memory": True,
        "rag_path": str(db_path),
        "max_window_tokens": 1500,
        "retrieval_top_k": 5,
        "embedding_provider": "local",
        "embedding_model": "",
        "embedding_api_key": "",
        "embedding_base_url": "",
        "auto_rebuild_collection": True,
        "holographic_mode": True,
    }
    config_path.write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    env = os.environ.copy()
    env["AGENT_MEMORY_CONFIG"] = str(config_path)
    env["HF_HOME"] = str(ROOT / ".hf-cache")
    env["HF_HUB_CACHE"] = str(ROOT / ".hf-cache" / "hub")
    env["HF_HUB_OFFLINE"] = "1"
    env["TRANSFORMERS_OFFLINE"] = "1"
    env["HF_HUB_DISABLE_SYMLINKS"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    payload = {
        "user_msg": "今天天气不错，想出去散步",
        "ai_msg": "很适合散步，记得穿舒服的鞋。",
    }
    try:
        proc = subprocess.run(
            [sys.executable, "-u", str(BRIDGE)],
            cwd=str(ROOT),
            env=env,
            input=json.dumps(payload, ensure_ascii=False) + "\n",
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=180,
            check=False,
        )
        result = [
            json.loads(line)
            for line in proc.stdout.splitlines()
            if line.strip().startswith("{")
        ][-1]
        assert result["ok"] is True, (proc.stdout, proc.stderr)
        assert result["fact_memory_id"] is None

        client = chromadb.PersistentClient(path=str(db_path))
        collection = client.get_collection("agent_memory")
        raw = collection.get(include=["documents", "metadatas"])
        rows = list(
            zip(
                raw.get("ids", []),
                raw.get("documents", []),
                raw.get("metadatas", []),
            )
        )
        assert len(rows) == 2
        by_source = {
            metadata["source_type"]: (document, metadata)
            for _, document, metadata in rows
        }
        assert by_source["user_message"][0] == payload["user_msg"]
        assert by_source["assistant_message"][0] == payload["ai_msg"]
        assert len(
            {
                int(metadata["timestamp_ms"])
                for _, metadata in by_source.values()
            }
        ) == 1
    finally:
        config_path.unlink(missing_ok=True)
        shutil.rmtree(db_path, ignore_errors=True)
