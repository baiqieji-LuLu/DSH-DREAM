#!/usr/bin/env python
"""Migrate an old Chroma collection to the default sqlite-vec backend.

Usage:
  python scripts/migrate_chroma_to_sqlite.py \
    --chroma-path <old/.chroma> \
    --sqlite-path <new/agent_memory.sqlite3>
"""

from __future__ import annotations

import argparse
from pathlib import Path

from agent_memory.storage.migrate import migrate_chroma_to_sqlite


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chroma-path", required=True, type=Path)
    parser.add_argument("--sqlite-path", required=True, type=Path)
    parser.add_argument("--collection", default="agent_memory")
    args = parser.parse_args()
    count = migrate_chroma_to_sqlite(
        args.chroma_path,
        args.sqlite_path,
        collection_name=args.collection,
    )
    print(f"migrated={count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
