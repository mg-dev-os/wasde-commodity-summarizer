#!/usr/bin/env python3
"""Inspect the local LanceDB vector store: list tables, row counts, and sample data."""

import sys
from pathlib import Path

# Add project root so we can import src
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import lancedb
from src.vector_store import get_db_path


def main():
    db_path = get_db_path()
    print(f"LanceDB path: {db_path}")
    if not db_path.exists():
        print("  (directory does not exist yet — run the app with use_maf_retrieval: true and use Search once to index a PDF)")
        return

    db = lancedb.connect(str(db_path))
    names = db.table_names()
    if not names:
        print("  No tables found.")
        return

    for name in names:
        table = db.open_table(name)
        try:
            arrow = table.to_arrow()
            count = arrow.num_rows
            rows = arrow.to_pylist()[:3]
        except Exception as e:
            print(f"\nTable: {name}  (error: {e})")
            continue
        print(f"\nTable: {name}  (rows: {count})")
        for i, row in enumerate(rows):
            text = (row.get("text") or "")[:200]
            if len(row.get("text") or "") > 200:
                text += "..."
            doc_id = row.get("doc_id", "")
            print(f"  [{i+1}] doc_id={doc_id!r}  text={text!r}")


if __name__ == "__main__":
    main()
