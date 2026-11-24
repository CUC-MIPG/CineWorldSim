from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rag.components import DEFAULT_CATALOG_PATH, get_book, list_books
from rag.components.indexer import build_and_save_chunks


def _resolve_targets(book_id: str | None, build_all: bool, catalog: Path) -> list[dict]:
    if build_all:
        return list_books(catalog)
    if not book_id:
        raise ValueError("Pass --book-id or use --all")
    return [get_book(book_id, catalog)]


def main() -> None:
    parser = argparse.ArgumentParser(description="Build chunk indexes for configured PDF books")
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG_PATH)
    parser.add_argument("--book-id", type=str)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument("--chunk-size", type=int)
    parser.add_argument("--chunk-overlap", type=int)
    args = parser.parse_args()

    targets = _resolve_targets(args.book_id, args.all, args.catalog)

    for book in targets:
        pdf_path = ROOT / book["pdf_path"]
        chunks_path = ROOT / book["chunks_path"]

        if chunks_path.exists() and not args.rebuild:
            print(f"[skip] {book['id']} -> {chunks_path} (already exists)")
            continue

        chunk_size = args.chunk_size or int(book["chunk_size"])
        chunk_overlap = args.chunk_overlap or int(book["chunk_overlap"])

        chunks = build_and_save_chunks(
            pdf_path=pdf_path,
            chunks_path=chunks_path,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            strip_footer_notes=bool(book.get("strip_footer_notes", False)),
            drop_if_contains=book.get("drop_if_contains", []),
        )
        print(
            f"[ok] {book['id']} built {len(chunks)} chunks "
            f"(size={chunk_size}, overlap={chunk_overlap}) -> {chunks_path}"
        )


if __name__ == "__main__":
    main()
