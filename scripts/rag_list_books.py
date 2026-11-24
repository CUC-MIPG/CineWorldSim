from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rag.components import DEFAULT_CATALOG_PATH, list_books


def main() -> None:
    parser = argparse.ArgumentParser(description="List all configured PDF books for RAG")
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG_PATH)
    args = parser.parse_args()

    books = list_books(args.catalog)
    print(f"Catalog: {args.catalog}")
    print(f"Total books: {len(books)}")
    for book in books:
        print(
            f"- {book['id']} | {book['title']} | "
            f"pdf={book['pdf_path']} | chunks={book['chunks_path']}"
        )


if __name__ == "__main__":
    main()
