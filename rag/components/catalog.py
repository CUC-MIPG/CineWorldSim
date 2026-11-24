from __future__ import annotations

import json
from pathlib import Path
from typing import Any

DEFAULT_CATALOG_PATH = Path("rag/resources/books.json")


class CatalogError(ValueError):
    """Invalid catalog structure or missing book id."""


def _normalize_book(raw: dict[str, Any]) -> dict[str, Any]:
    required = ["id", "title", "pdf_path", "chunks_path"]
    for key in required:
        if key not in raw or not raw[key]:
            raise CatalogError(f"Book entry missing required field: {key}")

    book = dict(raw)
    book.setdefault("language", "en")
    book.setdefault("chunk_size", 900)
    book.setdefault("chunk_overlap", 150)
    book.setdefault("strip_footer_notes", False)
    book.setdefault("retriever", "bm25")
    book.setdefault("bm25_preprocess", "default")
    book.setdefault("embedding_model", "sentence-transformers/all-MiniLM-L6-v2")
    book.setdefault("retrieval_k", 5)
    book.setdefault("source_preview_chars", 700)
    book.setdefault("answer_snippet_chars", 700)
    book.setdefault("drop_if_contains", [])
    return book


def load_catalog(catalog_path: Path = DEFAULT_CATALOG_PATH) -> dict[str, dict[str, Any]]:
    data = json.loads(catalog_path.read_text(encoding="utf-8"))
    raw_books = data.get("books", [])
    if not raw_books:
        raise CatalogError(f"No books found in catalog: {catalog_path}")

    books: dict[str, dict[str, Any]] = {}
    for raw in raw_books:
        book = _normalize_book(raw)
        book_id = book["id"]
        if book_id in books:
            raise CatalogError(f"Duplicated book id in catalog: {book_id}")
        books[book_id] = book
    return books


def list_books(catalog_path: Path = DEFAULT_CATALOG_PATH) -> list[dict[str, Any]]:
    books = load_catalog(catalog_path)
    return [books[k] for k in sorted(books.keys())]


def get_book(book_id: str, catalog_path: Path = DEFAULT_CATALOG_PATH) -> dict[str, Any]:
    books = load_catalog(catalog_path)
    if book_id not in books:
        known = ", ".join(sorted(books.keys()))
        raise CatalogError(f"Unknown book_id='{book_id}'. Available: {known}")
    return books[book_id]
