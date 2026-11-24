"""Reusable RAG components for multi-book PDF retrieval."""

from .catalog import DEFAULT_CATALOG_PATH, get_book, load_catalog, list_books
from .graph_rag import build_rag_app, run_query
from .indexer import build_and_save_chunks, filter_documents, load_chunks

__all__ = [
    "DEFAULT_CATALOG_PATH",
    "get_book",
    "load_catalog",
    "list_books",
    "build_rag_app",
    "run_query",
    "build_and_save_chunks",
    "filter_documents",
    "load_chunks",
]
