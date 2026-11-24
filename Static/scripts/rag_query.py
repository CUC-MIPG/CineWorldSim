from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rag.components import (
    DEFAULT_CATALOG_PATH,
    build_rag_app,
    get_book,
    run_query,
)
from rag.components.indexer import build_and_save_chunks, filter_documents, load_chunks


def main() -> None:
    load_dotenv(override=False)
    env_default_model = os.getenv("RAG_MODEL_NAME") or os.getenv("OPENAI_MODEL") or "gpt-4o-mini"
    parser = argparse.ArgumentParser(description="Query a configured PDF book with RAG")
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG_PATH)
    parser.add_argument("--book-id", type=str, required=True)
    parser.add_argument("--query", type=str, required=True)
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument("--chunk-size", type=int)
    parser.add_argument("--chunk-overlap", type=int)

    parser.add_argument("--k", type=int)
    parser.add_argument("--retriever", type=str, choices=["bm25", "vector"])
    parser.add_argument("--bm25-preprocess", type=str, choices=["default", "cjk"])
    parser.add_argument("--embedding-model", type=str)

    parser.add_argument("--use-llm", action="store_true")
    parser.add_argument("--model", type=str, default=env_default_model)
    parser.add_argument("--source-preview-chars", type=int)
    parser.add_argument("--answer-snippet-chars", type=int)

    args = parser.parse_args()

    book = get_book(args.book_id, args.catalog)
    pdf_path = ROOT / book["pdf_path"]
    chunks_path = ROOT / book["chunks_path"]

    chunk_size = args.chunk_size or int(book["chunk_size"])
    chunk_overlap = args.chunk_overlap or int(book["chunk_overlap"])

    if args.rebuild or not chunks_path.exists():
        chunks = build_and_save_chunks(
            pdf_path=pdf_path,
            chunks_path=chunks_path,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            strip_footer_notes=bool(book.get("strip_footer_notes", False)),
            drop_if_contains=book.get("drop_if_contains", []),
        )
    else:
        chunks = load_chunks(chunks_path)

    chunks = filter_documents(chunks, drop_if_contains=book.get("drop_if_contains", []))

    retriever = args.retriever or book.get("retriever", "bm25")
    bm25_preprocess = args.bm25_preprocess or book.get("bm25_preprocess", "default")
    embedding_model = args.embedding_model or book.get(
        "embedding_model", "sentence-transformers/all-MiniLM-L6-v2"
    )
    k = args.k or int(book.get("retrieval_k", 5))
    source_preview_chars = args.source_preview_chars or int(book.get("source_preview_chars", 700))
    answer_snippet_chars = args.answer_snippet_chars or int(book.get("answer_snippet_chars", 700))

    app = build_rag_app(
        chunks=chunks,
        k=k,
        retriever_type=retriever,
        bm25_preprocess=bm25_preprocess,
        embedding_model=embedding_model,
        answer_snippet_chars=answer_snippet_chars,
        use_llm=args.use_llm,
        model=args.model,
    )

    result = run_query(app, args.query)

    print(f"=== Book: {book['id']} | {book['title']} ===")
    print("=== Answer ===")
    print(result.get("answer", ""))
    print("\n=== Retrieved Sources ===")
    for i, doc in enumerate(result.get("retrieved_docs", []), start=1):
        page = doc.metadata.get("page", "?")
        source = doc.metadata.get("source", "unknown")
        preview = " ".join(doc.page_content.split())[:source_preview_chars]
        print(f"{i}. page={page} source={source} | {preview}")


if __name__ == "__main__":
    main()
