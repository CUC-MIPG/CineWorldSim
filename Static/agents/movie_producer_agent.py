from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from langchain_core.documents import Document

from .client import make_client, pick_model
from .movie_producer_prompts import build_prompt_messages
from rag.components import DEFAULT_CATALOG_PATH, list_books
from rag.components.indexer import load_chunks, filter_documents
from rag.components.graph_rag import _build_retriever, _build_vector_retriever  # noqa: SLF001


@dataclass
class BookStore:
    book_id: str
    title: str
    docs: list[Document]
    retriever_type: str
    bm25_preprocess: str
    embedding_model: str


_TOKEN_RE = re.compile(r"[\u4e00-\u9fff]|[a-zA-Z0-9_]+")
_INDEX_HINT_RE = re.compile(
    r"(index of names|index|bibliography|references|table of contents|目录|索引)",
    re.IGNORECASE,
)
_AD_HINT_RE = re.compile(r"(join us|youtube|facebook|twitter|www\.|isbn)", re.IGNORECASE)


class MovieProducerAgent:
    """
    Unified multi-knowledge-base RAG agent for film production research.
    - Mounts all book indexes from rag/resources/books.json
    - Provides retrieval/query functions
    - Supports prompt templates for production tasks
    """

    def __init__(
        self,
        root: Path | None = None,
        catalog_path: Path = DEFAULT_CATALOG_PATH,
        model: str | None = None,
    ):
        self.root = root or Path(__file__).resolve().parents[1]
        self.catalog_path = catalog_path
        self.client = make_client(scope="RAG")
        self.model = model or pick_model("gpt-5.5", scope="RAG")

        self.book_stores: dict[str, BookStore] = {}
        self._mount_all_books()

    def _mount_all_books(self) -> None:
        books = list_books(self.catalog_path)
        for book in books:
            chunks_path = self.root / book["chunks_path"]
            if not chunks_path.exists():
                continue

            docs = load_chunks(chunks_path)
            docs = filter_documents(docs, drop_if_contains=book.get("drop_if_contains", []))
            self.book_stores[book["id"]] = BookStore(
                book_id=book["id"],
                title=book["title"],
                docs=docs,
                retriever_type=book.get("retriever", "bm25"),
                bm25_preprocess=book.get("bm25_preprocess", "default"),
                embedding_model=book.get("embedding_model", "sentence-transformers/all-MiniLM-L6-v2"),
            )

    @staticmethod
    def _tokenize(text: str) -> set[str]:
        return set(_TOKEN_RE.findall((text or "").lower()))

    @staticmethod
    def _noise_score(text: str) -> float:
        compact = " ".join((text or "").split())
        if not compact:
            return 1.0

        score = 0.0
        if len(compact) < 48:
            score += 0.45
        if _INDEX_HINT_RE.search(compact):
            score += 0.55
        if _AD_HINT_RE.search(compact):
            score += 0.65

        commas = compact.count(",") + compact.count("，")
        digits = sum(ch.isdigit() for ch in compact)
        alpha = sum(ch.isalpha() for ch in compact)
        upper = sum(ch.isupper() for ch in compact)
        if commas >= 12 and digits >= 8:
            score += 0.35
        if alpha > 0 and (upper / alpha) > 0.45 and len(compact) > 120:
            score += 0.2
        return min(score, 1.0)

    @staticmethod
    def _is_structural_noise(text: str) -> bool:
        compact = " ".join((text or "").split())
        if not compact:
            return True
        low = compact.lower()

        if _AD_HINT_RE.search(low) and len(compact) < 500:
            return True

        commas = compact.count(",") + compact.count("，")
        semis = compact.count(";") + compact.count("；")
        digits = sum(ch.isdigit() for ch in compact)
        token_like = len(_TOKEN_RE.findall(low))

        if _INDEX_HINT_RE.search(low) and (commas >= 8 or digits >= 8):
            return True

        # Typical name-title index lines: dense punctuation + many numbers + weak sentence rhythm.
        if token_like > 0:
            punct_density = (commas + semis) / token_like
            digit_density = digits / max(len(compact), 1)
            if punct_density > 0.16 and digit_density > 0.06 and len(compact) > 140:
                return True

        return False

    def _relevance_score(self, query: str, doc: Document) -> float:
        q_tokens = self._tokenize(query)
        text = doc.page_content or ""
        d_tokens = self._tokenize(text)
        if not q_tokens or not d_tokens:
            return 0.0
        overlap = len(q_tokens & d_tokens)
        base = overlap / (len(q_tokens) ** 0.5 * len(d_tokens) ** 0.25)

        text_l = text.lower()
        phrase_bonus = 0.0
        for tok in q_tokens:
            if len(tok) >= 3 and tok in text_l:
                phrase_bonus += 0.02
        penalty = self._noise_score(text)
        return max(0.0, base + phrase_bonus - penalty)

    def _deduplicate_docs(self, docs: list[Document]) -> list[Document]:
        seen: set[tuple[Any, Any, str]] = set()
        out: list[Document] = []
        for d in docs:
            meta = d.metadata or {}
            book_id = meta.get("book_id")
            page = meta.get("page")
            sig = " ".join((d.page_content or "").split())[:220]
            key = (book_id, page, sig)
            if key in seen:
                continue
            seen.add(key)
            out.append(d)
        return out

    def mounted_books(self) -> list[dict[str, Any]]:
        return [
            {
                "book_id": b.book_id,
                "title": b.title,
                "chunks": len(b.docs),
                "retriever_type": b.retriever_type,
                "bm25_preprocess": b.bm25_preprocess,
                "embedding_model": b.embedding_model,
            }
            for b in sorted(self.book_stores.values(), key=lambda x: x.book_id)
        ]

    def list_knowledge_bases(self) -> list[dict[str, Any]]:
        return self.mounted_books()

    def _retrieve_from_book(
        self,
        store: BookStore,
        query: str,
        k: int = 4,
        retriever_type: str | None = None,
    ) -> list[Document]:
        rtype = retriever_type or store.retriever_type
        if rtype == "vector":
            retriever = _build_vector_retriever(
                chunks=store.docs,
                k=k,
                embedding_model=store.embedding_model,
            )
        else:
            retriever = _build_retriever(
                chunks=store.docs,
                k=k,
                preprocess_mode=store.bm25_preprocess,
            )
        docs = retriever.invoke(query)
        for d in docs:
            d.metadata = dict(d.metadata or {})
            d.metadata["book_id"] = store.book_id
            d.metadata["book_title"] = store.title
        return docs

    def retrieve_knowledge(
        self,
        query: str,
        k_per_book: int = 3,
        top_k_total: int = 12,
        book_ids: list[str] | None = None,
        retriever_type: str | None = None,
    ) -> list[Document]:
        selected = (
            [self.book_stores[b] for b in book_ids if b in self.book_stores]
            if book_ids
            else list(self.book_stores.values())
        )
        all_docs: list[Document] = []
        for store in selected:
            all_docs.extend(
                self._retrieve_from_book(
                    store=store,
                    query=query,
                    k=k_per_book,
                    retriever_type=retriever_type,
                )
            )

        deduped = self._deduplicate_docs(all_docs)
        ranked = sorted(deduped, key=lambda d: self._relevance_score(query, d), reverse=True)
        filtered = [
            d
            for d in ranked
            if self._noise_score(d.page_content or "") < 0.95
            and not self._is_structural_noise(d.page_content or "")
        ]
        if not filtered:
            filtered = [d for d in ranked if self._noise_score(d.page_content or "") < 0.98]
        return filtered[:top_k_total]

    @staticmethod
    def format_context(docs: list[Document], max_chars_per_doc: int = 700) -> str:
        rows = []
        for i, d in enumerate(docs, start=1):
            meta = d.metadata or {}
            bid = meta.get("book_id", "unknown_book")
            page = meta.get("page", "?")
            source = meta.get("source", "unknown_source")
            snippet = " ".join((d.page_content or "").split())[:max_chars_per_doc]
            rows.append(f"[{i}] [{bid} p.{page}] ({source}) {snippet}")
        return "\n".join(rows)

    def query(
        self,
        question: str,
        *,
        template: str = "general_research",
        constraints: str = "None",
        k_per_book: int = 3,
        top_k_total: int = 12,
        book_ids: list[str] | None = None,
        retriever_type: str | None = None,
        max_chars_per_doc: int = 700,
    ) -> dict[str, Any]:
        docs = self.retrieve_knowledge(
            query=question,
            k_per_book=k_per_book,
            top_k_total=top_k_total,
            book_ids=book_ids,
            retriever_type=retriever_type,
        )
        context = self.format_context(docs, max_chars_per_doc=max_chars_per_doc)
        messages = build_prompt_messages(
            template_name=template,
            question=question,
            context=context,
            constraints=constraints,
        )
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
            )
            answer = resp.choices[0].message.content if resp.choices else ""
        except Exception as exc:
            # Offline/network fallback: provide structured extractive answer.
            lines = [f"[offline-extractive-fallback] {type(exc).__name__}: {exc}"]
            lines.append(f"Question: {question}")
            lines.append(f"Template: {template}")
            lines.append("Top Evidence:")
            for i, d in enumerate(docs[: min(8, len(docs))], start=1):
                bid = d.metadata.get("book_id", "unknown_book")
                page = d.metadata.get("page", "?")
                snippet = " ".join((d.page_content or "").split())[:320]
                lines.append(f"{i}. [{bid} p.{page}] {snippet}")
            answer = "\n".join(lines)
        return {
            "question": question,
            "template": template,
            "constraints": constraints,
            "answer": answer,
            "retrieved_docs": [
                {
                    "book_id": d.metadata.get("book_id"),
                    "book_title": d.metadata.get("book_title"),
                    "page": d.metadata.get("page"),
                    "source": d.metadata.get("source"),
                    "preview": " ".join((d.page_content or "").split())[:max_chars_per_doc],
                }
                for d in docs
            ],
        }

    def query_with_template(
        self,
        template_name: str,
        question: str,
        **kwargs: Any,
    ) -> dict[str, Any]:
        return self.query(question=question, template=template_name, **kwargs)

    def query_general_research(self, question: str, **kwargs: Any) -> dict[str, Any]:
        return self.query(question, template="general_research", **kwargs)

    def query_story_development(self, question: str, **kwargs: Any) -> dict[str, Any]:
        return self.query(question, template="story_development", **kwargs)

    def query_scene_design(self, question: str, **kwargs: Any) -> dict[str, Any]:
        return self.query(question, template="scene_design", **kwargs)

    def query_shot_design(self, question: str, **kwargs: Any) -> dict[str, Any]:
        return self.query(question, template="shot_design", **kwargs)

    def save_query_result(self, result: dict[str, Any], out_path: Path) -> None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
