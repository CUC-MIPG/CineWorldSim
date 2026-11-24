from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

_FOOTNOTE_MARKER_RE = re.compile(
    r"^\s*(?:\[\d+\]|\(\d+\)|\d+[\.、\)]|[①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮⑯⑰⑱⑲⑳])\s*"
)
_INLINE_FOOTNOTE_MARK_RE = re.compile(r"[①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮⑯⑰⑱⑲⑳]")
_CITATION_HINT_RE = re.compile(
    r"(?:参见|载|出版社|出版|版|ISBN|http|www\.|Press|University|Routledge|Oxford|NewYork)",
    re.IGNORECASE,
)
_PURE_PAGE_NUMBER_RE = re.compile(r"^\s*\d{1,4}\s*$")


def _strip_footer_notes(text: str) -> str:
    lines = [ln.rstrip() for ln in text.splitlines()]
    if not lines:
        return text

    while lines and _PURE_PAGE_NUMBER_RE.match(lines[0]):
        lines.pop(0)
    while lines and _PURE_PAGE_NUMBER_RE.match(lines[-1]):
        lines.pop()

    if not lines:
        return ""

    start_idx = None
    threshold = max(1, int(len(lines) * 0.55))
    for i, line in enumerate(lines):
        if i < threshold:
            continue
        if _FOOTNOTE_MARKER_RE.match(line):
            tail_block = lines[i:]
            if any(_CITATION_HINT_RE.search(x) for x in tail_block):
                start_idx = i
                break

    if start_idx is not None:
        lines = lines[:start_idx]

    cleaned = "\n".join(ln for ln in lines if ln.strip()).strip()
    cleaned = _INLINE_FOOTNOTE_MARK_RE.sub("", cleaned)
    return cleaned


def filter_documents(
    docs: Iterable[Document],
    drop_if_contains: list[str] | None = None,
) -> list[Document]:
    patterns = [p for p in (drop_if_contains or []) if p]
    if not patterns:
        return list(docs)

    cleaned: list[Document] = []
    for doc in docs:
        text = doc.page_content or ""
        if any(p in text for p in patterns):
            continue
        cleaned.append(doc)
    return cleaned


def extract_documents_from_pdf(
    pdf_path: Path,
    strip_footer_notes: bool = False,
    drop_if_contains: list[str] | None = None,
) -> list[Document]:
    reader = PdfReader(str(pdf_path), strict=False)
    docs: list[Document] = []

    for page_idx, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if strip_footer_notes:
            text = _strip_footer_notes(text)
        if not text:
            continue
        docs.append(
            Document(
                page_content=text,
                metadata={"source": str(pdf_path), "page": page_idx},
            )
        )

    return filter_documents(docs, drop_if_contains=drop_if_contains)


def split_documents(
    docs: Iterable[Document],
    chunk_size: int = 900,
    chunk_overlap: int = 150,
) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", "。", "！", "？", "；", ". ", " "],
    )
    return splitter.split_documents(list(docs))


def save_chunks(chunks: Iterable[Document], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for chunk in chunks:
            row = {
                "page_content": chunk.page_content,
                "metadata": dict(chunk.metadata or {}),
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_chunks(chunks_path: Path) -> list[Document]:
    chunks: list[Document] = []
    with chunks_path.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            chunks.append(
                Document(
                    page_content=row.get("page_content", ""),
                    metadata=row.get("metadata", {}),
                )
            )
    return chunks


def build_and_save_chunks(
    pdf_path: Path,
    chunks_path: Path,
    chunk_size: int = 900,
    chunk_overlap: int = 150,
    strip_footer_notes: bool = False,
    drop_if_contains: list[str] | None = None,
) -> list[Document]:
    docs = extract_documents_from_pdf(
        pdf_path=pdf_path,
        strip_footer_notes=strip_footer_notes,
        drop_if_contains=drop_if_contains,
    )
    chunks = split_documents(docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    save_chunks(chunks, chunks_path)
    return chunks
