from __future__ import annotations

import os
import re
from typing import TypedDict

from dotenv import load_dotenv
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph

try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:  # Backward-compatible fallback.
    from langchain_community.embeddings import HuggingFaceEmbeddings


class RAGState(TypedDict):
    query: str
    retrieved_docs: list[Document]
    answer: str


def _cjk_preprocess(text: str) -> list[str]:
    return re.findall(r"[\u4e00-\u9fff]|[a-zA-Z0-9_]+", text.lower())


def _build_retriever(
    chunks: list[Document],
    k: int = 5,
    preprocess_mode: str = "default",
) -> BM25Retriever:
    preprocess_func = _cjk_preprocess if preprocess_mode == "cjk" else None
    if preprocess_func is None:
        retriever = BM25Retriever.from_documents(chunks)
    else:
        retriever = BM25Retriever.from_documents(chunks, preprocess_func=preprocess_func)
    retriever.k = k
    return retriever


def _build_vector_retriever(
    chunks: list[Document],
    k: int = 5,
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
):
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
    store = InMemoryVectorStore(embeddings)
    store.add_documents(chunks)
    return store.as_retriever(search_kwargs={"k": k})


def _build_llm(model: str, temperature: float = 0.2) -> ChatOpenAI:
    load_dotenv(override=False)
    api_key = os.getenv("RAG_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "RAG_API_KEY (or OPENAI_API_KEY fallback) is required when --use-llm is enabled."
        )

    base_url = (
        os.getenv("RAG_BASE_URL")
        or os.getenv("OPENAI_BASE_URL")
        or os.getenv("OPENAI_API_BASE")
    )
    selected_model = model or os.getenv("RAG_MODEL_NAME") or os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    kwargs = {
        "model": selected_model,
        "temperature": temperature,
        "api_key": api_key,
    }
    if base_url:
        kwargs["base_url"] = base_url
    return ChatOpenAI(**kwargs)


def _format_context(docs: list[Document]) -> str:
    rows = []
    for idx, doc in enumerate(docs, start=1):
        page = doc.metadata.get("page", "?")
        source = doc.metadata.get("source", "unknown")
        rows.append(f"[Doc {idx} | p.{page} | {source}]\n{doc.page_content}")
    return "\n\n".join(rows)


def _extractive_answer_with_config(
    query: str,
    docs: list[Document],
    snippet_chars: int = 320,
) -> str:
    if not docs:
        return "No relevant context found in the indexed PDF chunks."

    snippets = []
    for i, doc in enumerate(docs[:3], start=1):
        page = doc.metadata.get("page", "?")
        text = " ".join(doc.page_content.split())
        snippets.append(f"{i}. (page {page}) {text[:snippet_chars]}")
    joined = "\n".join(snippets)

    return (
        f"Query: {query}\n"
        "Top evidence snippets:\n"
        f"{joined}\n"
        "\nThis is extractive mode. Use --use-llm for generated synthesis."
    )


def build_rag_app(
    chunks: list[Document],
    k: int = 5,
    retriever_type: str = "bm25",
    bm25_preprocess: str = "default",
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    answer_snippet_chars: int = 320,
    use_llm: bool = False,
    model: str = "gpt-4o-mini",
):
    if retriever_type == "vector":
        retriever = _build_vector_retriever(
            chunks=chunks,
            k=k,
            embedding_model=embedding_model,
        )
    else:
        retriever = _build_retriever(chunks, k=k, preprocess_mode=bm25_preprocess)

    llm = _build_llm(model=model) if use_llm else None

    def retrieve_node(state: RAGState) -> RAGState:
        docs = retriever.invoke(state["query"])
        return {"retrieved_docs": docs}

    def generate_node(state: RAGState) -> RAGState:
        docs = state.get("retrieved_docs", [])
        query = state["query"]

        if llm is None:
            answer = _extractive_answer_with_config(
                query=query,
                docs=docs,
                snippet_chars=answer_snippet_chars,
            )
            return {"answer": answer}

        context = _format_context(docs)
        prompt = (
            "You are helping with film narratology research. "
            "Answer only from the provided context. "
            "If the context is insufficient, explicitly say so.\n\n"
            f"Question: {query}\n\n"
            f"Context:\n{context}\n"
        )
        response = llm.invoke([HumanMessage(content=prompt)])
        return {"answer": response.content}

    graph = StateGraph(RAGState)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("generate", generate_node)
    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", END)
    return graph.compile()


def run_query(app, query: str) -> dict:
    return app.invoke({"query": query})
