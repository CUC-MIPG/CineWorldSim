# RAG Workspace

This folder centralizes RAG design for all PDF books in the project.

## Structure

- `components/`: reusable LangChain + LangGraph components
- `resources/books.json`: per-book RAG configuration (catalog)
- `indexes/`: generated chunk indexes (`*.jsonl`)

## Core idea

- Every PDF is mapped to a `book_id` in `resources/books.json`
- Build pipeline uses per-book chunk strategy and cleaning rules
- Query pipeline uses per-book retrieval defaults (BM25/CJK/Vector)

Use scripts in the top-level `scripts/` folder to list books, build indexes and query.

MovieProducer agent is available at:

- `agents/movie_producer_agent.py` (all-book mounted RAG agent)
- `agents/movie_producer_prompts.py` (prompt template registry)
- `scripts/movie_producer_query.py` (CLI query entrypoint)

## RAG LLM Env Keys

When `--use-llm` is enabled, RAG reads:

- `RAG_API_KEY` (required, or fallback `OPENAI_API_KEY`)
- `RAG_BASE_URL` (optional, fallback `OPENAI_BASE_URL` / `OPENAI_API_BASE`)
- `RAG_MODEL_NAME` (optional default model)
