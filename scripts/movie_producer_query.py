from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.movie_producer_agent import MovieProducerAgent
from agents.movie_producer_prompts import list_prompt_templates


def main() -> None:
    parser = argparse.ArgumentParser(description="Query MovieProducer multi-knowledge-base agent")
    parser.add_argument("--list-kbs", action="store_true", help="List mounted knowledge bases and exit")
    parser.add_argument("--question", type=str, required=False, default="")
    parser.add_argument(
        "--template",
        type=str,
        default="general_research",
        choices=list_prompt_templates(),
    )
    parser.add_argument("--constraints", type=str, default="None")
    parser.add_argument("--k-per-book", type=int, default=3)
    parser.add_argument("--top-k-total", type=int, default=12)
    parser.add_argument("--book-ids", type=str, default="")
    parser.add_argument("--retriever", type=str, choices=["bm25", "vector"], default=None)
    parser.add_argument("--model", type=str, default=None, help="Override RAG model for this run")
    parser.add_argument("--save-json", type=Path, default=None)
    args = parser.parse_args()

    book_ids = [x.strip() for x in args.book_ids.split(",") if x.strip()] or None
    agent = MovieProducerAgent(root=ROOT, model=args.model)

    if args.list_kbs:
        print("=== Mounted Knowledge Bases ===")
        for i, kb in enumerate(agent.list_knowledge_bases(), start=1):
            print(
                f"{i}. book_id={kb['book_id']} | title={kb['title']} | chunks={kb['chunks']} | "
                f"retriever={kb['retriever_type']}"
            )
        return

    if not args.question.strip():
        raise SystemExit("--question is required unless --list-kbs is used.")

    result = agent.query(
        args.question.strip(),
        template=args.template,
        constraints=args.constraints,
        k_per_book=args.k_per_book,
        top_k_total=args.top_k_total,
        book_ids=book_ids,
        retriever_type=args.retriever,
    )

    print("=== MovieProducer Answer ===")
    print(result["answer"])
    print("\n=== Retrieved Sources ===")
    for i, d in enumerate(result["retrieved_docs"], start=1):
        print(
            f"{i}. book={d['book_id']} page={d['page']} source={d['source']} | {d['preview']}"
        )

    if args.save_json:
        agent.save_query_result(result, args.save_json)
        print(f"\nSaved JSON -> {args.save_json}")


if __name__ == "__main__":
    main()
