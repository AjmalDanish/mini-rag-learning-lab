"""Mini RAG Learning Lab - command-line entry point.

    python main.py                                  # demo on the sample leave policy
    python main.py --ingest docs/ report.pdf        # index PDFs (files or folders), persisted in chroma_db/
    python main.py "How many sick days do I get?"   # ask the indexed documents
    python main.py "question" --ingest docs/        # both at once (question goes first)
"""

import argparse
import logging
import sys
from pathlib import Path

from src.ingestion import SAMPLE_POLICY, create_pdf
from src.pipeline import RAGPipeline

DEMO_QUESTION = "How many weeks of maternity leave are available?"


def pdf_paths(targets: list[str]) -> list[Path]:
    """Expand files and folders into a list of PDF paths."""
    paths = []
    for t in map(Path, targets):
        if t.is_dir():
            paths.extend(sorted(t.glob("*.pdf")))
        elif t.suffix.lower() == ".pdf" and t.exists():
            paths.append(t)
        else:
            sys.exit(f"Not a PDF or folder: {t}")
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Ask questions over your PDFs.")
    parser.add_argument("question", nargs="?", help="question to ask (omit for the demo)")
    parser.add_argument("--ingest", nargs="+", metavar="PATH", help="PDF files or folders to index")
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

    rag = RAGPipeline()

    if args.ingest:
        for path in pdf_paths(args.ingest):
            print(f"Indexed {rag.ingest(path):>4} chunks  {path}")
        if not args.question:
            return

    question = args.question
    if question is None:
        Path("data").mkdir(exist_ok=True)
        rag.ingest(create_pdf("data/leave_policy.pdf", SAMPLE_POLICY))
        question = DEMO_QUESTION

    try:
        result = rag.ask(question)
    except RuntimeError as e:
        sys.exit(str(e))

    print("=" * 60)
    print("QUESTION:", question)
    print("=" * 60)
    print("\nANSWER:", result["answer"])
    print("\nSOURCES:")
    for text, meta, score in result["sources"]:
        print(f"  [{score:.3f}] {meta['source']} p.{meta['page']}: {' '.join(text.split())[:90]}")


if __name__ == "__main__":
    main()
