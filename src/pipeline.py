"""End-to-end RAG pipeline driven by config.yaml."""

import os
from pathlib import Path

import yaml

from .chunking import chunk_by_sentences, chunk_text
from .embeddings import Embedder
from .ingestion import extract_pdf
from .llm import LLMClient, demo_answer
from .prompts import build_prompt
from .retrieval import Retriever
from .vectordb import VectorStore

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.yaml"


def load_config(path: str | Path = CONFIG_PATH) -> dict:
    """Load the pipeline configuration from YAML."""
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


class RAGPipeline:
    """Ingest PDFs into ChromaDB and answer questions over them.

    The LLM is used when OPENAI_API_KEY is set (OPENAI_BASE_URL and LLM_MODEL
    override config.yaml); otherwise answers fall back to demo mode.
    """

    def __init__(self, config: dict | None = None):
        self.cfg = config or load_config()
        c = self.cfg
        self.embedder = Embedder(c["embedding"]["model_name"])
        self.store = VectorStore(c["vectordb"]["collection_name"], c["vectordb"]["persist_dir"])
        self.retriever = Retriever(c["retrieval"]["cross_encoder"]) if c["retrieval"]["use_reranker"] else None

        api_key = os.getenv("OPENAI_API_KEY")
        self.llm = LLMClient(
            api_key=api_key,
            base_url=os.getenv("OPENAI_BASE_URL", c["llm"]["base_url"]),
            model=os.getenv("LLM_MODEL", c["llm"]["model"]),
            timeout=c["llm"]["timeout"],
            max_retries=c["llm"]["max_retries"],
        ) if api_key else None

    def _chunk(self, text: str) -> list[str]:
        c = self.cfg["chunking"]
        if c["strategy"] == "sentence":
            return chunk_by_sentences(text, max_words=c["chunk_size_words"])
        return chunk_text(text, c["chunk_size_words"], c["overlap_words"])

    def ingest(self, pdf_path: str | Path) -> int:
        """Chunk a PDF page by page and (re)index it. Returns chunks stored."""
        chunks, metadatas, ids = [], [], []
        for page in extract_pdf(str(pdf_path)):
            for i, chunk in enumerate(self._chunk(page["text"])):
                meta = {**page["metadata"], "chunk_index": i}
                chunks.append(chunk)
                metadatas.append(meta)
                ids.append(f"{meta['source']}:p{meta['page']}:c{i}")

        # Re-ingesting a file replaces its old chunks instead of leaving stale ones.
        self.store.delete_source(Path(pdf_path).name)
        if chunks:
            self.store.add(chunks, self.embedder.encode(chunks).tolist(), metadatas, ids=ids)
        return len(chunks)

    def ask(self, question: str) -> dict:
        """Retrieve from ChromaDB, re-rank, and generate a grounded answer."""
        if not question.strip():
            raise ValueError("Question must not be empty")
        if self.store.count == 0:
            raise RuntimeError("Vector store is empty - ingest a PDF first")

        top_k = min(self.cfg["retrieval"]["top_k"], self.store.count)
        res = self.store.query(self.embedder.encode(question).tolist(), top_k)
        # Chroma returns cosine distance; convert back to similarity.
        results = [
            (doc, meta, 1.0 - dist)
            for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0])
        ]
        if self.retriever:
            results = self.retriever.rerank(question, results)

        p = self.cfg["prompt"]
        prompt = build_prompt(question, [(t, m) for t, m, _ in results], p["max_chunks"], p["system_prompt"])
        if self.llm:
            answer = self.llm.generate(prompt, temperature=self.cfg["llm"]["temperature"])
        else:
            answer = demo_answer(question, [t for t, _, _ in results])
        return {"answer": answer, "sources": results, "prompt": prompt}
