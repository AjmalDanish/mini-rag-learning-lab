"""Mini RAG Learning Lab - Main entry point."""

import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from src.ingestion import create_pdf, extract_pdf
from src.chunking import chunk_text
from src.embeddings import Embedder
from src.vectordb import VectorStore
from src.retrieval import Retriever
from src.prompts import build_prompt
from src.llm import LLMClient, demo_answer


# Sample policy text for demonstration
POLICY_TEXT = """COMPANY LEAVE POLICY

Annual Leave:
Employees receive 18 days of annual leave per year.

Sick Leave:
Employees receive 10 days of paid sick leave per year.

Maternity Leave:
Female employees receive 26 weeks of paid maternity leave.

Paternity Leave:
Employees receive 7 days of paid paternity leave.

Work From Home:
Employees may work remotely up to 2 days per week with manager approval.
"""


def run_pipeline(query: str, api_key: str = None) -> dict:
    """Run the complete RAG pipeline.
    
    Args:
        query: User question.
        api_key: Optional API key for LLM. If None, uses demo mode.
        
    Returns:
        Dict with answer, sources, and scores.
    """
    # Step 1: Create and extract PDF
    pdf_path = "leave_policy.pdf"
    create_pdf(pdf_path, POLICY_TEXT)
    pages = extract_pdf(pdf_path)
    
    # Step 2: Chunk text
    full_text = " ".join(p["text"] for p in pages)
    chunks = chunk_text(full_text)
    
    # Step 3: Generate embeddings
    embedder = Embedder()
    chunk_embeddings = embedder.encode(chunks)
    
    # Step 4: Store in vector database
    store = VectorStore()
    store.clear()
    
    metadatas = [{"source": "leave_policy.pdf", "page": 1, "chunk_index": i}
                 for i in range(len(chunks))]
    
    store.add(chunks, chunk_embeddings.tolist(), metadatas)
    
    # Step 5: Retrieve and rerank
    retriever = Retriever()
    query_embedding = embedder.encode(query)
    
    results = retriever.retrieve(
        query_embedding,
        chunk_embeddings,
        chunks,
        metadatas,
        top_k=3,
    )
    
    results = retriever.rerank(query, results)
    
    # Step 6: Build prompt and generate answer
    docs_with_meta = [(text, meta) for text, meta, _ in results]
    prompt = build_prompt(query, docs_with_meta)
    
    if api_key:
        client = LLMClient(api_key=api_key)
        answer = client.generate(prompt)
    else:
        answer = demo_answer(query, [text for text, _, _ in results])
    
    return {
        "answer": answer,
        "sources": [(text, meta, score) for text, meta, score in results],
        "prompt": prompt,
    }


if __name__ == "__main__":
    # Example usage
    query = "How many weeks of maternity leave are available?"
    result = run_pipeline(query)
    
    print("=" * 60)
    print("QUERY:", query)
    print("=" * 60)
    print("\nANSWER:", result["answer"])
    print("\nSOURCES:")
    for text, meta, score in result["sources"]:
        print(f"  [{score:.3f}] {text[:100]}...")
        print(f"         Source: {meta['source']}, page {meta['page']}")
