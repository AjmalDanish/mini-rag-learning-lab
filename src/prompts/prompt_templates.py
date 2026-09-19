"""Prompt templates for RAG generation."""


SYSTEM_PROMPT = """You are an HR assistant. Answer using ONLY the supplied context.
If the answer is not present in the context, say you don't know.
Do not use outside knowledge."""


def build_prompt(
    query: str,
    docs_with_meta: list[tuple[str, dict]],
    max_chunks: int = 3,
    system_prompt: str = SYSTEM_PROMPT,
) -> str:
    """Build a RAG prompt with context and query.
    
    Args:
        query: User question.
        docs_with_meta: List of (text, metadata) tuples.
        max_chunks: Maximum chunks to include.
        system_prompt: System instruction.
        
    Returns:
        Formatted prompt string.
    """
    selected = docs_with_meta[:max_chunks]
    context = "\n\n".join(
        f"[{i}] {text}\n    (source: {meta.get('source', 'unknown')}, page {meta.get('page', '?')})"
        for i, (text, meta) in enumerate(selected, start=1)
    )
    
    return f"""{system_prompt}

CONTEXT:
{context}

QUESTION:
{query}

ANSWER:"""
