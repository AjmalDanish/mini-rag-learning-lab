"""Text chunking strategies for RAG."""


def chunk_text(text: str, chunk_size_words: int = 20, overlap_words: int = 4) -> list[str]:
    """Naive word-based chunker with sliding window.
    
    Args:
        text: The text to chunk.
        chunk_size_words: Number of words per chunk.
        overlap_words: Number of overlapping words between chunks.
        
    Returns:
        List of text chunks.
    """
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk_words = words[i : i + chunk_size_words]
        if not chunk_words:
            break
        chunks.append(" ".join(chunk_words))
        if i + chunk_size_words >= len(words):
            break
        i += chunk_size_words - overlap_words
    return chunks


def chunk_by_sentences(text: str, max_words: int = 50) -> list[str]:
    """Sentence-aware chunking. Splits on sentence boundaries.
    
    Args:
        text: The text to chunk.
        max_words: Maximum words per chunk.
        
    Returns:
        List of text chunks.
    """
    import re
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks = []
    current_chunk = []
    current_len = 0
    
    for sent in sentences:
        sent_words = len(sent.split())
        if current_len + sent_words > max_words and current_chunk:
            chunks.append(" ".join(current_chunk))
            current_chunk = [sent]
            current_len = sent_words
        else:
            current_chunk.append(sent)
            current_len += sent_words
    
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    
    return chunks
