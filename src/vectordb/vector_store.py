"""Vector database operations using ChromaDB."""

import chromadb


class VectorStore:
    """ChromaDB vector store wrapper."""
    
    def __init__(self, collection_name: str = "documents", persist_dir: str = "./chroma_db"):
        """Initialize the vector store.
        
        Args:
            collection_name: Name of the collection.
            persist_dir: Directory to persist the database.
        """
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
    
    def clear(self) -> None:
        """Clear all documents from the collection."""
        existing = self.collection.get()["ids"]
        if existing:
            self.collection.delete(ids=existing)
    
    def delete_source(self, source: str) -> None:
        """Remove every chunk that came from the given source file."""
        self.collection.delete(where={"source": source})

    def add(
        self,
        chunks: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict] | None = None,
        prefix: str = "chunk",
        ids: list[str] | None = None,
    ) -> int:
        """Add documents to the collection.
        
        Args:
            chunks: List of text chunks.
            embeddings: List of embedding vectors.
            metadatas: Optional metadata for each chunk.
            prefix: ID prefix for chunks (used when ids is None).
            ids: Explicit chunk IDs.
            
        Returns:
            Number of chunks added.
        """
        ids = ids or [f"{prefix}_{i:02d}" for i in range(len(chunks))]
        if metadatas is None:
            metadatas = [{} for _ in chunks]
        
        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=chunks,
            metadatas=metadatas,
        )
        return len(chunks)
    
    def query(
        self,
        query_embedding: list[float],
        top_k: int = 3,
    ) -> dict:
        """Query the collection for similar documents.
        
        Args:
            query_embedding: Query vector.
            top_k: Number of results to return.
            
        Returns:
            Dict with 'documents', 'metadatas', 'distances'.
        """
        return self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
        )
    
    @property
    def count(self) -> int:
        """Return the number of documents in the collection."""
        return self.collection.count()
