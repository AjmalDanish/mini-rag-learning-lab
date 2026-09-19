"""Unit tests for the Mini RAG Learning Lab.

Run with: python -m pytest tests/test_core.py -v
"""
import os
import sys
import tempfile

import numpy as np
import pytest

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from ingestion.loader import create_pdf, extract_pdf
from chunking.chunker import chunk_text, chunk_by_sentences
from embeddings.embedder import Embedder
from vectordb.vector_store import VectorStore
from retrieval.retriever import Retriever
from prompts.prompt_templates import build_prompt
from utils.helpers import format_source, count_words


# ---------------------------------------------------------------------------
# chunk_text
# ---------------------------------------------------------------------------
class TestChunkText:
    def test_basic_chunking(self):
        text = " ".join(["word"] * 40)
        chunks = chunk_text(text, chunk_size_words=10, overlap_words=2)
        assert len(chunks) > 0
        for c in chunks:
            assert len(c.split()) <= 10

    def test_overlap(self):
        text = "A B C D E F G H I J"
        chunks = chunk_text(text, chunk_size_words=4, overlap_words=2)
        assert len(chunks) >= 2
        words1 = set(chunks[0].split())
        words2 = set(chunks[1].split())
        overlap = words1 & words2
        assert len(overlap) >= 1

    def test_single_chunk(self):
        text = "short text"
        chunks = chunk_text(text, chunk_size_words=20)
        assert len(chunks) == 1
        assert chunks[0] == text

    def test_empty_text(self):
        chunks = chunk_text("")
        assert chunks == []

    def test_exact_boundary(self):
        text = "A B C D"
        chunks = chunk_text(text, chunk_size_words=4, overlap_words=0)
        assert len(chunks) == 1


# ---------------------------------------------------------------------------
# chunk_by_sentences
# ---------------------------------------------------------------------------
class TestChunkBySentences:
    def test_basic_sentence_chunking(self):
        text = "First sentence. Second sentence. Third sentence."
        chunks = chunk_by_sentences(text, max_words=10)
        assert len(chunks) >= 1

    def test_respects_max_words(self):
        text = "This is a sentence. " * 20
        chunks = chunk_by_sentences(text, max_words=10)
        for chunk in chunks:
            assert len(chunk.split()) <= 20  # some tolerance for sentence boundaries


# ---------------------------------------------------------------------------
# build_prompt
# ---------------------------------------------------------------------------
class TestBuildPrompt:
    def test_basic_prompt(self):
        docs = [
            ("Test document", {"source": "test.pdf", "page": 1}),
        ]
        prompt = build_prompt("What is this?", docs)
        assert "What is this?" in prompt
        assert "Test document" in prompt
        assert "test.pdf" in prompt

    def test_max_chunks_limit(self):
        docs = [
            (f"Doc {i}", {"source": "test.pdf", "page": 1})
            for i in range(10)
        ]
        prompt = build_prompt("Question", docs, max_chunks=3)
        assert "Doc 0" in prompt
        assert "Doc 2" in prompt
        assert "Doc 5" not in prompt

    def test_context_contains_source_info(self):
        docs = [
            ("Important text", {"source": "manual.pdf", "page": 5}),
        ]
        prompt = build_prompt("Query", docs)
        assert "manual.pdf" in prompt
        assert "page 5" in prompt


# ---------------------------------------------------------------------------
# extract_pdf (requires PyMuPDF)
# ---------------------------------------------------------------------------
class TestExtractPdf:
    def test_creates_and_extracts(self):
        try:
            import pymupdf
        except ImportError:
            pytest.skip("PyMuPDF not installed")

        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            path = f.name
        try:
            create_pdf(path, "Hello world")
            pages = extract_pdf(path)
            assert len(pages) == 1
            assert "Hello world" in pages[0]["text"]
            assert pages[0]["metadata"]["page"] == 1
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# cosine similarity (manual)
# ---------------------------------------------------------------------------
class TestCosineSimilarity:
    def _cosine(self, a, b):
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

    def test_identical_vectors(self):
        v = np.array([1.0, 2.0, 3.0])
        assert abs(self._cosine(v, v) - 1.0) < 1e-6

    def test_orthogonal_vectors(self):
        a = np.array([1.0, 0.0])
        b = np.array([0.0, 1.0])
        assert abs(self._cosine(a, b)) < 1e-6

    def test_opposite_vectors(self):
        a = np.array([1.0, 0.0])
        b = np.array([-1.0, 0.0])
        assert abs(self._cosine(a, b) + 1.0) < 1e-6

    def test_known_value(self):
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([4.0, 5.0, 6.0])
        expected = 32.0 / (np.sqrt(14) * np.sqrt(77))
        assert abs(self._cosine(a, b) - expected) < 1e-6


# ---------------------------------------------------------------------------
# one-hot equivalence
# ---------------------------------------------------------------------------
class TestOneHotLookup:
    def test_one_hot_equals_row_lookup(self):
        E = np.array([[0.0, 0.0], [1.1, 0.3], [1.0, -0.1], [0.4, 0.8]])
        token_id = 2
        one_hot = np.zeros(len(E))
        one_hot[token_id] = 1.0
        via_onehot = one_hot @ E
        via_row = E[token_id]
        np.testing.assert_array_almost_equal(via_onehot, via_row)


# ---------------------------------------------------------------------------
# format_source
# ---------------------------------------------------------------------------
class TestFormatSource:
    def test_format_source(self):
        meta = {"source": "test.pdf", "page": 3}
        assert format_source(meta) == "test.pdf, page 3"

    def test_format_source_missing(self):
        assert format_source({}) == "unknown, page ?"


# ---------------------------------------------------------------------------
# count_words
# ---------------------------------------------------------------------------
class TestCountWords:
    def test_basic(self):
        assert count_words("hello world") == 2

    def test_empty(self):
        assert count_words("") == 0


# ---------------------------------------------------------------------------
# Notebook structure (if notebook exists)
# ---------------------------------------------------------------------------
class TestNotebookStructure:
    def _notebook_path(self):
        return os.path.join(
            os.path.dirname(__file__), "..", "Mini_RAG_Learning_Lab.ipynb"
        )

    def test_notebook_is_valid_json(self):
        import json
        path = self._notebook_path()
        if not os.path.exists(path):
            pytest.skip("Notebook not found")
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        assert "cells" in data

    def test_no_empty_trailing_cells(self):
        import json
        path = self._notebook_path()
        if not os.path.exists(path):
            pytest.skip("Notebook not found")
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        last_cell = data["cells"][-1]
        assert "".join(last_cell["source"]).strip() != ""

    def test_all_markdown_cells_have_content(self):
        import json
        path = self._notebook_path()
        if not os.path.exists(path):
            pytest.skip("Notebook not found")
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        for i, cell in enumerate(data["cells"]):
            if cell["cell_type"] == "markdown":
                content = "".join(cell["source"]).strip()
                assert content != "", f"Markdown cell {i} is empty"
