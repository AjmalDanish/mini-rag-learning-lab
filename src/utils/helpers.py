"""Utility functions for the RAG pipeline."""

import os
import json


def ensure_dir(path: str) -> None:
    """Ensure a directory exists."""
    os.makedirs(path, exist_ok=True)


def save_json(data: dict | list, path: str) -> None:
    """Save data to a JSON file."""
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def load_json(path: str) -> dict | list:
    """Load data from a JSON file."""
    with open(path) as f:
        return json.load(f)


def format_source(metadata: dict) -> str:
    """Format metadata into a source citation string."""
    source = metadata.get("source", "unknown")
    page = metadata.get("page", "?")
    return f"{source}, page {page}"


def count_words(text: str) -> int:
    """Count words in text."""
    return len(text.split())
