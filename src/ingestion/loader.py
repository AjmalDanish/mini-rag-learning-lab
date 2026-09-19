"""PDF text extraction using PyMuPDF."""

import os
import pymupdf


def create_pdf(path: str, text: str) -> str:
    """Create a PDF file from text."""
    doc = pymupdf.open()
    page = doc.new_page()
    rect = pymupdf.Rect(72, 72, page.rect.width - 72, page.rect.height - 72)
    page.insert_textbox(rect, text, fontsize=13, fontname="helv")
    doc.save(path)
    doc.close()
    return path


def extract_pdf(path: str) -> list[dict]:
    """Extract text and metadata from a PDF file.
    
    Args:
        path: Path to the PDF file.
        
    Returns:
        List of dicts with 'text' and 'metadata' keys.
    """
    doc = pymupdf.open(path)
    pages = []
    for page_no, page in enumerate(doc, start=1):
        pages.append({
            "text": page.get_text().strip(),
            "metadata": {
                "source": os.path.basename(path),
                "page": page_no,
            },
        })
    doc.close()
    return pages
