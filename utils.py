import io
from typing import List

import pdfplumber
from docx import Document
import PyPDF2
import logging

from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def extract_text_from_file(uploaded_file) -> str:
    """Extract readable text from an uploaded file-like object.
    Supports: PDF (.pdf), Word (.docx), plain text (.txt).
    Returns plain string (empty string on failure).
    """
    name = uploaded_file.name.lower()
    data = uploaded_file.read()
    if not data:
        return ""
    stream = io.BytesIO(data)

    try:
        if name.endswith('.pdf'):
            text = ''
            try:
                with pdfplumber.open(stream) as pdf:
                    for page in pdf.pages:
                        page_text = page.extract_text() or ''
                        text += page_text + "\n"
            except Exception:
                # fallback to PyPDF2
                stream.seek(0)
                try:
                    reader = PyPDF2.PdfReader(stream)
                    for p in reader.pages:
                        text += p.extract_text() or ''
                except Exception:
                    logging.exception('Failed to parse PDF')
            return text.strip()

        if name.endswith('.docx'):
            stream.seek(0)
            try:
                doc = Document(stream)
                paragraphs = [p.text for p in doc.paragraphs]
                return '\n'.join(paragraphs).strip()
            except Exception:
                logging.exception('Failed to parse DOCX')
                return ''

        if name.endswith('.txt'):
            try:
                return data.decode('utf-8', errors='ignore')
            except Exception:
                return data.decode('latin-1', errors='ignore')

        # unknown extension: try decode as text
        try:
            return data.decode('utf-8', errors='ignore')
        except Exception:
            return ''
    finally:
        try:
            stream.close()
        except Exception:
            pass


def load_sentence_transformer(model_name: str = 'all-MiniLM-L6-v2') -> SentenceTransformer:
    """Load and return a SentenceTransformer model."""
    return SentenceTransformer(model_name)


def embed_texts(model: SentenceTransformer, texts: List[str]) -> np.ndarray:
    """Return numpy embeddings for a list of texts (or single text)."""
    if isinstance(texts, str):
        texts = [texts]
    # clean small texts
    texts = [t if t is not None else '' for t in texts]
    emb = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    return emb


def compute_cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two 1D vectors or 2D arrays.
    If a or b are 2D with single row, it will handle appropriately.
    """
    if a is None or b is None:
        return 0.0
    a = np.array(a).reshape(1, -1)
    b = np.array(b).reshape(1, -1)
    sim = cosine_similarity(a, b)[0][0]
    return float(sim)
