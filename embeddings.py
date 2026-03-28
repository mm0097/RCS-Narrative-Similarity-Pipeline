import os
import numpy as np
import google.generativeai as genai
from sentence_transformers import SentenceTransformer

from .config import MINILM_MODEL, MINILM_DIM, GEMINI_MODEL, GEMINI_DIM

# Lazy singleton for the MiniLM model
_minilm_model: SentenceTransformer | None = None


def _get_minilm() -> SentenceTransformer:
    global _minilm_model
    if _minilm_model is None:
        _minilm_model = SentenceTransformer(MINILM_MODEL)
    return _minilm_model


def embed_texts_minilm(texts: list[str]) -> np.ndarray:
    """Embed texts with all-MiniLM-L6-v2. Returns (N, 384) normalized array."""
    model = _get_minilm()
    embeddings = model.encode(texts, normalize_embeddings=True)
    return np.array(embeddings, dtype=np.float32)


def embed_story_gemini(story_text: str) -> np.ndarray:
    """Embed a full story with Gemini text-embedding-004. Returns (768,) numpy array."""
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise EnvironmentError("GEMINI_API_KEY environment variable not set.")
    genai.configure(api_key=api_key)
    result = genai.embed_content(
        model=GEMINI_MODEL,
        content=story_text,
        task_type="SEMANTIC_SIMILARITY",
        output_dimensionality=GEMINI_DIM,
    )
    return np.array(result["embedding"], dtype=np.float32)
