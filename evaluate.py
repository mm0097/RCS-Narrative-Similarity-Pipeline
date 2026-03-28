import numpy as np
from typing import Callable


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors. Returns value in [-1, 1]."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def evaluate_triplet(
    anchor_emb: np.ndarray,
    a_emb: np.ndarray,
    b_emb: np.ndarray,
) -> tuple[float, float, bool]:
    """
    Compare anchor to text_a and text_b.

    Returns:
        (sim_a, sim_b, predicted_a_is_closer)
    """
    sim_a = cosine_similarity(anchor_emb, a_emb)
    sim_b = cosine_similarity(anchor_emb, b_emb)
    return sim_a, sim_b, sim_a > sim_b


def run_evaluation(
    stories: list[dict],
    embed_fn: Callable[[str], np.ndarray],
    verbose: bool = True,
) -> dict:
    """
    Evaluate the embedding function on a list of story triplets.

    stories: list of dicts with keys:
        anchor_text, text_a, text_b, text_a_is_closer (bool)
    embed_fn: callable(story_text) -> np.ndarray

    Returns:
        {accuracy, n_correct, n_total, predictions: list of dicts}
    """
    predictions = []
    n_correct = 0

    for i, story in enumerate(stories):
        if verbose:
            print(f"[{i + 1}/{len(stories)}] Embedding triplet...")

        anchor_emb = embed_fn(story["anchor_text"])
        a_emb = embed_fn(story["text_a"])
        b_emb = embed_fn(story["text_b"])

        sim_a, sim_b, predicted = evaluate_triplet(anchor_emb, a_emb, b_emb)
        label = bool(story["text_a_is_closer"])
        correct = predicted == label

        if correct:
            n_correct += 1

        predictions.append({
            "sim_a": sim_a,
            "sim_b": sim_b,
            "predicted": predicted,
            "label": label,
            "correct": correct,
        })

    n_total = len(stories)
    accuracy = n_correct / n_total if n_total > 0 else 0.0

    if verbose:
        print(f"\nAccuracy: {accuracy:.4f} ({n_correct}/{n_total})")

    return {
        "accuracy": accuracy,
        "n_correct": n_correct,
        "n_total": n_total,
        "predictions": predictions,
    }
