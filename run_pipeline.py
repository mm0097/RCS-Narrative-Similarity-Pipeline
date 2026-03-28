"""
CLI entry point for the narrative similarity inference pipeline.

Usage:
    python pipeline/run_pipeline.py \\
        --input data/track_a.jsonl \\
        --output predictions.jsonl \\
        --method node_aggregation|gnn|gemini|fused \\
        --gemini-key $GEMINI_API_KEY
"""

import argparse
import json
import os
import sys

import numpy as np
from dotenv import load_dotenv
from tqdm import tqdm

# Allow running as `python pipeline/run_pipeline.py` from the repo root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.evaluate import cosine_similarity
from pipeline.graph_embedding import story_to_graph_embedding
from pipeline.embeddings import embed_story_gemini


def load_jsonl(path: str) -> list[dict]:
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def write_jsonl(path: str, records: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")


def build_embed_fn(method: str):
    """Return a callable (story_text) -> np.ndarray for the chosen method."""
    if method == "node_aggregation":
        return lambda text: story_to_graph_embedding(text, method="node_aggregation")
    elif method == "gnn":
        return lambda text: story_to_graph_embedding(text, method="gnn")
    elif method == "gemini":
        return embed_story_gemini
    elif method == "fused":
        # Fused scoring (GNN story-node readout + Gemini) is handled separately
        return None
    else:
        raise ValueError(f"Unknown method: {method!r}")


def _fuse(gnn_emb: np.ndarray, gemini_emb: np.ndarray) -> np.ndarray:
    """L2-normalise each 2048-D vector, then average: 0.5*(norm(gnn) + norm(gemini))."""
    g = gnn_emb / (np.linalg.norm(gnn_emb) + 1e-8)
    s = gemini_emb / (np.linalg.norm(gemini_emb) + 1e-8)
    return 0.5 * (g + s)


def score_triplet_fused(anchor: str, text_a: str, text_b: str) -> tuple[float, float, bool]:
    """
    Fused scoring: embedding-level fusion of GNN story-node readout (2048-D)
    and Gemini (2048-D). Fused vector = 0.5*(norm(gnn) + norm(gemini)).
    """
    # GNN story-node readout → 2048-D
    g_anchor = story_to_graph_embedding(anchor, method="gnn")
    g_a = story_to_graph_embedding(text_a, method="gnn")
    g_b = story_to_graph_embedding(text_b, method="gnn")

    # Gemini embeddings → 2048-D
    sem_anchor = embed_story_gemini(anchor)
    sem_a = embed_story_gemini(text_a)
    sem_b = embed_story_gemini(text_b)

    fused_anchor = _fuse(g_anchor, sem_anchor)
    fused_a = _fuse(g_a, sem_a)
    fused_b = _fuse(g_b, sem_b)

    sim_a = cosine_similarity(fused_anchor, fused_a)
    sim_b = cosine_similarity(fused_anchor, fused_b)

    return sim_a, sim_b, sim_a > sim_b


def main():
    load_dotenv()

    parser = argparse.ArgumentParser(description="Narrative Similarity Inference Pipeline")
    parser.add_argument("--input", required=True, help="Path to input JSONL file")
    parser.add_argument("--output", required=True, help="Path to output predictions JSONL")
    parser.add_argument(
        "--method",
        choices=["node_aggregation", "gnn", "gemini", "fused"],
        default="node_aggregation",
        help="Embedding/scoring method",
    )
    parser.add_argument(
        "--gemini-key",
        default=None,
        help="Gemini API key (overrides GEMINI_API_KEY env var)",
    )
    args = parser.parse_args()

    # Set API key
    if args.gemini_key:
        os.environ["GEMINI_API_KEY"] = args.gemini_key

    if args.method in ("gemini", "fused") and not os.environ.get("GEMINI_API_KEY"):
        parser.error("--gemini-key or GEMINI_API_KEY env var required for gemini/fused methods.")

    stories = load_jsonl(args.input)
    print(f"Loaded {len(stories)} stories from {args.input}")

    predictions = []
    n_correct = 0
    has_labels = "text_a_is_closer" in stories[0] if stories else False

    embed_fn = build_embed_fn(args.method)

    for story in tqdm(stories, desc=f"Processing [{args.method}]"):
        anchor = story["anchor_text"]
        text_a = story["text_a"]
        text_b = story["text_b"]

        if args.method == "fused":
            sim_a, sim_b, predicted = score_triplet_fused(anchor, text_a, text_b)
        else:
            anchor_emb = embed_fn(anchor)
            a_emb = embed_fn(text_a)
            b_emb = embed_fn(text_b)
            sim_a = cosine_similarity(anchor_emb, a_emb)
            sim_b = cosine_similarity(anchor_emb, b_emb)
            predicted = sim_a > sim_b

        record = {
            "sim_a": float(sim_a),
            "sim_b": float(sim_b),
            "predicted": bool(predicted),
        }

        if has_labels:
            label = bool(story["text_a_is_closer"])
            record["label"] = label
            record["correct"] = predicted == label
            if predicted == label:
                n_correct += 1

        predictions.append(record)

    write_jsonl(args.output, predictions)
    print(f"\nPredictions written to {args.output}")

    if has_labels:
        accuracy = n_correct / len(stories)
        print(f"Accuracy: {accuracy:.4f} ({n_correct}/{len(stories)})")


if __name__ == "__main__":
    main()
