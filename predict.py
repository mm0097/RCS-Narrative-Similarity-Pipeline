"""
Predict text_a_is_closer for test_track_a.jsonl.

Output format matches dev_track_a.jsonl: original fields + text_a_is_closer (bool).

Usage:
    python predict.py \\
        --input test_track_a.jsonl \\
        --output predictions.jsonl \\
        --method fused|gnn|gemini|node_aggregation
"""

import argparse
import json
import os
import sys

from dotenv import load_dotenv
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pipeline.evaluate import cosine_similarity
from pipeline.graph_embedding import story_to_graph_embedding
from pipeline.embeddings import embed_story_gemini
from pipeline.run_pipeline import _fuse, load_jsonl, write_jsonl


def embed(text: str, method: str):
    if method == "gnn":
        return story_to_graph_embedding(text, method="gnn")
    elif method == "node_aggregation":
        return story_to_graph_embedding(text, method="node_aggregation")
    elif method == "gemini":
        return embed_story_gemini(text)
    elif method == "fused":
        gnn_emb = story_to_graph_embedding(text, method="gnn")
        gem_emb = embed_story_gemini(text)
        return _fuse(gnn_emb, gem_emb)
    else:
        raise ValueError(f"Unknown method: {method!r}")


def main():
    load_dotenv()

    parser = argparse.ArgumentParser(description="Predict text_a_is_closer for SemEval Track A")
    parser.add_argument("--input", default="test_track_a.jsonl")
    parser.add_argument("--output", default="predictions.jsonl")
    parser.add_argument(
        "--method",
        choices=["fused", "gnn", "gemini", "node_aggregation"],
        default="fused",
    )
    parser.add_argument("--gemini-key", default=None)
    args = parser.parse_args()

    if args.gemini_key:
        os.environ["GEMINI_API_KEY"] = args.gemini_key

    if args.method in ("gemini", "fused") and not os.environ.get("GEMINI_API_KEY"):
        parser.error("--gemini-key or GEMINI_API_KEY env var required for gemini/fused.")

    records = load_jsonl(args.input)
    print(f"Loaded {len(records)} records from {args.input}")

    out = []
    for rec in tqdm(records, desc=f"Predicting [{args.method}]"):
        anchor = rec["anchor_text"]
        text_a = rec["text_a"]
        text_b = rec["text_b"]

        emb_anchor = embed(anchor, args.method)
        emb_a = embed(text_a, args.method)
        emb_b = embed(text_b, args.method)

        text_a_is_closer = bool(cosine_similarity(emb_anchor, emb_a) > cosine_similarity(emb_anchor, emb_b))

        out.append({**rec, "text_a_is_closer": text_a_is_closer})

    write_jsonl(args.output, out)
    print(f"Predictions written to {args.output}")


if __name__ == "__main__":
    main()
