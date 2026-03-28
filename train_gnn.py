"""
GNN training script — triplet margin loss.

Two-phase workflow:
  1. Preprocess: run LLM extraction + build graphs + embed nodes with all-MiniLM
     → cache HeteroData objects to disk  (slow, done once)
  2. Train: load cache → train HeteroGNN with TripletMarginLoss → save checkpoint

Usage:
    # Phase 1 — only needs to run once
    python pipeline/train_gnn.py preprocess \\
        --input input_data/synthetic_data_for_classification.jsonl \\
        --cache cache/graph_cache.pt

    # Phase 2
    python pipeline/train_gnn.py train \\
        --input input_data/synthetic_data_for_classification.jsonl \\
        --cache cache/graph_cache.pt \\
        --checkpoint pipeline/checkpoints/gnn.pt \\
        --epochs 20 --lr 1e-4 --margin 0.5 --batch-size 32
"""

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline.config import (
    MINILM_DIM, GNN_HIDDEN, GNN_OUT, GNN_LAYERS,
)
from pipeline.embeddings import embed_texts_minilm
from pipeline.extraction import extract_narrative_components, generate_story_summary
from pipeline.graph import create_rich_story_graph, add_action_relationships
from pipeline.gnn import HeteroGNN, TORCH_GEOMETRIC_AVAILABLE

if not TORCH_GEOMETRIC_AVAILABLE:
    raise SystemExit("torch-geometric is required for training.")

from torch_geometric.data import HeteroData


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

NODE_TYPES = ["Story", "Theme", "Action", "Outcome"]


def story_hash(text: str) -> str:
    return hashlib.md5(text.encode()).hexdigest()


def load_jsonl(path: str) -> list[dict]:
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def build_hetero_data(story_text: str) -> HeteroData:
    """Full pipeline: story text → HeteroData with MiniLM node features."""
    extraction = extract_narrative_components(story_text)
    summary = generate_story_summary(story_text)
    G = create_rich_story_graph(extraction, story_id=summary)
    G = add_action_relationships(G)

    data = HeteroData()
    node_to_idx: dict[str, dict] = {nt: {} for nt in NODE_TYPES}

    for node, attrs in G.nodes(data=True):
        nt = attrs.get("type")
        if nt in node_to_idx:
            node_to_idx[nt][node] = len(node_to_idx[nt])

    for nt in NODE_TYPES:
        nodes = [n for n, d in G.nodes(data=True) if d.get("type") == nt]
        if nodes:
            texts = [G.nodes[n]["text"] for n in nodes]
            embs = embed_texts_minilm(texts)          # (N, 384), normalised
            data[nt].x = torch.FloatTensor(embs)
        else:
            data[nt].x = torch.zeros((0, MINILM_DIM))

    for u, v, d in G.edges(data=True):
        rel = d.get("rel")
        if not rel:
            continue
        src_type = G.nodes[u].get("type")
        dst_type = G.nodes[v].get("type")
        if not src_type or not dst_type:
            continue
        edge_key = (src_type, rel, dst_type)
        src_i = node_to_idx[src_type].get(u, -1)
        dst_i = node_to_idx[dst_type].get(v, -1)
        if src_i < 0 or dst_i < 0:
            continue
        if hasattr(data[edge_key], "edge_index") and data[edge_key].edge_index is not None:
            existing = data[edge_key].edge_index
            new_edge = torch.LongTensor([[src_i], [dst_i]])
            data[edge_key].edge_index = torch.cat([existing, new_edge], dim=1)
        else:
            data[edge_key].edge_index = torch.LongTensor([[src_i], [dst_i]])

    return data


def collect_edge_types(cache: dict) -> list[tuple]:
    """Collect all edge type tuples seen across the cached graphs."""
    edge_type_set = set()
    for data in cache.values():
        for key in data.edge_types:
            edge_type_set.add(key)
    return list(edge_type_set)


# ---------------------------------------------------------------------------
# Phase 1 — Preprocessing
# ---------------------------------------------------------------------------

def preprocess(args):
    records = load_jsonl(args.input)

    # Collect all unique story texts
    unique: dict[str, str] = {}  # hash -> text
    for r in records:
        for field in ("anchor_text", "text_a", "text_b"):
            text = r[field]
            h = story_hash(text)
            if h not in unique:
                unique[h] = text

    print(f"Unique stories to process: {len(unique)}")

    # Load existing cache if present
    cache_path = Path(args.cache)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache: dict[str, HeteroData] = {}
    if cache_path.exists():
        cache = torch.load(cache_path, weights_only=False)
        print(f"Loaded {len(cache)} cached stories from {cache_path}")

    missing = {h: t for h, t in unique.items() if h not in cache}
    print(f"Stories needing extraction: {len(missing)}")

    for h, text in tqdm(missing.items(), desc="Extracting"):
        try:
            data = build_hetero_data(text)
            cache[h] = data
        except Exception as e:
            print(f"  Failed ({h[:8]}…): {e}")

    torch.save(cache, cache_path)
    print(f"Cache saved to {cache_path}  ({len(cache)} entries)")


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class TripletDataset(Dataset):
    def __init__(self, records: list[dict], cache: dict[str, HeteroData]):
        self.triplets = []
        for r in records:
            a_h = story_hash(r["anchor_text"])
            ta_h = story_hash(r["text_a"])
            tb_h = story_hash(r["text_b"])
            if a_h not in cache or ta_h not in cache or tb_h not in cache:
                continue
            # pos = closer story, neg = farther story
            if r["text_a_is_closer"]:
                pos_h, neg_h = ta_h, tb_h
            else:
                pos_h, neg_h = tb_h, ta_h
            self.triplets.append((a_h, pos_h, neg_h))
        self.cache = cache

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        a_h, pos_h, neg_h = self.triplets[idx]
        return self.cache[a_h], self.cache[pos_h], self.cache[neg_h]


# ---------------------------------------------------------------------------
# Forward pass helper
# ---------------------------------------------------------------------------

def forward_graph(model: HeteroGNN, data: HeteroData,
                  known_edge_types: set) -> torch.Tensor:
    """Run a single HeteroData object through the GNN → (2048,) embedding."""
    x_dict = {
        nt: data[nt].x
        for nt in NODE_TYPES
        if nt in data.node_types
        and data[nt].x is not None
        and data[nt].x.numel() > 0
    }
    edge_index_dict = {
        key: data[key].edge_index
        for key in data.edge_types
        if key in known_edge_types
        and hasattr(data[key], "edge_index")
        and data[key].edge_index is not None
    }
    return model(x_dict, edge_index_dict)   # (2048,)


def collate_triplets(batch):
    """Keep triplets as a list of (anchor, pos, neg) HeteroData tuples."""
    return batch


# ---------------------------------------------------------------------------
# Phase 2 — Training
# ---------------------------------------------------------------------------

def train(args):
    cache_path = Path(args.cache)
    if not cache_path.exists():
        raise FileNotFoundError(f"Cache not found: {cache_path}  — run preprocess first.")

    cache: dict[str, HeteroData] = torch.load(cache_path, weights_only=False)
    print(f"Loaded cache: {len(cache)} stories")

    records = load_jsonl(args.input)
    dataset = TripletDataset(records, cache)
    print(f"Triplets: {len(dataset)}")

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_triplets,
    )

    edge_types = collect_edge_types(cache)
    print(f"Edge types in training data: {len(edge_types)}")

    model = HeteroGNN(
        in_channels=MINILM_DIM,
        hidden_channels=GNN_HIDDEN,
        out_channels=GNN_OUT,
        edge_types=edge_types,
        node_types=NODE_TYPES,
        num_layers=GNN_LAYERS,
    )
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion = torch.nn.TripletMarginLoss(margin=args.margin, p=2, reduction="mean")

    ckpt_path = Path(args.checkpoint)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    known_edge_types = set(edge_types)

    for epoch in range(1, args.epochs + 1):
        total_loss = 0.0
        n_batches = 0

        for batch in tqdm(loader, desc=f"Epoch {epoch}/{args.epochs}"):
            optimizer.zero_grad()

            anchors, positives, negatives = [], [], []
            for anchor_data, pos_data, neg_data in batch:
                anchors.append(forward_graph(model, anchor_data, known_edge_types))
                positives.append(forward_graph(model, pos_data, known_edge_types))
                negatives.append(forward_graph(model, neg_data, known_edge_types))

            anchor_emb = torch.stack(anchors)    # (B, 2048)
            pos_emb = torch.stack(positives)
            neg_emb = torch.stack(negatives)

            loss = criterion(anchor_emb, pos_emb, neg_emb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)
        print(f"Epoch {epoch:3d}  loss={avg_loss:.4f}")

    torch.save({"model_state_dict": model.state_dict(), "edge_types": edge_types}, ckpt_path)
    print(f"Checkpoint saved to {ckpt_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="GNN training — triplet loss")
    sub = parser.add_subparsers(dest="command", required=True)

    # preprocess
    pp = sub.add_parser("preprocess", help="Extract + cache story graphs")
    pp.add_argument("--input", required=True)
    pp.add_argument("--cache", default="cache/graph_cache.pt")

    # train
    tr = sub.add_parser("train", help="Train GNN on cached graphs")
    tr.add_argument("--input", required=True)
    tr.add_argument("--cache", default="cache/graph_cache.pt")
    tr.add_argument("--checkpoint", default="pipeline/checkpoints/gnn.pt")
    tr.add_argument("--epochs", type=int, default=20)
    tr.add_argument("--lr", type=float, default=1e-4)
    tr.add_argument("--margin", type=float, default=0.5)
    tr.add_argument("--batch-size", type=int, default=32)

    args = parser.parse_args()
    if args.command == "preprocess":
        preprocess(args)
    else:
        train(args)


if __name__ == "__main__":
    main()
