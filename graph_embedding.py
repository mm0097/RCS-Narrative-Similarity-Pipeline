import numpy as np
import networkx as nx

from .config import MINILM_DIM, GNN_HIDDEN, GNN_OUT, GNN_LAYERS
from .embeddings import embed_texts_minilm
from .extraction import extract_narrative_components, generate_story_summary
from .graph import create_rich_story_graph, add_action_relationships
from .gnn import TORCH_GEOMETRIC_AVAILABLE, HeteroGNN


def node_aggregation(G: nx.DiGraph) -> np.ndarray:
    """
    Aggregate node embeddings by type.

    For each node type (Theme, Action, Outcome):
      - embed all texts with all-MiniLM (already L2-normalised by sentence-transformers)
      - sum the per-node embeddings → 384-D component vector
    Concatenate [theme | action | outcome] → 1152-D, then L2-normalise.
    """
    def _embed_nodes(node_list: list) -> np.ndarray:
        texts = [G.nodes[n]["text"] for n in node_list]
        embs = embed_texts_minilm(texts)          # (N, 384) normalised
        return embs.sum(axis=0)                   # (384,)

    zero = np.zeros(MINILM_DIM, dtype=np.float32)

    theme_nodes = [n for n, d in G.nodes(data=True) if d.get("type") == "Theme"]
    action_nodes = [n for n, d in G.nodes(data=True) if d.get("type") == "Action"]
    outcome_nodes = [n for n, d in G.nodes(data=True) if d.get("type") == "Outcome"]

    theme_emb = _embed_nodes(theme_nodes) if theme_nodes else zero.copy()
    action_emb = _embed_nodes(action_nodes) if action_nodes else zero.copy()
    outcome_emb = _embed_nodes(outcome_nodes) if outcome_nodes else zero.copy()

    graph_emb = np.concatenate([theme_emb, action_emb, outcome_emb])  # (1152,)

    norm = np.linalg.norm(graph_emb)
    if norm > 0:
        graph_emb = graph_emb / norm

    return graph_emb


def gnn_embedding(G: nx.DiGraph) -> np.ndarray:
    """
    Produce a graph embedding using a HeteroGNN forward pass with random weights.
    Falls back to node_aggregation if PyG is unavailable or the graph has no edges.
    """
    if not TORCH_GEOMETRIC_AVAILABLE or G.number_of_edges() == 0:
        return node_aggregation(G)

    try:
        import torch
        from torch_geometric.data import HeteroData

        data = HeteroData()
        node_types = ["Story", "Theme", "Action", "Outcome"]
        node_to_idx: dict[str, dict] = {nt: {} for nt in node_types}

        for node, attrs in G.nodes(data=True):
            nt = attrs.get("type")
            if nt in node_to_idx:
                node_to_idx[nt][node] = len(node_to_idx[nt])

        for nt in node_types:
            nodes = [n for n, d in G.nodes(data=True) if d.get("type") == nt]
            if nodes:
                texts = [G.nodes[n]["text"] for n in nodes]
                embs = embed_texts_minilm(texts)   # (N, 384)
                data[nt].x = torch.FloatTensor(embs)
            else:
                data[nt].x = torch.zeros((0, MINILM_DIM))

        edge_type_set: set = set()
        for u, v, d in G.edges(data=True):
            rel = d.get("rel")
            if rel:
                src_type = G.nodes[u].get("type")
                dst_type = G.nodes[v].get("type")
                if src_type and dst_type:
                    edge_type_set.add((src_type, rel, dst_type))

        for src_type, rel, dst_type in edge_type_set:
            edges = [
                (u, v) for u, v, d in G.edges(data=True)
                if d.get("rel") == rel
                and G.nodes[u].get("type") == src_type
                and G.nodes[v].get("type") == dst_type
            ]
            if edges:
                src_idx = [node_to_idx[src_type].get(u, -1) for u, v in edges]
                dst_idx = [node_to_idx[dst_type].get(v, -1) for u, v in edges]
                valid = [(s, d) for s, d in zip(src_idx, dst_idx) if s >= 0 and d >= 0]
                if valid:
                    data[src_type, rel, dst_type].edge_index = (
                        torch.LongTensor(valid).t()
                    )

        present_node_types = [nt for nt in node_types if nt in x_dict]
        model = HeteroGNN(
            in_channels=MINILM_DIM,
            hidden_channels=GNN_HIDDEN,
            out_channels=GNN_OUT,
            edge_types=list(edge_type_set),
            node_types=present_node_types,
            num_layers=GNN_LAYERS,
        )
        model.eval()

        with torch.no_grad():
            x_dict = {
                nt: data[nt].x
                for nt in node_types
                if nt in data.node_types and data[nt].x is not None and data[nt].x.numel() > 0
            }
            edge_index_dict = {
                key: data[key].edge_index
                for key in data.edge_types
                if hasattr(data[key], "edge_index") and data[key].edge_index is not None
            }

            if not x_dict:
                return node_aggregation(G)

            emb = model(x_dict, edge_index_dict)

        return emb.numpy()

    except Exception as e:
        print(f"GNN embedding failed ({e}), falling back to node_aggregation.")
        return node_aggregation(G)


def story_to_graph_embedding(story: str, method: str = "node_aggregation") -> np.ndarray:
    """
    Full pipeline: story text → extraction → graph → embedding.

    method: "node_aggregation" | "gnn"
    """
    extraction = extract_narrative_components(story)
    summary = generate_story_summary(story)
    G = create_rich_story_graph(extraction, story_id=summary)
    G = add_action_relationships(G)

    if method == "gnn":
        return gnn_embedding(G)
    return node_aggregation(G)
