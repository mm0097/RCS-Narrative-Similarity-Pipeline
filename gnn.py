try:
    import torch
    import torch.nn.functional as F
    from torch_geometric.nn import HeteroConv, SAGEConv

    TORCH_GEOMETRIC_AVAILABLE = True
except ImportError:
    TORCH_GEOMETRIC_AVAILABLE = False


if TORCH_GEOMETRIC_AVAILABLE:
    class HeteroGNN(torch.nn.Module):
        """
        Heterogeneous GNN for story graphs. Forward-pass only — no training code.

        Architecture:
          1. Per-node-type LayerNorm on 384-D input features
          2. 3 × HeteroConv(SAGEConv) layers → 512-D hidden
          3. Readout: Story node embedding (512-D)
          4. Linear projection → 2048-D output
        """

        def __init__(self, in_channels: int, hidden_channels: int, out_channels: int,
                     edge_types: list, node_types: list, num_layers: int = 3):
            super().__init__()

            # Per node type input layer norms
            self.input_norms = torch.nn.ModuleDict({
                nt: torch.nn.LayerNorm(in_channels) for nt in node_types
            })

            # 3 HeteroConv (SAGEConv) layers
            self.convs = torch.nn.ModuleList()
            for _ in range(num_layers):
                conv_dict = {et: SAGEConv((-1, -1), hidden_channels) for et in edge_types}
                self.convs.append(HeteroConv(conv_dict, aggr="mean"))

            # Linear projection: 512 → 2048
            self.lin = torch.nn.Linear(hidden_channels, out_channels)

        def forward(self, x_dict: dict, edge_index_dict: dict) -> "torch.Tensor":
            # Apply per-node-type layer norms to inputs
            x_dict = {
                nt: self.input_norms[nt](x) if nt in self.input_norms else x
                for nt, x in x_dict.items()
            }

            # 3 HeteroConv layers with ReLU
            for conv in self.convs:
                x_dict_new = conv(x_dict, edge_index_dict)
                x_dict_new = {nt: F.relu(x) for nt, x in x_dict_new.items() if x is not None}
                # Restore node types dropped by conv (no incoming edges)
                for nt in x_dict:
                    if nt not in x_dict_new:
                        x_dict_new[nt] = x_dict[nt]
                x_dict = x_dict_new

            # Readout: Story node (index 0 — there is exactly one Story node per graph)
            story_emb = x_dict.get("Story")
            if story_emb is None or story_emb.numel() == 0:
                # Fallback: mean-pool all node types
                valid = [torch.mean(x, dim=0) for x in x_dict.values()
                         if x is not None and x.numel() > 0]
                if not valid:
                    return torch.zeros(self.lin.out_features)
                story_emb = torch.stack(valid).mean(dim=0, keepdim=True)

            graph_emb = story_emb[0]           # (512,)
            return self.lin(graph_emb)         # (2048,)

else:
    class HeteroGNN:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            raise ImportError("torch-geometric is not installed; HeteroGNN is unavailable.")
