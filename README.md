# Narrative Similarity Pipeline

Inference and training pipeline for narrative similarity using graph-based and semantic embeddings.

---

## Setup

```bash
pip install -r pipeline/requirements.txt
```

Set your Gemini API key (required for `gemini` and `fused` methods):

```bash
export GEMINI_API_KEY=your_key_here
# or add it to a .env file in the project root
```

Ollama must be running with the `gpt-oss:20b` model available:

```bash
ollama serve
ollama pull gpt-oss:20b
```

---

## Architecture Overview

Each story goes through the following pipeline:

```
story text
    │
    ├─ LLM extraction (Ollama gpt-oss:20b)
    │       → abstract themes, course of action, outcome, plot type
    │
    ├─ Graph construction (NetworkX DiGraph)
    │       nodes: Story, Theme, Action, Outcome
    │       edges: structural + LLM-grounded semantic edges
    │
    └─ Embedding (two backends)
            node_aggregation  → 1152-D  (3 × 384, MiniLM sum-pooled per node type)
            gnn               → 2048-D  (HeteroGNN Story node readout)
            gemini            → 2048-D  (Gemini text-embedding-004 on full story)
            fused             → 2048-D  (0.5 × norm(gnn) + 0.5 × norm(gemini))
```

### GNN Architecture

| Component | Detail |
|---|---|
| Input features | all-MiniLM-L6-v2, 384-D per node |
| Input normalisation | per-node-type `LayerNorm(384)` |
| Message passing | 3 × `HeteroConv(SAGEConv)`, hidden 512-D |
| Readout | Story node embedding (512-D) |
| Projection | `Linear(512 → 2048)` |

---

## Inference — `run_pipeline.py`

Runs the full pipeline on a JSONL file of story triplets and writes predictions.

### Input format

Each line must be a JSON object with:

```json
{
  "anchor_text": "...",
  "text_a": "...",
  "text_b": "...",
  "text_a_is_closer": true   // optional — enables accuracy reporting
}
```

### Usage

```bash
python pipeline/run_pipeline.py \
    --input  data/track_a.jsonl \
    --output predictions.jsonl \
    --method node_aggregation
```

### Methods

| `--method` | Embedding | Output dim | Notes |
|---|---|---|---|
| `node_aggregation` | all-MiniLM + graph | 1152 | Fast, no API key needed |
| `gnn` | all-MiniLM + GNN | 2048 | Requires trained checkpoint |
| `gemini` | Gemini API | 2048 | Requires `GEMINI_API_KEY` |
| `fused` | GNN story-node readout + Gemini | 2048 | Embedding-level: `0.5*(norm(gnn) + norm(gemini))` |

### Output format

Each line of the output JSONL:

```json
{
  "sim_a": 0.812,
  "sim_b": 0.743,
  "predicted": true,
  "label": true,       // only if input has text_a_is_closer
  "correct": true      // only if input has text_a_is_closer
}
```

Accuracy is printed at the end if labels are present.

### Full example

```bash
python pipeline/run_pipeline.py \
    --input  input_data/synthetic_data_for_classification.jsonl \
    --output predictions.jsonl \
    --method fused \
    --gemini-key $GEMINI_API_KEY
```

---

## Prediction — `predict.py`

Generates predictions for a test file in the same format as the labelled dev set (`dev_track_a.jsonl`): original fields preserved, `text_a_is_closer` appended.

### Usage

```bash
python predict.py \
    --input  test_track_a.jsonl \
    --output predictions.jsonl \
    --method fused
```

### Output format

Each output line mirrors `dev_track_a.jsonl`:

```json
{
  "anchor_text": "...",
  "text_a": "...",
  "text_b": "...",
  "text_a_is_closer": true
}
```

### Options

| Flag | Default | Description |
|---|---|---|
| `--input` | `test_track_a.jsonl` | Input JSONL (no labels required) |
| `--output` | `predictions.jsonl` | Output JSONL |
| `--method` | `fused` | `fused` / `gnn` / `gemini` / `node_aggregation` |
| `--gemini-key` | — | Gemini API key (or set `GEMINI_API_KEY`) |

---

## Training — `train_gnn.py`

Trains the HeteroGNN with triplet margin loss. Requires the same JSONL format as inference.

Training is split into two phases to avoid re-running LLM extraction on every run.

### Phase 1 — Preprocess (run once)

Extracts narrative components for every unique story, builds the graph, embeds nodes with all-MiniLM, and caches everything as `HeteroData` objects.

```bash
python pipeline/train_gnn.py preprocess \
    --input input_data/synthetic_data_for_classification.jsonl \
    --cache cache/graph_cache.pt
```

- Skips stories already present in the cache — safe to interrupt and resume.
- With 5692 unique stories and Ollama, expect several hours depending on hardware.

### Phase 2 — Train

Loads the cache, builds the GNN, and trains with `TripletMarginLoss`.

```bash
python pipeline/train_gnn.py train \
    --input  input_data/synthetic_data_for_classification.jsonl \
    --cache  cache/graph_cache.pt \
    --checkpoint pipeline/checkpoints/gnn.pt \
    --epochs 20 \
    --lr 1e-4 \
    --margin 0.5 \
    --batch-size 32
```

### Training options

| Flag | Default | Description |
|---|---|---|
| `--epochs` | 20 | Number of training epochs |
| `--lr` | 1e-4 | AdamW learning rate |
| `--margin` | 0.5 | Triplet loss margin |
| `--batch-size` | 32 | Triplets per batch |
| `--cache` | `cache/graph_cache.pt` | Path to preprocessed cache |
| `--checkpoint` | `pipeline/checkpoints/gnn.pt` | Where to save the trained model |

### Checkpoint format

The saved `.pt` file contains:

```python
{
    "model_state_dict": { ... },   # HeteroGNN weights
    "edge_types": [ ... ],         # edge type schema used during training
}
```

### Using a trained checkpoint for inference

Load the checkpoint in your own code:

```python
import torch
from pipeline.gnn import HeteroGNN
from pipeline.config import MINILM_DIM, GNN_HIDDEN, GNN_OUT, GNN_LAYERS

ckpt = torch.load("pipeline/checkpoints/gnn.pt", weights_only=False)
model = HeteroGNN(
    in_channels=MINILM_DIM,
    hidden_channels=GNN_HIDDEN,
    out_channels=GNN_OUT,
    edge_types=ckpt["edge_types"],
    node_types=["Story", "Theme", "Action", "Outcome"],
    num_layers=GNN_LAYERS,
)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()
```

---

## Module Reference

| File | Responsibility |
|---|---|
| `config.py` | All constants and hyperparameters |
| `llm.py` | Ollama `chat` / `embed` wrappers |
| `embeddings.py` | all-MiniLM (local) + Gemini API wrappers |
| `extraction.py` | Pydantic schemas + LLM extraction functions |
| `graph.py` | Graph construction and edge addition |
| `gnn.py` | `HeteroGNN` model definition |
| `graph_embedding.py` | `node_aggregation`, `gnn_embedding`, `story_to_graph_embedding` |
| `evaluate.py` | `cosine_similarity`, `evaluate_triplet`, `run_evaluation` |
| `run_pipeline.py` | CLI inference entry point (outputs sim scores + predicted label) |
| `predict.py` | Prediction script — outputs in `dev_track_a.jsonl` format |
| `train_gnn.py` | CLI training entry point (preprocess + train) |
