import networkx as nx

from .config import USE_LLM_SEMANTIC_EDGES, SEMANTIC_EDGE_CONFIDENCE_THRESHOLD
from .extraction import (
    NarrativeExtractionOutput,
    extract_theme_groundings,
    extract_action_relationships,
)


def create_rich_story_graph(extraction: NarrativeExtractionOutput, story_id: str) -> nx.DiGraph:
    """Create a directed story graph with structural edges plus semantic grounding links."""
    G = nx.DiGraph()

    story_node_id = f"story_{story_id}"
    G.add_node(story_node_id, type="Story", text=story_id, plot_type=extraction.plot_type)

    theme_nodes = []
    for i, theme in enumerate(extraction.abstract_theme):
        theme_id = f"theme_{i}"
        theme_nodes.append(theme_id)
        G.add_node(theme_id, type="Theme", text=theme)
        G.add_edge(story_node_id, theme_id, rel="story_has_theme")

    action_ids = []
    for i, action in enumerate(extraction.course_of_action):
        action_id = f"action_{i}"
        action_ids.append(action_id)
        G.add_node(action_id, type="Action", text=action, position=i)
        if i == 0:
            G.add_edge(story_node_id, action_id, rel="story_starts_with")
        else:
            G.add_edge(action_ids[i - 1], action_id, rel="next_action")

    outcome_ids = []
    for i, outcome in enumerate(extraction.outcome):
        outcome_id = f"outcome_{i}"
        outcome_ids.append(outcome_id)
        G.add_node(outcome_id, type="Outcome", text=outcome)
        G.add_edge(story_node_id, outcome_id, rel="story_has_outcome")
        if action_ids:
            G.add_edge(action_ids[-1], outcome_id, rel="action_leads_to_outcome")

    if USE_LLM_SEMANTIC_EDGES:
        G = add_theme_grounded_edges(G)

    for theme_node in theme_nodes:
        G.add_edge(theme_node, story_node_id, rel="theme_supports_story", weight=1.0)

    if action_ids:
        G.add_edge(action_ids[-1], story_node_id, rel="action_leads_to_story", weight=1.0)

    for outcome_node in outcome_ids:
        G.add_edge(outcome_node, story_node_id, rel="outcome_reflects_story", weight=1.0)

    return G


def add_theme_grounded_edges(G: nx.DiGraph) -> nx.DiGraph:
    """Add grounded theme→action and theme→outcome edges (confidence ≥ threshold)."""
    theme_nodes = [n for n, d in G.nodes(data=True) if d.get("type") == "Theme"]
    if not theme_nodes:
        return G

    action_nodes = [n for n, d in G.nodes(data=True) if d.get("type") == "Action"]
    outcome_nodes = [n for n, d in G.nodes(data=True) if d.get("type") == "Outcome"]

    theme_texts = [G.nodes[n]["text"] for n in theme_nodes]
    action_texts = [G.nodes[n]["text"] for n in action_nodes]
    outcome_texts = [G.nodes[n]["text"] for n in outcome_nodes]

    response = extract_theme_groundings(theme_texts, action_texts, outcome_texts)
    if not response:
        return G

    theme_map = {f"theme_{i}": node for i, node in enumerate(theme_nodes)}
    action_map = {f"action_{i}": node for i, node in enumerate(action_nodes)}
    outcome_map = {f"outcome_{i}": node for i, node in enumerate(outcome_nodes)}

    for link in response.theme_to_action:
        theme_node = theme_map.get(link.theme_id)
        action_node = action_map.get(link.action_id)
        if theme_node and action_node and link.confidence >= SEMANTIC_EDGE_CONFIDENCE_THRESHOLD:
            G.add_edge(theme_node, action_node, rel="theme_explains_action",
                       weight=float(link.confidence))

    for link in response.theme_to_outcome:
        theme_node = theme_map.get(link.theme_id)
        outcome_node = outcome_map.get(link.outcome_id)
        if theme_node and outcome_node and link.confidence >= SEMANTIC_EDGE_CONFIDENCE_THRESHOLD:
            G.add_edge(theme_node, outcome_node, rel="theme_explains_outcome",
                       weight=float(link.confidence))

    return G


def add_action_relationships(G: nx.DiGraph) -> nx.DiGraph:
    """Add explicit causal/dependency edges between action nodes."""
    action_nodes = [n for n, d in G.nodes(data=True) if d.get("type") == "Action"]
    if len(action_nodes) < 2:
        return G

    action_texts = [G.nodes[n]["text"] for n in action_nodes]
    relationships = extract_action_relationships(action_texts, action_nodes)

    for rel in relationships:
        if rel["confidence"] >= SEMANTIC_EDGE_CONFIDENCE_THRESHOLD:
            G.add_edge(
                rel["source_node"],
                rel["target_node"],
                rel=f"action_{rel['rel_type']}",
                weight=float(rel["confidence"]),
            )

    return G
