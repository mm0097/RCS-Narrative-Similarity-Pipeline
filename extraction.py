from typing import List, Optional
from pydantic import BaseModel

from .config import LLM_MODEL, LLM_TEMPERATURE
from .llm import gen_ollama

# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------

class NarrativeExtractionOutput(BaseModel):
    abstract_theme: List[str]
    course_of_action: List[str]
    outcome: List[str]
    plot_type: str


class StorySummaryOutput(BaseModel):
    summary: str


class ThemeToActionType(BaseModel):
    theme_id: str
    action_id: str
    confidence: float


class ThemeToOutcomeType(BaseModel):
    theme_id: str
    outcome_id: str
    confidence: float


class SemanticMeaningModelResponse(BaseModel):
    theme_to_action: List[ThemeToActionType]
    theme_to_outcome: List[ThemeToOutcomeType]


class ActionRelationship(BaseModel):
    source_action_id: str
    target_action_id: str
    relationship_type: str
    confidence: float


class ActionRelationshipsResponse(BaseModel):
    relationships: List[ActionRelationship]


# ---------------------------------------------------------------------------
# Prompt (Prompt 1 — full 40-rule version)
# ---------------------------------------------------------------------------

_EXTRACTION_PROMPT = """# Narrative Similarity Extraction Prompt
You are given a story. Your task is to evaluate their narrative similarity by extracting three components from each story: **abstract theme**, **course of action**, and **outcome**. Follow these definitions and instructions precisely.
---
## 1. Abstract Theme
Extract the story's high-level conceptual core:
1. Themes must be high-level conceptual ideas, not plot, not events, not characters, not settings.
2. Each theme must be a short conceptual phrase (3–8 words) not words or full lengthy descriptions.
3. Produce all the distinct themes covering different conceptual dimensions (emotional, situational, existential, relational, or moral).
4. Do not merge themes. Each theme must reflect a different conceptual dimension.
5. Do not describe what literally happens in the story.
6. If the story supports multiple interpretations, include them separately.
7. The core of the story should be the most important theme.
8. Do not extract themes from each sentence. Focus on the story as a whole for extraction.
9. Order the themes by importance.

---
## 2. Course of Action
Extract the central **sequence of events**:
1. Produce all steps in chronological order.
2. Each step must be a single major action, turning point, or causal development.
3. Maintain chronological order even if the story is nonlinear.
4. Keep steps short, factual, and action-focused.
5. Ignore minor details and background information.
6. If an action is implied but not stated, mark it with "(inferred)".
7. Keep each step short, direct, and action-focused.
8. Only mention the core of the action without extra details.
9. Use simple sentences.
---
## 3. Outcomes
Extract the story's final state:
- Describe how major conflicts/story core themes conclude or remain unresolved.
- Include the final fates of primary characters.
- Capture explicit or strongly implied morals or lessons.
- Exclude intermediate states that are later reversed.
- If the ending is ambiguous, describe the ambiguity directly.
- Use simple phrases not words or full lengthy descriptions.
- Do not repeat actions from the course of action. Focus only on the final state.
---
## 4. Plot Type
Identify the overall narrative structure type from these categories:
- Tragedy: protagonist fails or dies
- Comedy: protagonist succeeds, happy ending
- Quest: journey to achieve a goal
- Rebirth: character transformation or redemption
- Overcoming the Monster: defeating an antagonist or obstacle
- Rags to Riches: rise from poverty/obscurity to success
- Other: if none of the above fit well
---
## General Rules
- Focus only on the three core components; ignore minor details unless they influence the main narrative.
- Largely ignore side storylines unless they affect the primary conflict or message.
- Keep extractions clear, concise, and content-focused.
- Mark ambiguous or inferred elements clearly.
- Keep the extractions simple and straightforward.
- Do not be philosophical or abstract beyond the story's content.
---
## Important Rules
Theme, Course of Action, Outcome are all distinct and independent from each other.

## **Required Output Format**
Return the extracted information in the following JSON structure:
```json
{{
  "abstract_theme": [ list of themes ],
  "course_of_action": [ list of course of actions in chronological order ],
  "outcome": [ List of outcomes ],
  "plot_type": "one of: Tragedy, Comedy, Quest, Rebirth, Overcoming the Monster, Rags to Riches, Other"
}}
```
Fill each array with the extracted elements following the rules above.
## Input
{Story}"""


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------

def extract_narrative_components(story: str) -> NarrativeExtractionOutput:
    """Extract abstract theme, course of action, outcome, and plot type from a story."""
    prompt = _EXTRACTION_PROMPT.format(Story=story)
    response_text = gen_ollama(
        prompt=prompt,
        model_name=LLM_MODEL,
        system_instruction="You are an expert at extracting structured narrative information from stories. Always output valid JSON.",
        json_schema=NarrativeExtractionOutput.model_json_schema(),
        temperature=LLM_TEMPERATURE,
    )
    if response_text is None:
        raise RuntimeError("LLM returned no response for narrative extraction.")
    return NarrativeExtractionOutput.model_validate_json(response_text)


def generate_story_summary(story: str) -> str:
    """Generate a one-sentence summary of the story."""
    prompt = f"""Provide a very short, one-sentence summary of this story. Keep it concise and capture the essence:

Story: {story}

Return only the summary sentence, nothing else."""
    try:
        response_text = gen_ollama(
            prompt=prompt,
            model_name=LLM_MODEL,
            system_instruction="You are an expert at summarizing stories. Always output valid JSON.",
            json_schema=StorySummaryOutput.model_json_schema(),
            temperature=LLM_TEMPERATURE,
        )
        if response_text:
            return StorySummaryOutput.model_validate_json(response_text).summary
        return "Story summary unavailable"
    except Exception as e:
        print(f"Summary generation failed: {e}")
        return "Story summary unavailable"


def extract_theme_groundings(
    theme_texts: List[str],
    action_texts: List[str],
    outcome_texts: List[str],
) -> Optional[SemanticMeaningModelResponse]:
    """Identify explicit theme → action/outcome links using LLM."""
    if not theme_texts:
        return None

    actions_list = "\n".join([f"action_{i}: {a}" for i, a in enumerate(action_texts)]) if action_texts else "None"
    outcomes_list = "\n".join([f"outcome_{i}: {o}" for i, o in enumerate(outcome_texts)]) if outcome_texts else "None"
    themes_list = "\n".join([f"theme_{i}: {t}" for i, t in enumerate(theme_texts)])

    prompt = f"""The following are the actions, outcomes, and themes from a single story. For each theme, list the actions and outcomes that explicitly exemplify it. Only encode relationships that are literally stated in the story. Do not infer or invent new relations.

ACTIONS:
{actions_list}

OUTCOMES:
{outcomes_list}

THEMES:
{themes_list}

Return JSON with the structure:
{{
  "theme_to_action": [
    {{
      "theme_id": "theme_0",
      "action_id": "action_1",
      "confidence": 0.9
    }}
  ],
  "theme_to_outcome": [
    {{
      "theme_id": "theme_0",
      "outcome_id": "outcome_0",
      "confidence": 0.8
    }}
  ]
}}
Use the IDs exactly as provided and only include relationships you are confident about. Confidence must be between 0.0 and 1.0.
"""
    try:
        response_text = gen_ollama(
            prompt=prompt,
            model_name=LLM_MODEL,
            system_instruction="You are an expert at identifying explicit ties between themes and actions/outcomes. Always output valid JSON.",
            json_schema=SemanticMeaningModelResponse.model_json_schema(),
            temperature=LLM_TEMPERATURE,
        )
        if not response_text:
            return None
        return SemanticMeaningModelResponse.model_validate_json(response_text)
    except Exception:
        return None


def extract_action_relationships(
    action_texts: List[str], action_nodes: List[str]
) -> list[dict]:
    """Extract explicit causal/dependency relationships between actions using LLM."""
    if len(action_texts) < 2:
        return []

    actions_list = "\n".join([f"action_{i}: {a}" for i, a in enumerate(action_texts)])

    prompt = f"""Analyze event-to-event relationships between these story actions.

CRITICAL RULES:
1. You may ONLY encode relationships that are EXPLICITLY WRITTEN in the story text
2. Do NOT infer causality
3. Do NOT add logical causality
4. Do NOT add semantic causality
5. Do NOT add LLM-invented causality

ACTIONS (in chronological order):
{actions_list}

You may encode:
- next_action (always valid - already handled)
- Explicit causal relations stated in text (rare)
- Explicit conflict (rare)
- Explicit reaction (rare)
- Explicit purpose (rare)

You MUST NOT encode:
- Inferred causality
- Logical causality
- Semantic causality
- LLM-invented causality

Return JSON with only EXPLICITLY STATED relationships:
{{
  "relationships": [
    {{
      "source_action_id": "action_0",
      "target_action_id": "action_2",
      "relationship_type": "causes",
      "confidence": 0.9
    }}
  ]
}}

If it's not literally written in the story text, you must NOT encode it as an event-to-event relation.

Confidence:
- 0.9-1.0: Explicitly stated with causal language (e.g., "because", "caused", "led to")
- 0.7-0.9: Explicitly stated reaction or purpose
- Below 0.7: Don't include

Use exact IDs: action_0, action_1, etc."""

    try:
        response_text = gen_ollama(
            prompt=prompt,
            model_name=LLM_MODEL,
            system_instruction="You are an expert at identifying ONLY explicit relationships stated in narrative text. Do not infer or invent relationships. Always output valid JSON.",
            json_schema=ActionRelationshipsResponse.model_json_schema(),
            temperature=LLM_TEMPERATURE,
        )
        if not response_text:
            return []

        result = ActionRelationshipsResponse.model_validate_json(response_text)
        action_id_map = {f"action_{i}": node for i, node in enumerate(action_nodes)}

        relationships = []
        for rel in result.relationships:
            src_id = rel.source_action_id.strip()
            tgt_id = rel.target_action_id.strip()
            if src_id in action_id_map and tgt_id in action_id_map:
                relationships.append({
                    "source_node": action_id_map[src_id],
                    "target_node": action_id_map[tgt_id],
                    "rel_type": rel.relationship_type,
                    "confidence": rel.confidence,
                })
        return relationships
    except Exception:
        return []
