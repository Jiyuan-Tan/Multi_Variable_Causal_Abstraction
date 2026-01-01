"""Grading module for evaluating agent output against references."""

import json
from pathlib import Path
from typing import TypedDict, Any


class GradeOutput(TypedDict):
    score: float
    max_score: float
    error: str | None
    field_scores: dict[str, float]
    details: dict[str, dict[str, Any]]
    output: dict[str, Any] | None


def score_list_field(predicted: list, reference: list) -> dict:
    """Score a list field using set-based F1 score.

    Returns dict with score, precision, recall, predicted, reference.
    """
    if not predicted:  # Agent gave no answer
        return {
            "score": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "predicted": predicted,
            "reference": reference,
        }

    # Convert unhashable types (like lists) to hashable types (like tuples)
    def make_hashable(item):
        if isinstance(item, list):
            return tuple(make_hashable(i) for i in item)
        elif isinstance(item, dict):
            return tuple(sorted((k, make_hashable(v)) for k, v in item.items()))
        return item

    pred_set = set(make_hashable(item) for item in predicted)
    ref_set = set(make_hashable(item) for item in reference)

    intersection = pred_set & ref_set

    if not intersection:
        return {
            "score": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "predicted": predicted,
            "reference": reference,
        }

    precision = len(intersection) / len(pred_set)
    recall = len(intersection) / len(ref_set)
    f1 = 2 * (precision * recall) / (precision + recall)

    return {
        "score": f1,
        "precision": precision,
        "recall": recall,
        "predicted": predicted,
        "reference": reference,
    }


def grade_output(path_to_seed: str, worktree_path: Path) -> GradeOutput:
    """Grade agent output against reference.

    Returns grade dictionary with score, field_scores, and details.
    """
    # Extract seed name from path
    seed_name = Path(path_to_seed).stem

    # Read agent output
    output_file = worktree_path / "agents" / "coder" / "outputs" / "output.json"
    if not output_file.exists():
        return {
            "score": 0.0,
            "max_score": 1.0,
            "error": "output.json not found",
            "field_scores": {},
            "details": {},
            "output": None,
        }

    # Read reference (from original repo, not worktree)
    # Resolve relative to this file's location to handle invocation from any directory
    agents_dir = Path(__file__).parent.resolve()
    reference_file = agents_dir / "coder" / "references" / f"{seed_name}.json"
    if not reference_file.exists():
        return {
            "score": 0.0,
            "max_score": 1.0,
            "error": f"reference file not found: {reference_file}",
            "field_scores": {},
            "details": {},
            "output": None,
        }

    with open(output_file) as f:
        output_data = json.load(f)

    with open(reference_file) as f:
        reference_data = json.load(f)

    # Score each field
    field_scores = {}
    details = {}

    for field_name, reference_value in reference_data.items():
        output_value = output_data.get(field_name, [])

        # Handle list fields (assume all fields are lists for now)
        if isinstance(reference_value, list):
            result = score_list_field(output_value, reference_value)
            field_scores[field_name] = result["score"]
            details[field_name] = result
        else:
            # Simple equality for non-list fields
            score = 1.0 if output_value == reference_value else 0.0
            field_scores[field_name] = score
            details[field_name] = {
                "score": score,
                "predicted": output_value,
                "reference": reference_value,
            }

    # Penalize extra fields not in reference
    for field_name in output_data.keys():
        if field_name not in reference_data:
            field_scores[field_name] = 0.0
            details[field_name] = {
                "score": 0.0,
                "predicted": output_data[field_name],
                "reference": None,
                "error": "Extra field not in reference",
            }

    # Overall score is average of field scores
    overall_score = (
        sum(field_scores.values()) / len(field_scores) if field_scores else 0.0
    )

    return {
        "score": overall_score,
        "max_score": 1.0,
        "error": None,
        "field_scores": field_scores,
        "details": details,
        "output": output_data,
    }
