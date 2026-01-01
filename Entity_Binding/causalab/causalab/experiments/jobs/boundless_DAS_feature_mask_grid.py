"""
Train Boundless DAS (mask intervention with tie_masks=False) on any component type.

This module provides a unified function to train Boundless DAS on attention heads,
residual stream positions, or MLPs. Unlike DBM (tie_masks=True) which selects/deselects
entire units, Boundless DAS learns which feature dimensions within each unit are important.

Key Features:
1. Accepts pre-built InterchangeTarget (single or Dict[int, InterchangeTarget] for per-layer)
2. Auto-detects component type from unit IDs
3. Trains with intervention_type="mask" and tie_masks=False
4. Generates appropriate feature count heatmap visualization
5. Saves trained models and evaluation results

Output Structure:
================
output_dir/
├── metadata.json               # Experiment configuration and summary
├── models/                     # Trained models
│   └── {ComponentType}(...)/
├── training/                   # Training artifacts
│   └── feature_indices.json
├── train_eval/                 # Training set evaluation
│   ├── scores.json
│   └── raw_results.json
├── test_eval/                  # Test set evaluation
│   ├── scores.json
│   └── raw_results.json
└── heatmaps/                   # Visualization images
    └── {var}_features.png

Usage Example:
==============
```python
from experiments.interchange_targets import (
    build_attention_head_targets,
    build_residual_stream_targets,
    build_mlp_targets,
)
from experiments.jobs.boundless_DAS_feature_mask_grid import train_boundless_DAS

# Single target mode (all units at once)
targets = build_attention_head_targets(
    pipeline, layers, heads, token_position, mode="one_target_all_units"
)
result = train_boundless_DAS(
    causal_model=causal_model,
    interchange_target=targets[("all",)],  # Single InterchangeTarget
    train_dataset_path=train_path,
    test_dataset_path=test_path,
    pipeline=pipeline,
    target_variable="answer",
    output_dir="outputs/attention_boundless",
    metric=metric,
    n_features=32,
)

# Per-layer mode
targets = build_residual_stream_targets(
    pipeline, layers, token_positions, mode="one_target_per_layer"
)
# Convert {(layer,): target} to {layer: target}
per_layer_targets = {key[0]: target for key, target in targets.items()}
result = train_boundless_DAS(
    causal_model=causal_model,
    interchange_target=per_layer_targets,  # Dict[int, InterchangeTarget]
    ...
)
```
"""

import copy
import logging
import os
from typing import Any, Callable, Dict, Optional, Tuple, Union

import numpy as np

from causalab.causal.causal_model import CausalModel
from causalab.experiments.configs.train_config import DEFAULT_CONFIG
from causalab.experiments.interchange_targets import detect_component_type_from_targets
from causalab.experiments.io import save_experiment_metadata
from causalab.experiments.train import train_interventions
from causalab.experiments.visualizations.feature_masks import plot_feature_counts
from causalab.neural.model_units import InterchangeTarget
from causalab.neural.pipeline import LMPipeline

logger = logging.getLogger(__name__)


def train_boundless_DAS(
    causal_model: CausalModel,
    interchange_target: Union[InterchangeTarget, Dict[int, InterchangeTarget]],
    train_dataset_path: str,
    test_dataset_path: str,
    pipeline: LMPipeline,
    target_variable_group: Tuple[str, ...],
    output_dir: str,
    metric: Callable[[Any, Any], bool],
    n_features: int = 32,
    config: Optional[Dict[str, Any]] = None,
    save_results: bool = True,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Train Boundless DAS (mask with tie_masks=False) on any component type.

    This function accepts either:
    - A single InterchangeTarget (trains one model, returns single scores)
    - A Dict[int, InterchangeTarget] mapping layers to targets (trains per-layer, returns per-layer scores)

    The component type (attention_head, residual_stream, or mlp) is auto-detected.

    Args:
        causal_model: Causal model for generating expected outputs
        interchange_target: Either:
            - Single InterchangeTarget (e.g., from mode="one_target_all_units")
            - Dict[int, InterchangeTarget] mapping layer -> target (e.g., from mode="one_target_per_layer")
        train_dataset_path: Path to filtered training dataset directory
        test_dataset_path: Path to filtered test dataset directory
        pipeline: LMPipeline object with loaded model
        target_variable_group: Tuple of target variable names (e.g., ("answer",))
        output_dir: Output directory for results and models
        metric: Function to compare neural output with expected output
        n_features: Number of features for SubspaceFeaturizer (default: 32)
        config: Training configuration dict. Will be configured for mask intervention.
                (default: DEFAULT_CONFIG with mask settings)
        save_results: Whether to save metadata and results to disk (default: True)
        verbose: Whether to print progress information (default: True)

    Returns:
        Dictionary containing:
            For single target mode:
                - train_score: float
                - test_score: float
                - feature_indices: Dict[str, Optional[List[int]]]
            For per-layer mode:
                - train_scores: Dict[int, float] (layer -> score)
                - test_scores: Dict[int, float] (layer -> score)
                - feature_indices: Dict[int, Dict[str, Optional[List[int]]]] (layer -> indices)
            Common fields:
                - component_type: detected component type
                - mode: "single" or "per_layer"
                - metadata: experiment configuration and summary
                - output_paths: paths to saved files and directories

    Raises:
        FileNotFoundError: If dataset paths do not exist
        ValueError: If component type cannot be detected
    """
    # Configure logging level based on verbose flag
    if verbose:
        logger.setLevel(logging.DEBUG)

    # Determine mode: single target or per-layer
    is_per_layer = isinstance(interchange_target, dict)
    mode = "per_layer" if is_per_layer else "single"

    # Detect component type
    if is_per_layer:
        # For dict of targets, wrap in expected format and detect
        targets_for_detection = {
            (layer,): target for layer, target in interchange_target.items()
        }
        component_type = detect_component_type_from_targets(targets_for_detection)
    else:
        # Wrap single target in dict format for detection
        component_type = detect_component_type_from_targets(
            {("single",): interchange_target}
        )

    # Setup configuration
    if config is None:
        config = copy.deepcopy(DEFAULT_CONFIG)

    # Configure for Boundless DAS (mask with tie_masks=False)
    config["intervention_type"] = "mask"
    config["featurizer_kwargs"] = {"tie_masks": False}
    config["DAS"] = {"n_features": n_features}

    # Set defaults if not present
    config.setdefault("train_batch_size", 32)
    config.setdefault("evaluation_batch_size", 64)
    config.setdefault("training_epoch", 20)
    config.setdefault("init_lr", 0.001)
    config.setdefault("id", f"{component_type}_boundless_DAS")

    # Convert input to format expected by train_interventions
    if is_per_layer:
        # Dict[int, InterchangeTarget] -> Dict[tuple, InterchangeTarget]
        targets_dict = {
            (layer,): target for layer, target in interchange_target.items()
        }
    else:
        # Single target -> wrap in dict
        targets_dict = {("single",): interchange_target}

    # Train using train_interventions
    result = train_interventions(
        causal_model=causal_model,
        interchange_targets=targets_dict,
        train_dataset_path=train_dataset_path,
        test_dataset_path=test_dataset_path,
        pipeline=pipeline,
        target_variable_group=target_variable_group,
        output_dir=output_dir,
        metric=metric,
        config=config,
        save_results=save_results,
    )

    # Build return structure based on mode
    return_result: Dict[str, Any]
    if is_per_layer:
        # Per-layer mode: extract scores and indices per layer
        train_scores = {}
        test_scores = {}
        feature_indices = {}

        for key, res in result["results_by_key"].items():
            layer = key[0]
            train_scores[layer] = res["train_score"]
            test_scores[layer] = res["test_score"]
            feature_indices[layer] = res["feature_indices"]

        return_result = {
            "train_scores": train_scores,
            "test_scores": test_scores,
            "feature_indices": feature_indices,
            "component_type": component_type,
            "mode": mode,
            "metadata": result["metadata"],
            "output_paths": result.get("output_paths", {}),
        }

        # Find best layer
        best_layer = max(test_scores, key=lambda k: test_scores[k])
        return_result["metadata"]["best_layer"] = best_layer
        return_result["metadata"]["best_test_score"] = test_scores[best_layer]
        return_result["metadata"]["avg_test_score"] = float(
            np.mean(list(test_scores.values()))
        )
        return_result["metadata"]["layers"] = sorted(interchange_target.keys())

    else:
        # Single target mode: extract single scores
        single_result = result["results_by_key"][("single",)]

        return_result = {
            "train_score": single_result["train_score"],
            "test_score": single_result["test_score"],
            "feature_indices": single_result["feature_indices"],
            "component_type": component_type,
            "mode": mode,
            "metadata": result["metadata"],
            "output_paths": result.get("output_paths", {}),
        }

    # Update metadata with common fields
    return_result["metadata"]["component_type"] = component_type
    return_result["metadata"]["mode"] = mode
    return_result["metadata"]["n_features"] = n_features
    return_result["metadata"]["tie_masks"] = False

    # Generate visualizations
    if save_results:
        heatmap_dir = os.path.join(output_dir, "heatmaps")
        os.makedirs(heatmap_dir, exist_ok=True)

        # Generate display name for title and filename
        var_name = "_".join(target_variable_group)
        heatmap_path = os.path.join(heatmap_dir, f"{var_name}_features.png")

        if is_per_layer:
            plot_feature_counts(
                feature_indices=return_result["feature_indices"],
                scores=return_result["test_scores"],
                n_features=n_features,
                title=f"Boundless DAS: {var_name.replace('_', ' ').title()}",
                save_path=heatmap_path,
            )
        else:
            plot_feature_counts(
                feature_indices=return_result["feature_indices"],
                scores=return_result["test_score"],
                n_features=n_features,
                title=f"Boundless DAS: {var_name.replace('_', ' ').title()}",
                save_path=heatmap_path,
            )

        return_result["output_paths"]["heatmap_dir"] = heatmap_dir
        return_result["output_paths"]["heatmap"] = heatmap_path

        # Save enhanced metadata
        save_experiment_metadata(return_result["metadata"], output_dir)

    return return_result
