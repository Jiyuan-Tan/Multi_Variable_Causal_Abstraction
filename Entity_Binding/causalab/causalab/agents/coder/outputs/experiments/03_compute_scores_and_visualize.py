"""
Compute causal scores and generate visualizations from raw results.

This script:
1. Loads raw_results.pkl from step 2
2. Loads causal model and dataset metadata
3. Uses compute_intervention_scores() to add causal analysis
4. Generates heatmaps

This is the cheap, fast step that can be repeated with different analyses
without re-running the expensive GPU interventions from step 2.
"""

import os
import argparse
import pickle
import json
from pathlib import Path

from causalab.causal.counterfactual_dataset import CounterfactualDataset
from datasets import load_from_disk, Dataset
from causalab.experiments.metric import causal_score_intervention_outputs
from causalab.experiments import (
    plot_residual_stream_heatmap,
    print_residual_stream_patching_analysis,
)

from causalab.agents.coder.outputs.causal_models import causal_model
from causalab.agents.coder.outputs.checker import checker


def main():
    parser = argparse.ArgumentParser(
        description="Compute scores and visualize from raw results"
    )
    parser.add_argument(
        "--results", type=str, required=True, help="Path to raw_results.pkl from step 2"
    )
    parser.add_argument(
        "--target-vars", nargs="+", required=True, help="Target variables to analyze"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: same as results)",
    )
    args = parser.parse_args()

    # Determine output directory
    if args.output_dir is None:
        args.output_dir = os.path.dirname(args.results)

    print("=" * 80)
    print("COMPUTE SCORES AND VISUALIZE")
    print("=" * 80)
    print(f"\nRaw results: {args.results}")
    print(f"Target variables: {args.target_vars}")
    print(f"Output directory: {args.output_dir}")

    # Load Raw Results
    print("\n" + "=" * 80)
    print("LOADING RAW RESULTS")
    print("=" * 80)

    with open(args.results, "rb") as f:
        raw_results = pickle.load(f)

    print(f"✓ Loaded raw results from: {args.results}")

    # Load Metadata
    metadata_path = os.path.join(
        os.path.dirname(args.results), "experiment_metadata.json"
    )
    if os.path.exists(metadata_path):
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
        print(f"✓ Loaded metadata from: {metadata_path}")
        print(f"  Model: {metadata.get('model', 'unknown')}")
        print(
            f"  Total interventions: {metadata.get('total_interventions', 'unknown')}"
        )
    else:
        print("⚠ Warning: No metadata file found")
        metadata = {}

    # Load Dataset
    print("\n" + "=" * 80)
    print("LOADING DATASET")
    print("=" * 80)

    dataset_path = metadata.get("dataset_path")
    if dataset_path:
        dataset_name = Path(dataset_path).parent.name
        hf_dataset = load_from_disk(dataset_path)
        if not isinstance(hf_dataset, Dataset):
            raise TypeError(f"Expected Dataset, got {type(hf_dataset).__name__}")
        dataset = CounterfactualDataset(dataset=hf_dataset, id=dataset_name)
        print(f"✓ Loaded dataset: {dataset_name} ({len(dataset)} examples)")
    else:
        raise ValueError("dataset_path not found in metadata")

    # Compute Scores
    print("\n" + "=" * 80)
    print("COMPUTING CAUSAL SCORES")
    print("=" * 80)

    print("\nUsing causal_score_intervention_outputs() utility function")
    print("This computes causal model scores for raw intervention results\n")

    # raw_results structure from step 2: {dataset_name: {(layer, pos_id): {"string": [...]}}}
    # We need to extract the inner dict for the dataset we loaded
    if dataset_name in raw_results:
        dataset_raw_results = raw_results[dataset_name]
    else:
        # If raw_results is already the inner format, use directly
        dataset_raw_results = raw_results

    # Convert target_vars to tuple groups for new API
    target_variable_groups = [tuple([var]) for var in args.target_vars]

    result = causal_score_intervention_outputs(
        raw_results=dataset_raw_results,
        dataset=dataset,
        causal_model=causal_model,
        target_variable_groups=target_variable_groups,
        metric=checker,
    )

    # Extract scores from results
    scores = {key: res["avg_score"] for key, res in result["results_by_key"].items()}

    print(f"✓ Computed scores for {len(scores)} interventions")

    # Generate Visualizations
    print("\n" + "=" * 80)
    print("GENERATING VISUALIZATIONS")
    print("=" * 80)

    heatmap_dir = os.path.join(args.output_dir, "heatmaps")
    os.makedirs(heatmap_dir, exist_ok=True)

    # Extract layers and positions from keys
    layers = sorted(set(k[0] for k in scores.keys()))
    positions = sorted(set(k[1] for k in scores.keys()))

    var_str = "-".join(args.target_vars)
    save_path = os.path.join(heatmap_dir, f"{dataset_name}_{var_str}_heatmap.png")

    plot_residual_stream_heatmap(
        scores=scores,
        layers=layers,
        token_position_ids=positions,
        title=f"Residual Stream Intervention Accuracy ({var_str})",
        save_path=save_path,
    )
    print(f"✓ Saved heatmap: {save_path}")

    # Generate text-based analysis
    text_analysis_path = os.path.join(
        heatmap_dir, f"{dataset_name}_{var_str}_analysis.txt"
    )
    print_residual_stream_patching_analysis(
        scores=scores,
        layers=layers,
        token_position_ids=positions,
        title=f"Residual Stream Intervention Analysis ({var_str})",
        save_path=text_analysis_path,
    )
    print(f"✓ Saved text analysis: {text_analysis_path}")

    # Save scored results
    avg_score = sum(scores.values()) / len(scores) if scores else 0.0

    scored_results = {
        "scores_by_key": {str(k): v for k, v in scores.items()},
        "target_variables": args.target_vars,
        "dataset_name": dataset_name,
        "avg_score": avg_score,
    }

    scored_results_path = os.path.join(args.output_dir, "results_with_scores.json")
    with open(scored_results_path, "w") as f:
        json.dump(scored_results, f, indent=2)
    print(f"✓ Saved scored results: {scored_results_path}")

    # Also save as pickle for compatibility
    scored_pkl_path = os.path.join(args.output_dir, "results_with_scores.pkl")
    with open(scored_pkl_path, "wb") as f:
        pickle.dump({"scores": scores, **scored_results}, f)
    print(f"✓ Saved scored results (pickle): {scored_pkl_path}")

    print("\n" + "=" * 80)
    print("COMPLETE")
    print("=" * 80)
    print(f"\nOutput directory: {args.output_dir}")
    print("  - heatmaps/ (includes .png heatmap and .txt text analysis)")
    print("  - results_with_scores.json")
    print("  - results_with_scores.pkl")
    print(f"\nAverage score: {avg_score:.3f}")
    print("\nYou can re-run this script with different --target-vars without")
    print("re-running the expensive interventions!")


if __name__ == "__main__":
    main()
