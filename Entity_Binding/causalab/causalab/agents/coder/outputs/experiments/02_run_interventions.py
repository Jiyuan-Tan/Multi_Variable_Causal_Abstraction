"""
Run interventions and save raw results for the MCQA task (NEW API - no scoring yet).

This script:
1. Loads filtered dataset from step 1
2. Loads language model
3. Creates token positions
4. Runs PatchResidualStream interventions across all (layer, position) combinations
5. Saves RAW results WITHOUT computing causal scores (done in step 3)

KEY API CHANGES:
- Experiment constructor: NO causal_model, NO checker
- Config must include "id" field
- perform_interventions(): NO target_variables_list parameter
- Returns raw results only (scoring done separately in step 3)
"""

import torch
import os
import argparse
import pickle
import json
from pathlib import Path

from causalab.neural.pipeline import LMPipeline
from causalab.neural.pyvene_core.interchange import run_interchange_interventions
from causalab.experiments.interchange_targets import build_residual_stream_targets
from causalab.causal.counterfactual_dataset import CounterfactualDataset
from datasets import load_from_disk, Dataset

from causalab.agents.coder.outputs.token_positions import create_token_positions
from causalab.agents.coder.outputs.config import MAX_TASK_TOKENS, MAX_NEW_TOKENS
from causalab.agents.coder.outputs.templates import TEMPLATES

# ============================================================================
# Configuration
# ============================================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Get output dir for absolute paths
OUTPUT_DIR = Path(__file__).parent.parent


def main():
    parser = argparse.ArgumentParser(
        description="Run interventions and save raw results"
    )
    parser.add_argument("--model", type=str, required=True, help="Model name or path")
    parser.add_argument(
        "--dataset", type=str, required=True, help="Path to filtered dataset directory"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(OUTPUT_DIR / "results"),
        help="Output directory for results",
    )
    parser.add_argument(
        "--batch-size", type=int, default=512, help="Batch size for interventions"
    )
    parser.add_argument(
        "--test", action="store_true", help="Test mode (layer 0 only, small batch)"
    )
    args = parser.parse_args()

    # Test mode settings
    if args.test:
        args.batch_size = 8
        print("\n⚠ TEST MODE: Layer 0 only, batch_size=8")

    print("=" * 80)
    print("RUN INTERVENTIONS")
    print("=" * 80)
    print(f"\nModel: {args.model}")
    print(f"Dataset: {args.dataset}")
    print(f"Batch size: {args.batch_size}")
    print(f"Output directory: {args.output_dir}")

    # Load Model
    print("\n" + "=" * 80)
    print("LOADING MODEL")
    print("=" * 80)

    pipeline = LMPipeline(
        args.model,
        max_new_tokens=MAX_NEW_TOKENS,
        device=DEVICE,
        max_length=MAX_TASK_TOKENS,
    )

    print(f"✓ Model loaded: {args.model}")
    print(f"  Device: {pipeline.model.device}")
    print(f"  Layers: {pipeline.model.config.num_hidden_layers}")

    # Load Filtered Dataset
    print("\n" + "=" * 80)
    print("LOADING DATASET")
    print("=" * 80)

    filtered_datasets = {}
    dataset_name = Path(args.dataset).parent.name
    hf_dataset = load_from_disk(args.dataset)
    if not isinstance(hf_dataset, Dataset):
        raise TypeError(f"Expected Dataset, got {type(hf_dataset).__name__}")
    filtered_datasets[dataset_name] = CounterfactualDataset(
        dataset=hf_dataset, id=dataset_name
    )
    print(
        f"✓ Loaded dataset: {dataset_name} ({len(filtered_datasets[dataset_name])} examples)"
    )

    # Create Token Positions
    print("\n" + "=" * 80)
    print("CREATING TOKEN POSITIONS")
    print("=" * 80)

    template = TEMPLATES[0]
    token_position_factories = create_token_positions(pipeline, template)
    token_positions = [
        factory(pipeline) for factory in token_position_factories.values()
    ]

    print(f"✓ Created {len(token_positions)} token positions:")
    for name in token_position_factories.keys():
        print(f"  - {name}")

    # Run Interventions
    print("\n" + "=" * 80)
    print("RUNNING INTERVENTIONS")
    print("=" * 80)

    num_layers = pipeline.model.config.num_hidden_layers

    # Determine which layers to run
    if args.test:
        layers = [0]  # Test mode: only layer 0
        print("\n⚠ TEST MODE: Running layer 0 only")
    else:
        layers = [-1] + list(range(num_layers))  # All layers including input (-1)

    total_interventions = len(layers) * len(token_positions) * len(filtered_datasets)

    print(f"\nLayers: {len(layers)}")
    print(f"Token positions: {len(token_positions)}")
    print(f"Datasets: {len(filtered_datasets)}")
    print(f"Total interventions: {total_interventions}")
    print("\nThis may take some time...")

    # Build targets using new API
    targets = build_residual_stream_targets(
        pipeline=pipeline,
        layers=layers,
        token_positions=token_positions,
        mode="one_target_per_unit",
    )

    print("\nRunning interventions...")

    # Run interventions for each dataset
    all_raw_results = {}
    for dataset_name, dataset in filtered_datasets.items():
        raw_results = {}
        for key, target in targets.items():
            raw_results[key] = run_interchange_interventions(
                pipeline=pipeline,
                counterfactual_dataset=dataset,
                interchange_target=target,
                batch_size=args.batch_size,
                output_scores=False,
            )
        all_raw_results[dataset_name] = raw_results

    raw_results = all_raw_results

    print("\n✓ Interventions complete!")

    # Save Raw Results
    print("\n" + "=" * 80)
    print("SAVING RAW RESULTS")
    print("=" * 80)

    os.makedirs(args.output_dir, exist_ok=True)

    # Save raw results
    raw_results_path = os.path.join(args.output_dir, "raw_results.pkl")
    with open(raw_results_path, "wb") as f:
        pickle.dump(raw_results, f)
    print(f"\n✓ Saved raw results: {raw_results_path}")

    # Save metadata
    metadata = {
        "model": args.model,
        "dataset_path": args.dataset,
        "batch_size": args.batch_size,
        "num_layers": num_layers,
        "layers_used": layers,
        "num_token_positions": len(token_positions),
        "token_position_names": list(token_position_factories.keys()),
        "num_datasets": len(filtered_datasets),
        "total_interventions": total_interventions,
        "test_mode": args.test,
    }

    metadata_path = os.path.join(args.output_dir, "experiment_metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"✓ Saved metadata: {metadata_path}")

    print("\n" + "=" * 80)
    print("COMPLETE")
    print("=" * 80)
    print(f"\nOutput directory: {args.output_dir}")
    print("\nRaw results saved (no scoring yet).")
    print("Next step: Run 03_compute_scores_and_visualize.py to compute causal scores")
    print("\nExample:")
    print(
        f"  uv run experiments/03_compute_scores_and_visualize.py --results {raw_results_path}"
    )


if __name__ == "__main__":
    main()
