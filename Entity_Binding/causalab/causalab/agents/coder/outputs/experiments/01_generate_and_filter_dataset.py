"""
Generate and filter counterfactual datasets for the MCQA task.

This script:
1. Generates counterfactual datasets using MCQA counterfactual generators
2. Loads language model
3. Filters to keep only examples where model succeeds on both original and counterfactual
4. Saves both original and filtered datasets with metadata
"""

import torch
import os
import argparse
import json
from pathlib import Path

from causalab.neural.pipeline import LMPipeline
from causalab.experiments.filter import filter_dataset
from causalab.causal.counterfactual_dataset import CounterfactualDataset

from causalab.agents.coder.outputs.causal_models import causal_model
from causalab.agents.coder.outputs.counterfactuals import COUNTERFACTUAL_GENERATORS
from causalab.agents.coder.outputs.checker import checker
from causalab.agents.coder.outputs.config import MAX_TASK_TOKENS, MAX_NEW_TOKENS


# ============================================================================
# Configuration
# ============================================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Get output dir for absolute paths
OUTPUT_DIR = Path(__file__).parent.parent


def main():
    parser = argparse.ArgumentParser(
        description="Generate and filter counterfactual datasets"
    )
    parser.add_argument("--model", type=str, required=True, help="Model name or path")
    parser.add_argument(
        "--size", type=int, default=256, help="Number of examples per dataset"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Batch size for filtering (default: same as size)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(OUTPUT_DIR / "datasets"),
        help="Output directory for datasets",
    )
    parser.add_argument(
        "--test", action="store_true", help="Test mode (8 examples, small batch)"
    )
    args = parser.parse_args()

    # Test mode settings
    if args.test:
        args.size = 8
        args.batch_size = 8
        print("\n⚠ TEST MODE: Using 8 examples and batch_size=8")

    if args.batch_size is None:
        args.batch_size = args.size

    print("=" * 80)
    print("GENERATE AND FILTER COUNTERFACTUAL DATASETS")
    print("=" * 80)
    print(f"\nModel: {args.model}")
    print(f"Dataset size: {args.size}")
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

    # Create Counterfactual Datasets
    print("\n" + "=" * 80)
    print("CREATING COUNTERFACTUAL DATASETS")
    print("=" * 80)

    print(f"\nGenerating {args.size} examples per dataset...")

    counterfactual_datasets = {}
    for name, generator in COUNTERFACTUAL_GENERATORS.items():
        counterfactual_datasets[name] = CounterfactualDataset.from_sampler(
            args.size, generator
        )

    print(f"\n✓ Created {len(counterfactual_datasets)} datasets:")
    for name, dataset in counterfactual_datasets.items():
        print(f"  - {name}: {len(dataset)} examples")

    # Filter Datasets
    print("\n" + "=" * 80)
    print("FILTERING DATASETS")
    print("=" * 80)

    print(
        "\nFiltering to examples where model answers both original and counterfactual correctly...\n"
    )

    filtered_datasets = {}
    for name, dataset in counterfactual_datasets.items():
        filtered_datasets[name] = filter_dataset(
            dataset, pipeline, causal_model, checker, batch_size=args.batch_size
        )
        print(f"  - {name}: {len(dataset)} → {len(filtered_datasets[name])}")

    total_original = sum(len(dataset) for dataset in counterfactual_datasets.values())
    total_kept = sum(len(dataset) for dataset in filtered_datasets.values())
    keep_rate = total_kept / total_original * 100 if total_original > 0 else 0

    print("\nFiltering summary:")
    print(f"  Original: {total_original} examples")
    print(f"  Kept: {total_kept} examples ({keep_rate:.1f}%)")

    if keep_rate < 75:
        print(
            "\n⚠ WARNING: Keep rate is unexpectedly low. This is unexpected and usually indicates a problem with the way you've set up the task."
        )

    # Save Datasets
    print("\n" + "=" * 80)
    print("SAVING DATASETS")
    print("=" * 80)

    os.makedirs(args.output_dir, exist_ok=True)

    for name, dataset in counterfactual_datasets.items():
        dataset_dir = os.path.join(args.output_dir, name)

        # Save original dataset
        original_dir = os.path.join(dataset_dir, "original_dataset")
        os.makedirs(original_dir, exist_ok=True)
        dataset.dataset.save_to_disk(original_dir)
        print(f"\n✓ Saved original dataset: {original_dir}")

        # Save filtered dataset
        if name in filtered_datasets:
            filtered_dir = os.path.join(dataset_dir, "filtered_dataset")
            os.makedirs(filtered_dir, exist_ok=True)
            filtered_datasets[name].dataset.save_to_disk(filtered_dir)
            print(f"✓ Saved filtered dataset: {filtered_dir}")

    # Save metadata
    metadata = {
        "model": args.model,
        "dataset_size": args.size,
        "batch_size": args.batch_size,
        "total_original": total_original,
        "total_kept": total_kept,
        "keep_rate": keep_rate,
        "datasets": {
            name: {
                "original_size": len(counterfactual_datasets[name]),
                "filtered_size": len(filtered_datasets[name])
                if name in filtered_datasets
                else 0,
            }
            for name in counterfactual_datasets.keys()
        },
    }

    metadata_path = os.path.join(args.output_dir, "dataset_metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n✓ Saved metadata: {metadata_path}")

    print("\n" + "=" * 80)
    print("COMPLETE")
    print("=" * 80)
    print(f"\nOutput directory: {args.output_dir}")
    print(
        "\nNext step: Run 02_run_interventions.py to perform interventions on filtered datasets"
    )


if __name__ == "__main__":
    main()
