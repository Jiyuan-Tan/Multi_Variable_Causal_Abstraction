"""
Train attention head masks using DBM (Desiderata-Based Masking).

This script:
1. Loads filtered datasets from step 1
2. Trains binary masks over attention heads to identify which heads carry causal variables
3. Evaluates on test data to check generalization
4. Generates mask heatmaps and analysis
"""

import torch
import os
import argparse
import pickle
import json
from pathlib import Path
from datasets import load_from_disk, Dataset

from causalab.neural.pipeline import LMPipeline
from causalab.experiments.train import train_interventions
from causalab.experiments.filter import filter_dataset
from causalab.experiments.interchange_targets import build_attention_head_targets
from causalab.experiments import plot_attention_head_mask, get_selected_heads
from causalab.neural.token_position_builder import get_all_tokens, TokenPosition
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
    parser = argparse.ArgumentParser(description="Train attention head masks with DBM")
    parser.add_argument("--model", type=str, required=True, help="Model name or path")
    parser.add_argument(
        "--dataset", type=str, required=True, help="Path to filtered dataset directory"
    )
    parser.add_argument(
        "--target-var", type=str, required=True, help="Target variable to localize"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(OUTPUT_DIR / "results" / "dbm"),
        help="Output directory",
    )
    parser.add_argument(
        "--epochs", type=int, default=20, help="Number of training epochs"
    )
    parser.add_argument(
        "--regularization", type=float, default=0.1, help="Sparsity regularization"
    )
    parser.add_argument(
        "--batch-size", type=int, default=32, help="Training batch size"
    )
    parser.add_argument(
        "--eval-batch-size", type=int, default=512, help="Evaluation batch size"
    )
    parser.add_argument(
        "--test", action="store_true", help="Test mode (2 epochs, small data)"
    )
    args = parser.parse_args()

    # Test mode settings
    if args.test:
        args.epochs = 2
        args.batch_size = 8
        args.eval_batch_size = 8
        print("\n⚠ TEST MODE: 2 epochs, batch_size=8")

    print("=" * 80)
    print("TRAIN ATTENTION HEAD MASKS WITH DBM")
    print("=" * 80)
    print(f"\nModel: {args.model}")
    print(f"Dataset: {args.dataset}")
    print(f"Target variable: {args.target_var}")
    print(f"Epochs: {args.epochs}")
    print(f"Regularization: {args.regularization}")
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
    print(f"  Heads per layer: {pipeline.get_num_attention_heads()}")

    # Load Datasets
    print("\n" + "=" * 80)
    print("LOADING DATASETS")
    print("=" * 80)

    # Load training dataset
    dataset_name = Path(args.dataset).parent.name
    hf_dataset = load_from_disk(args.dataset)
    if not isinstance(hf_dataset, Dataset):
        raise TypeError(f"Expected Dataset, got {type(hf_dataset).__name__}")
    train_dataset = CounterfactualDataset(dataset=hf_dataset, id=dataset_name)

    print(f"✓ Train dataset loaded: {dataset_name} ({len(train_dataset)} examples)")

    # Generate test dataset
    print("\nGenerating test dataset...")
    test_size = len(train_dataset) if not args.test else 8
    test_counterfactual_datasets = {}
    for name, generator in COUNTERFACTUAL_GENERATORS.items():
        if name == dataset_name:
            test_counterfactual_datasets[name] = CounterfactualDataset.from_sampler(
                test_size, generator
            )

    # Filter test datasets
    test_datasets = {}
    for name, test_cf_dataset in test_counterfactual_datasets.items():
        test_datasets[name] = filter_dataset(
            test_cf_dataset,
            pipeline,
            causal_model,
            checker,
            batch_size=args.eval_batch_size,
        )

    print(
        f"✓ Test dataset created: {dataset_name} ({len(test_datasets[dataset_name])} examples)"
    )

    # Save test dataset for reuse
    test_dataset_dir = os.path.join(args.output_dir, "test_dataset")
    os.makedirs(test_dataset_dir, exist_ok=True)
    test_datasets[dataset_name].dataset.save_to_disk(test_dataset_dir)

    # Configure DBM
    print("\n" + "=" * 80)
    print("CONFIGURING DBM")
    print("=" * 80)

    num_heads = pipeline.get_num_attention_heads()
    num_layers = pipeline.model.config.num_hidden_layers

    dbm_config = {
        "id": f"{args.model.replace('/', '_')}_dbm_{args.target_var}",
        "intervention_type": "mask",
        "featurizer_kwargs": {"tie_masks": True},
        "train_batch_size": args.batch_size,
        "eval_batch_size": args.eval_batch_size,
        "training_epoch": args.epochs,
        "masking": {
            "regularization_coefficient": args.regularization,
        },
        "output_scores": False,
    }

    print("Configuration:")
    print(f"  Layers: {num_layers}")
    print(f"  Heads per layer: {num_heads}")
    print(f"  Total heads: {num_layers * num_heads}")
    print(f"  Training epochs: {args.epochs}")
    print(f"  Regularization: {args.regularization}")

    # Create token position that includes all tokens
    all_tokens = TokenPosition(
        lambda x: get_all_tokens(x, pipeline), pipeline, id="all_tokens"
    )

    # Build attention head targets - all heads in one target for DBM
    layers = list(range(num_layers))
    heads = list(range(num_heads))

    targets = build_attention_head_targets(
        pipeline=pipeline,
        layers=layers,
        heads=heads,
        token_position=all_tokens,
        mode="one_target_all_units",
    )

    # Train Masks
    print("\n" + "=" * 80)
    print("TRAINING MASKS")
    print("=" * 80)

    print(f"\nTraining binary masks to localize '{args.target_var}' variable...")
    print("This may take 5-15 minutes...\n")

    results = train_interventions(
        causal_model=causal_model,
        interchange_targets=targets,
        train_dataset_path=args.dataset,
        test_dataset_path=test_dataset_dir,
        pipeline=pipeline,
        target_variable_group=(args.target_var,),
        output_dir=args.output_dir,
        metric=checker,
        config=dbm_config,
    )

    print("\n✓ Training complete!")

    # Extract scores and feature indices
    train_score = results.get("avg_train_score", 0.0)
    test_score = results.get("avg_test_score", 0.0)

    # Get feature indices for visualization
    results_by_key = results.get("results_by_key", {})
    feature_indices = {}
    for _, key_results in results_by_key.items():
        if "feature_indices" in key_results:
            feature_indices.update(key_results["feature_indices"])

    # Results
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    print(f"\nVariable: {args.target_var}")
    print(f"Dataset: {dataset_name}")
    print("\nInterchange Intervention Accuracy:")
    print(f"  Train: {train_score:.3f}")
    print(f"  Test:  {test_score:.3f}")

    if test_score < train_score - 0.1:
        print("\n⚠ WARNING: Significant gap between train and test performance")
        print("  Consider increasing regularization or using more training data")
    elif test_score >= 0.8:
        print(
            "\n✓ Strong generalization! Learned masks identify robust attention heads"
        )

    # Save Results
    print("\n" + "=" * 80)
    print("SAVING RESULTS")
    print("=" * 80)

    os.makedirs(args.output_dir, exist_ok=True)

    # Save mask heatmap
    print("\nGenerating mask heatmap...")
    heatmap_path = os.path.join(
        args.output_dir, f"{dataset_name}_{args.target_var}_mask.png"
    )
    plot_attention_head_mask(
        feature_indices=feature_indices,
        layers=layers,
        heads=heads,
        title=f"DBM Mask: {args.target_var}",
        save_path=heatmap_path,
    )
    print(f"✓ Saved heatmap: {heatmap_path}")

    # Get selected heads from mask
    selected_heads = get_selected_heads(feature_indices)
    print(f"\n✓ Selected {len(selected_heads)} attention heads")

    if not selected_heads:
        print(
            "\n⚠ WARNING: No attention heads were selected. This is unexpected and usually indicates a problem with the way you've set up the experiment."
        )

    # Save metadata
    metadata = {
        "model": args.model,
        "dataset": args.dataset,
        "target_variable": args.target_var,
        "epochs": args.epochs,
        "regularization": args.regularization,
        "train_score": float(train_score),
        "test_score": float(test_score),
        "selected_heads": selected_heads,
        "config": dbm_config,
    }

    metadata_path = os.path.join(args.output_dir, "dbm_metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"✓ Saved metadata: {metadata_path}")

    # Save full results (includes both train and test evaluation)
    results_path = os.path.join(args.output_dir, "dbm_results.pkl")
    with open(results_path, "wb") as f:
        pickle.dump(results, f)
    print(f"✓ Saved results: {results_path}")

    print("\n" + "=" * 80)
    print("COMPLETE")
    print("=" * 80)
    print(f"\nOutput directory: {args.output_dir}")
    print("\nThe heatmap shows which attention heads were selected by DBM.")
    print(f"Selected heads ({len(selected_heads)}): {selected_heads}")
    print("These heads carry the causal variable.")


if __name__ == "__main__":
    main()
