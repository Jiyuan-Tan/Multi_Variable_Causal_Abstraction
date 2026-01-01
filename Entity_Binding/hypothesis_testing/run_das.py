#!/usr/bin/env python
"""
DAS Testing for Entity Binding

This script:
1. Creates a randomly sampled entity binding dataset with 10 groups and 3 entities
2. Filters the dataset based on model predictions (base and counterfactual must both be correct)
3. Checks task accuracy (must be >80%, otherwise suggests larger model)
4. Runs boundless DAS to find alignment of positional_query_group
5. Saves weights and IIA accuracy for all alignments tested

Usage:
    python run_das.py --model MODEL_ID [--gpu GPU_ID] [--size DATASET_SIZE]
    python run_das.py --model gpt2 --gpu 0 --size 1024
"""

import sys
import os
import argparse
import torch
import json
import pickle
from pathlib import Path
from typing import Dict, Any

# Add the causalab package to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'causalab'))

from causalab.tasks.entity_binding.config import EntityBindingTaskConfig, create_sample_action_config
from causalab.tasks.entity_binding.causal_models import create_positional_causal_model, sample_valid_entity_binding_input
from causalab.tasks.entity_binding.counterfactuals import swap_query_group
from causalab.tasks.entity_binding.templates import TemplateProcessor
from causalab.neural.pipeline import LMPipeline
from causalab.causal.counterfactual_dataset import CounterfactualDataset
from causalab.experiments.filter import filter_dataset
from causalab.experiments.interchange_targets import build_residual_stream_targets
from causalab.experiments.jobs.boundless_DAS_feature_mask_grid import train_boundless_DAS
from causalab.neural.token_position_builder import TokenPosition, get_last_token_index


# ========== Register Qwen3 support for pyvene ==========
# Qwen3 has the same architecture as Qwen2, so we reuse the same mappings
def register_qwen3_for_pyvene():
    """Register Qwen3 model types with pyvene's type mappings."""
    try:
        import transformers.models.qwen3.modeling_qwen3 as qwen3_modeling
        from pyvene.models.intervenable_modelcard import type_to_module_mapping, type_to_dimension_mapping
        from pyvene.models.qwen2.modelings_intervenable_qwen2 import (
            qwen2_type_to_module_mapping,
            qwen2_type_to_dimension_mapping,
            qwen2_lm_type_to_module_mapping,
            qwen2_lm_type_to_dimension_mapping,
            qwen2_classifier_type_to_module_mapping,
            qwen2_classifier_type_to_dimension_mapping,
        )
        
        # Register Qwen3 models using Qwen2 mappings (same architecture)
        if hasattr(qwen3_modeling, 'Qwen3Model'):
            type_to_module_mapping[qwen3_modeling.Qwen3Model] = qwen2_type_to_module_mapping
            type_to_dimension_mapping[qwen3_modeling.Qwen3Model] = qwen2_type_to_dimension_mapping
        
        if hasattr(qwen3_modeling, 'Qwen3ForCausalLM'):
            type_to_module_mapping[qwen3_modeling.Qwen3ForCausalLM] = qwen2_lm_type_to_module_mapping
            type_to_dimension_mapping[qwen3_modeling.Qwen3ForCausalLM] = qwen2_lm_type_to_dimension_mapping
        
        if hasattr(qwen3_modeling, 'Qwen3ForSequenceClassification'):
            type_to_module_mapping[qwen3_modeling.Qwen3ForSequenceClassification] = qwen2_classifier_type_to_module_mapping
            type_to_dimension_mapping[qwen3_modeling.Qwen3ForSequenceClassification] = qwen2_classifier_type_to_dimension_mapping
        
        print("Successfully registered Qwen3 support for pyvene")
    except ImportError as e:
        print(f"Warning: Could not register Qwen3 for pyvene: {e}")
    except Exception as e:
        print(f"Warning: Error registering Qwen3 for pyvene: {e}")

register_qwen3_for_pyvene()
# ========================================================


def create_custom_config() -> EntityBindingTaskConfig:
    """
    Create a custom entity binding config with 10 groups and 3 entities per group.
    
    Uses the action template structure (person, object, location) with expanded pools.
    """
    config = create_sample_action_config()
    
    # Set max groups to 10
    config.max_groups = 10
    config.max_entities_per_group = 3  # person, object, location
    
    # Expand entity pools to support 10 groups
    # Person pool
    config.entity_pools[0] = [
        "Pete", "Ann", "Bob", "Sue", "Tim", "Kate", "Dan", "Lily",
        "Max", "Eva", "Sam", "Zoe", "Leo", "Mia", "Noah", "Ava",
        "Ben", "Liz", "Tom", "Joy"
    ]
    
    # Object pool
    config.entity_pools[1] = [
        "jam", "water", "book", "coin", "pen", "key", "phone", "watch",
        "cup", "box", "bag", "hat", "map", "card", "lamp", "ball",
        "rope", "tape", "tool", "clip"
    ]
    
    # Location pool
    config.entity_pools[2] = [
        "cup", "box", "table", "shelf", "drawer", "bag", "pocket", "basket",
        "desk", "chair", "floor", "rack", "case", "tray", "bin", "stand",
        "cabinet", "corner", "bench", "counter"
    ]
    
    # Add instruction wrapper for better performance
    config.prompt_prefix = "We will ask a question about the following sentences.\n\n"
    config.statement_question_separator = "\n\n"
    config.prompt_suffix = "\nAnswer:"
    
    return config


def create_dataset_with_n_groups(config: EntityBindingTaskConfig, num_groups: int, size: int) -> CounterfactualDataset:
    """
    Create a dataset with exactly num_groups active groups.
    
    Args:
        config: Task configuration
        num_groups: Number of active groups to use
        size: Number of examples to generate
        
    Returns:
        CounterfactualDataset with the specified number of groups
    """
    from causalab.tasks.entity_binding.causal_models import create_direct_causal_model
    import random
    
    def generator():
        # Sample input and ensure it has exactly num_groups
        max_attempts = 100
        for attempt in range(max_attempts):
            input_sample = sample_valid_entity_binding_input(config, ensure_positional_uniqueness=True)
            if input_sample["active_groups"] == num_groups:
                break
        else:
            # Force active_groups if we couldn't get it naturally
            input_sample["active_groups"] = num_groups
            # Ensure query_group is valid
            query_group = input_sample.get("query_group", 0)
            if query_group >= num_groups:
                input_sample["query_group"] = query_group % num_groups
        
        # Regenerate raw_input with correct active_groups
        model = create_direct_causal_model(config)
        model.new_raw_input(input_sample)
        
        # Create counterfactual by swapping with another group (same logic as swap_query_group)
        active_groups = input_sample["active_groups"]
        query_group = input_sample["query_group"]
        
        # Choose a different group to swap with
        other_groups = [g for g in range(active_groups) if g != query_group]
        if not other_groups or len(other_groups) == 0:
            # Only one group, create a new sample with same number of groups
            counterfactual_input = sample_valid_entity_binding_input(config, ensure_positional_uniqueness=True)
            counterfactual_input["active_groups"] = num_groups
            query_group_cf = counterfactual_input.get("query_group", 0)
            if query_group_cf >= num_groups:
                counterfactual_input["query_group"] = query_group_cf % num_groups
            model.new_raw_input(counterfactual_input)
            return {"input": input_sample, "counterfactual_inputs": [counterfactual_input]}
        
        swap_group = random.choice(other_groups)
        
        # Create counterfactual by swapping entity groups
        counterfactual = input_sample.copy()
        entities_per_group = input_sample["entities_per_group"]
        for e in range(entities_per_group):
            key_query = f"entity_g{query_group}_e{e}"
            key_swap = f"entity_g{swap_group}_e{e}"
            counterfactual[key_query] = input_sample[key_swap]
            counterfactual[key_swap] = input_sample[key_query]
        
        # Update query_group to follow where the original query entity moved
        counterfactual["query_group"] = swap_group
        if "raw_input" in counterfactual:
            del counterfactual["raw_input"]
        
        model.new_raw_input(counterfactual)
        
        return {"input": input_sample, "counterfactual_inputs": [counterfactual]}
    
    return CounterfactualDataset.from_sampler(size, generator, id=f"entity_binding_{num_groups}groups")


def checker(neural_output, causal_output):
    """Check if neural network output matches causal model output."""
    neural_str = neural_output["string"].strip().lower()
    causal_str = causal_output.strip().lower()
    return causal_str in neural_str or neural_str in causal_str


def filter_dataset_with_accuracy_check(
    dataset: CounterfactualDataset,
    pipeline: LMPipeline,
    causal_model,
    min_accuracy: float = 0.8,
    batch_size: int = 32,
    verbose: bool = True
):
    """
    Filter dataset and check if accuracy meets threshold.
    
    Args:
        dataset: Dataset to filter
        pipeline: Language model pipeline
        causal_model: Causal model for evaluation
        min_accuracy: Minimum accuracy threshold (default 0.8)
        batch_size: Batch size for filtering
        verbose: Whether to print progress
        
    Returns:
        Tuple of (filtered_dataset, stats_dict)
        stats_dict contains: original_size, filtered_size, accuracy, passed_threshold
        
    Raises:
        ValueError: If accuracy is below threshold and we should suggest a larger model
    """
    if verbose:
        print(f"Filtering dataset ({len(dataset)} examples)...")
    
    filtered_dataset = filter_dataset(
        dataset=dataset,
        pipeline=pipeline,
        causal_model=causal_model,
        metric=checker,
        batch_size=batch_size,
        validate_counterfactuals=True
    )
    accuracy = len(filtered_dataset) / len(dataset) if len(dataset) > 0 else 0.0
    
    stats = {
        "original_size": len(dataset),
        "filtered_size": len(filtered_dataset),
        "accuracy": accuracy,
        "passed_threshold": accuracy >= min_accuracy
    }
    
    if verbose:
        print(f"  Original size: {stats['original_size']}")
        print(f"  Filtered size: {stats['filtered_size']}")
        print(f"  Accuracy: {accuracy:.1%}")
    
    if accuracy < min_accuracy:
        error_msg = (
            f"Task accuracy ({accuracy:.1%}) is below threshold ({min_accuracy:.1%}). "
            f"Only {stats['filtered_size']}/{stats['original_size']} examples passed filtering. "
            "Please use a larger model."
        )
        if verbose:
            print(f"\n⚠ WARNING: {error_msg}\n")
        raise ValueError(error_msg)
    
    return filtered_dataset, stats


def main():
    parser = argparse.ArgumentParser(
        description="Hypothesis testing for entity binding with Boundless DAS"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt2",
        help="HuggingFace model ID (default: gpt2)",
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=None,
        help="GPU ID to use (default: use cuda:0 if available, else cpu)",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=1024,
        help="Dataset size to generate (default: 1024)",
    )
    parser.add_argument(
        "--num-groups",
        type=int,
        default=10,
        help="Number of entity groups (default: 10)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory (default: auto-generated)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for filtering and training (default: 32)",
    )
    parser.add_argument(
        "--n-features",
        type=int,
        default=32,
        help="Number of features for DAS (default: 32)",
    )
    parser.add_argument(
        "--min-accuracy",
        type=float,
        default=0.8,
        help="Minimum accuracy threshold (default: 0.8)",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Test mode: reduced dataset size and batch size",
    )
    
    args = parser.parse_args()
    
    # Test mode overrides
    if args.test:
        args.size = 16
        args.batch_size = 8
        print("\n*** TEST MODE: size=16, batch_size=8 ***\n")
    
    # Auto-generate output path
    if args.output is None:
        test_suffix = "_test" if args.test else ""
        args.output = f"hypothesis_testing/outputs/{args.model.replace('/', '_')}{test_suffix}"
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup device
    if args.gpu is not None:
        device = f"cuda:{args.gpu}"
    elif torch.cuda.is_available():
        device = "cuda:0"
    else:
        device = "cpu"
    
    print("=" * 70)
    print("Entity Binding Hypothesis Testing with Boundless DAS")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Model: {args.model}")
    print(f"  Device: {device}")
    print(f"  Dataset size: {args.size}")
    print(f"  Number of groups: {args.num_groups}")
    print(f"  Entities per group: 3")
    print(f"  Output directory: {output_dir}")
    print(f"  Test mode: {args.test}")
    print()
    
    # Step 1: Create task configuration
    print("Step 1: Creating task configuration...")
    config = create_custom_config()
    print(f"  Max groups: {config.max_groups}")
    print(f"  Entities per group: {config.max_entities_per_group}")
    print(f"  Template: {config.statement_template}")
    print()
    
    # Step 2: Generate dataset
    print(f"Step 2: Generating dataset with {args.num_groups} groups...")
    dataset = create_dataset_with_n_groups(config, args.num_groups, args.size)
    print(f"  Generated {len(dataset)} examples")
    
    # Save partial result: raw dataset
    raw_dataset_path = output_dir / "raw_dataset"
    dataset.dataset.save_to_disk(str(raw_dataset_path))
    print(f"  Saved raw dataset to {raw_dataset_path}")
    
    # Save partial summary
    partial_summary = {
        "model": args.model,
        "device": device,
        "num_groups": args.num_groups,
        "entities_per_group": 3,
        "dataset_size": args.size,
        "raw_dataset_size": len(dataset),
        "step": "dataset_generated"
    }
    partial_summary_path = output_dir / "partial_summary.json"
    with open(partial_summary_path, "w") as f:
        json.dump(partial_summary, f, indent=2)
    print()
    
    # Show example
    if len(dataset) > 0:
        print("Example from dataset:")
        example = dataset[0]
        print(f"  Input:  {example['input']['raw_input']}...")
        print(f"  Counter: {example['counterfactual_inputs'][0]['raw_input']}...")
        print()
    
    # Step 3: Load model and causal model
    print(f"Step 3: Loading model {args.model}...")
    pipeline = LMPipeline(
        args.model,
        max_new_tokens=5,
        device=device,
        max_length=512,
    )
    pipeline.tokenizer.padding_side = "left"
    num_layers = pipeline.get_num_layers()
    print(f"  Model loaded ({num_layers} layers)")
    print()
    
    causal_model = create_positional_causal_model(config)
    print(f"  Causal model: {causal_model.id}")
    print()
    
    # Step 4: Filter dataset
    print("Step 4: Filtering dataset...")
    try:
        filtered_dataset, filter_stats = filter_dataset_with_accuracy_check(
            dataset,
            pipeline,
            causal_model,
            min_accuracy=args.min_accuracy,
            batch_size=args.batch_size,
            verbose=True
        )
        
        # Save partial result: filtered dataset
        filtered_dataset_path = output_dir / "filtered_dataset"
        filtered_dataset.dataset.save_to_disk(str(filtered_dataset_path))
        print(f"  Saved filtered dataset to {filtered_dataset_path}")
        
        # Update partial summary
        partial_summary.update({
            "filter_stats": filter_stats,
            "filtered_dataset_size": len(filtered_dataset),
            "step": "dataset_filtered"
        })
        with open(partial_summary_path, "w") as f:
            json.dump(partial_summary, f, indent=2)
        print()
    except ValueError as e:
        print(f"\n❌ Error: {e}")
        # Save error state to partial summary
        partial_summary.update({
            "error": str(e),
            "step": "filtering_failed"
        })
        with open(partial_summary_path, "w") as f:
            json.dump(partial_summary, f, indent=2)
        return 1
    
    # Step 5: Split into train/test (80/20)
    print("Step 5: Splitting dataset into train/test...")
    train_size = int(len(filtered_dataset) * 0.8)
    test_size = len(filtered_dataset) - train_size
    
    train_dataset = CounterfactualDataset(
        dataset=filtered_dataset.dataset.select(range(train_size)),
        id="train"
    )
    test_dataset = CounterfactualDataset(
        dataset=filtered_dataset.dataset.select(range(train_size, len(filtered_dataset))),
        id="test"
    )
    
    print(f"  Train size: {len(train_dataset)}")
    print(f"  Test size: {len(test_dataset)}")
    print()
    
    # Save datasets
    train_path = output_dir / "train_dataset"
    test_path = output_dir / "test_dataset"
    train_dataset.dataset.save_to_disk(str(train_path))
    test_dataset.dataset.save_to_disk(str(test_path))
    print(f"  Saved datasets to {output_dir}")
    
    # Update partial summary
    partial_summary.update({
        "train_size": len(train_dataset),
        "test_size": len(test_dataset),
        "step": "train_test_split"
    })
    with open(partial_summary_path, "w") as f:
        json.dump(partial_summary, f, indent=2)
    print()
    
    # Step 6: Setup for boundless DAS
    print("Step 6: Setting up Boundless DAS...")
    
    # Training configuration (needed for saving config)
    train_config = {
        "train_batch_size": args.batch_size,
        "evaluation_batch_size": args.batch_size,
        "training_epoch": 10,
        "init_lr": 0.001,
        "masking": {
            "regularization_coefficient": 0.01,
            "temperature_annealing_fraction": 0.5,
            "temperature_schedule": (1.0, 0.001),
        },
    }
    
    # Create token position (last token)
    def last_token_indexer(input_dict, is_original=True):
        return get_last_token_index(input_dict["raw_input"], pipeline)
    
    token_position = TokenPosition(
        last_token_indexer,
        pipeline,
        id="last_token"
    )
    token_positions = [token_position]
    
    # Build residual stream targets for all layers
    layers = list(range(num_layers))
    residual_targets = build_residual_stream_targets(
        pipeline=pipeline,
        layers=layers,
        token_positions=token_positions,
        mode="one_target_per_layer",
    )
    
    # Convert to {layer: target} format
    residual_targets_by_layer = {key[0]: target for key, target in residual_targets.items()}
    
    print(f"  Created targets for {len(residual_targets_by_layer)} layers")
    print(f"  Token position: last_token")
    print(f"  Target variable: positional_query_group")
    
    # Save partial result: DAS configuration
    das_config = {
        "num_layers": num_layers,
        "layers": layers,
        "token_position": "last_token",
        "target_variable": "positional_query_group",
        "n_features": args.n_features,
        "train_config": train_config
    }
    das_config_path = output_dir / "das_config.json"
    with open(das_config_path, "w") as f:
        json.dump(das_config, f, indent=2)
    print(f"  Saved DAS configuration to {das_config_path}")
    
    # Update partial summary
    partial_summary.update({
        "num_layers": num_layers,
        "n_features": args.n_features,
        "step": "das_setup_complete"
    })
    with open(partial_summary_path, "w") as f:
        json.dump(partial_summary, f, indent=2)
    print()
    
    # Step 7: Run boundless DAS
    print("Step 7: Running Boundless DAS...")
    print("  This may take a while depending on model size and dataset size...")
    print()
    
    das_output_dir = output_dir / "boundless_das"
    das_output_dir.mkdir(exist_ok=True)
    
    # Update partial summary before DAS training
    partial_summary.update({
        "step": "das_training_started",
        "das_output_dir": str(das_output_dir)
    })
    with open(partial_summary_path, "w") as f:
        json.dump(partial_summary, f, indent=2)
    
    try:
        result = train_boundless_DAS(
            causal_model=causal_model,
            interchange_target=residual_targets_by_layer,
            train_dataset_path=str(train_path),
            test_dataset_path=str(test_path),
            pipeline=pipeline,
            target_variable_group=("positional_query_group",),
            output_dir=str(das_output_dir),
            metric=checker,
            n_features=args.n_features,
            config=train_config,
            save_results=True,
            verbose=True,
        )
        
        print()
        print("✓ Boundless DAS training complete!")
        print()
        
        # Update partial summary with IIA scores for each layer
        partial_summary.update({
            "step": "das_training_complete",
            "best_layer": result["metadata"]["best_layer"],
            "best_test_score": result["metadata"]["best_test_score"],
            "avg_test_score": result["metadata"]["avg_test_score"],
            "test_scores_by_layer": result["test_scores"],  # IIA accuracy for each layer
            "train_scores_by_layer": result["train_scores"],
        })
        with open(partial_summary_path, "w") as f:
            json.dump(partial_summary, f, indent=2)
        print(f"  Updated partial_summary.json with IIA scores for all layers")
        print()
        
        # Step 8: Save results summary
        print("Step 8: Saving results summary...")
        
        summary = {
            "model": args.model,
            "device": device,
            "num_groups": args.num_groups,
            "entities_per_group": 3,
            "dataset_size": args.size,
            "filter_stats": filter_stats,
            "train_size": len(train_dataset),
            "test_size": len(test_dataset),
            "num_layers": num_layers,
            "n_features": args.n_features,
            "target_variable": "positional_query_group",
            "best_layer": result["metadata"]["best_layer"],
            "best_test_score": result["metadata"]["best_test_score"],
            "avg_test_score": result["metadata"]["avg_test_score"],
            "test_scores_by_layer": result["test_scores"],
            "train_scores_by_layer": result["train_scores"],
        }
        
        # Save summary
        summary_path = output_dir / "summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        
        # Save full result
        result_path = output_dir / "boundless_das_result.pkl"
        with open(result_path, "wb") as f:
            pickle.dump(result, f)
        
        # Check for saved model weights
        models_dir = das_output_dir / "models"
        if models_dir.exists():
            model_paths = list(models_dir.glob("*"))
            summary["model_weights_path"] = str(models_dir)
            summary["num_saved_models"] = len(model_paths)
        
        print(f"  Summary saved to: {summary_path}")
        print(f"  Full result saved to: {result_path}")
        if models_dir.exists():
            model_count = len(list(models_dir.glob("*")))
            print(f"  Model weights saved to: {models_dir} ({model_count} layer models)")
        print()
        
        # Print summary
        print("=" * 70)
        print("RESULTS SUMMARY")
        print("=" * 70)
        print(f"Model: {args.model}")
        print(f"Filter accuracy: {filter_stats['accuracy']:.1%}")
        print(f"Best layer: {result['metadata']['best_layer']}")
        print(f"Best test score (IIA accuracy): {result['metadata']['best_test_score']:.3f}")
        print(f"Average test score: {result['metadata']['avg_test_score']:.3f}")
        print()
        print(f"Test scores by layer:")
        for layer in sorted(result["test_scores"].keys()):
            score = result["test_scores"][layer]
            marker = " ★" if layer == result["metadata"]["best_layer"] else ""
            print(f"  Layer {layer:2d}: {score:.3f}{marker}")
        print()
        print(f"All results saved to: {output_dir}")
        print()
        print("Saved files:")
        print(f"  - Summary: {summary_path}")
        print(f"  - Full result: {result_path}")
        if models_dir.exists():
            print(f"  - Model weights: {models_dir}/ (one directory per layer)")
            print(f"  - Feature indices: {das_output_dir}/training/feature_indices.json")
            print(f"  - Heatmaps: {das_output_dir}/heatmaps/")
        print("=" * 70)
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Error during Boundless DAS training: {e}")
        import traceback
        traceback.print_exc()
        
        # Try to load any partial results from DAS output directory
        test_scores_path = das_output_dir / "test_eval" / "scores.json"
        train_scores_path = das_output_dir / "train_eval" / "scores.json"
        
        partial_test_scores = {}
        partial_train_scores = {}
        
        if test_scores_path.exists():
            try:
                with open(test_scores_path, "r") as f:
                    scores_dict = json.load(f)
                # Convert string keys back to layer numbers
                # Keys are like "(0,)" or "0" depending on format
                for key_str, score in scores_dict.items():
                    try:
                        # Try to parse as tuple string like "(0,)"
                        if key_str.startswith("(") and key_str.endswith(")"):
                            layer = int(key_str.strip("()").split(",")[0])
                        else:
                            layer = int(key_str)
                        partial_test_scores[layer] = score
                    except (ValueError, IndexError):
                        pass
                print(f"  Loaded partial test scores from {test_scores_path}")
            except Exception as load_error:
                print(f"  Warning: Could not load test scores: {load_error}")
        
        if train_scores_path.exists():
            try:
                with open(train_scores_path, "r") as f:
                    scores_dict = json.load(f)
                # Convert string keys back to layer numbers
                for key_str, score in scores_dict.items():
                    try:
                        if key_str.startswith("(") and key_str.endswith(")"):
                            layer = int(key_str.strip("()").split(",")[0])
                        else:
                            layer = int(key_str)
                        partial_train_scores[layer] = score
                    except (ValueError, IndexError):
                        pass
                print(f"  Loaded partial train scores from {train_scores_path}")
            except Exception as load_error:
                print(f"  Warning: Could not load train scores: {load_error}")
        
        # Save error state to partial summary, including any partial scores
        partial_summary.update({
            "error": str(e),
            "error_traceback": traceback.format_exc(),
            "step": "das_training_failed",
            "partial_test_scores_by_layer": partial_test_scores if partial_test_scores else None,
            "partial_train_scores_by_layer": partial_train_scores if partial_train_scores else None,
        })
        with open(partial_summary_path, "w") as f:
            json.dump(partial_summary, f, indent=2)
        print(f"\n  Partial results saved to: {output_dir}")
        print(f"  Check partial_summary.json for progress details")
        if partial_test_scores:
            print(f"  Found IIA scores for {len(partial_test_scores)} layer(s): {sorted(partial_test_scores.keys())}")
        return 1


if __name__ == "__main__":
    sys.exit(main())

