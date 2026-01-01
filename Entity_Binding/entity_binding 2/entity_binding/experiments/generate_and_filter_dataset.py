<<<<<<< HEAD
#!/usr/bin/env python
"""
Generate and filter counterfactual datasets for entity binding tasks.

This script:
1. Generates counterfactual datasets using entity binding counterfactual generators
2. Loads a language model
3. Filters to keep only examples where model succeeds on both original and counterfactual
4. Saves both original and filtered datasets with metadata

Usage:
    python 01_generate_and_filter_dataset.py --model MODEL --config CONFIG [--cf-type CF_TYPE]
    python 01_generate_and_filter_dataset.py --model meta-llama/Meta-Llama-3.1-8B-Instruct --config love
    python 01_generate_and_filter_dataset.py --test  # Run in test mode (small dataset)
"""

import argparse
import json
import os
from pathlib import Path

import torch

from causal.counterfactual_dataset import CounterfactualDataset
from experiments.filter import filter_dataset
from experiments.io import create_metadata, save_datasets, save_metadata
from neural.pipeline import LMPipeline
from tasks.entity_binding.counterfactuals import (
    swap_query_group,
    random_counterfactual,
)
from tasks.entity_binding.experiments.utils import (
    get_task_config,
    get_causal_model,
    get_pipeline_config,
    get_checker,
)

# ============================================================================
# Configuration
# ============================================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Map counterfactual type names to generator functions
COUNTERFACTUAL_GENERATORS = {
    "swap_query_group": swap_query_group,
    "random": random_counterfactual,
}
=======
#!/usr/bin/env -S uv run python
"""
Universal Dataset Generation and Filtering Script for Entity Binding

This script generates counterfactual datasets and filters them based on model performance.
Saves both the original (unfiltered) and filtered datasets.

Usage:
    python generate_and_filter_dataset.py --config CONFIG --model MODEL [options]
    python generate_and_filter_dataset.py --config love --model meta-llama/Llama-3.1-8B-Instruct
    python generate_and_filter_dataset.py --config action --test  # Test mode
"""

import sys

sys.path.append("/mnt/polished-lake/home/atticus/CausalAbstraction")

import argparse
import torch
from pathlib import Path
import json

from causalab.tasks.entity_binding.experiment_config import (
    get_task_config,
    get_causal_model,
    get_counterfactual_generator,
    get_checker,
)
from causalab.neural.pipeline import LMPipeline
from causalab.causal.counterfactual_dataset import CounterfactualDataset
from causalab.experiments.filter_experiment import FilterExperiment
>>>>>>> origin/main


def main():
    parser = argparse.ArgumentParser(
<<<<<<< HEAD
        description="Generate and filter counterfactual datasets for entity binding"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="meta-llama/Meta-Llama-3.1-8B-Instruct",
        help="Model name or path (default: meta-llama/Meta-Llama-3.1-8B-Instruct)",
=======
        description="Generate and filter entity binding dataset"
>>>>>>> origin/main
    )
    parser.add_argument(
        "--config",
        type=str,
<<<<<<< HEAD
        default="love",
        choices=["love", "action"],
        help="Task configuration name (default: love)",
    )
    parser.add_argument(
        "--cf-type",
        type=str,
        default="swap_query_group",
        choices=list(COUNTERFACTUAL_GENERATORS.keys()),
        help="Counterfactual generator type (default: swap_query_group)",
=======
        required=True,
        help="Task configuration: love, action, or positional_entity",
    )
    parser.add_argument(
        "--counterfactual",
        type=str,
        default="swap_query_group",
        help="Counterfactual generator name (default: swap_query_group)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="meta-llama/Llama-3.1-8B-Instruct",
        help="Model to use (default: meta-llama/Llama-3.1-8B-Instruct)",
>>>>>>> origin/main
    )
    parser.add_argument(
        "--size",
        type=int,
<<<<<<< HEAD
        default=256,
        help="Number of examples to generate (default: 256)",
=======
        default=128,
        help="Number of counterfactual pairs to generate (default: 128)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory (default: auto-generated from config)",
>>>>>>> origin/main
    )
    parser.add_argument(
        "--batch-size",
        type=int,
<<<<<<< HEAD
        default=None,
        help="Batch size for filtering (default: same as size)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for datasets (default: auto-generated)",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Test mode (8 examples, small batch)",
    )
    parser.add_argument(
        "--change-answer",
        action="store_true",
        help="For swap_query_group: change the answer entity in counterfactuals",
    )
    parser.add_argument(
        "--query-indices",
        type=str,
        default=None,
        help="Fix query indices (comma-separated, e.g., '0' or '0,1'). Default: random",
    )
    parser.add_argument(
        "--answer-index",
        type=int,
        default=None,
        help="Fix answer index (e.g., 0 or 1). Default: random",
    )
    args = parser.parse_args()

    # Test mode settings
    if args.test:
        args.size = 8
        args.batch_size = 8
        print("\n*** TEST MODE: Using size=8, batch_size=8 ***\n")

    if args.batch_size is None:
        args.batch_size = args.size

    # Auto-generate output path if not specified
    if args.output_dir is None:
        model_short = args.model.split("/")[-1].replace("-", "_").lower()
        change_answer_suffix = "_change_answer" if args.change_answer else ""
        args.output_dir = (
            f"tasks/entity_binding/datasets/{args.cf_type}{change_answer_suffix}_{model_short}_{args.config}"
        )

    # ========================================================================
    # Configuration
    # ========================================================================
    print("=" * 80)
    print("ENTITY BINDING: GENERATE AND FILTER COUNTERFACTUAL DATASETS")
    print("=" * 80)
    # Parse query_indices if provided
    parsed_query_indices = None
    if args.query_indices is not None:
        parsed_query_indices = tuple(int(x.strip()) for x in args.query_indices.split(","))

    print("\nConfiguration:")
    print(f"  Model: {args.model}")
    print(f"  Task config: {args.config}")
    print(f"  Counterfactual type: {args.cf_type}")
    print(f"  Change answer: {args.change_answer}")
    print(f"  Query indices: {parsed_query_indices}")
    print(f"  Answer index: {args.answer_index}")
    print(f"  Dataset size: {args.size}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Output directory: {args.output_dir}")
    print()

    # ========================================================================
    # Create Task Configuration
    # ========================================================================
    print("=" * 80)
    print("CREATING TASK CONFIGURATION")
    print("=" * 80)

    task_config = get_task_config(args.config)

    # Apply fixed query_indices and answer_index if provided
    if parsed_query_indices is not None:
        task_config.fixed_query_indices = parsed_query_indices
    if args.answer_index is not None:
        task_config.fixed_answer_index = args.answer_index

    print(f"\nTask configuration:")
    print(f"  Max groups: {task_config.max_groups}")
    print(f"  Max entities per group: {task_config.max_entities_per_group}")
    print(f"  Entity roles: {task_config.entity_roles}")
    print(f"  Statement template: {task_config.statement_template}")
    print(f"  Fixed query indices: {task_config.fixed_query_indices}")
    print(f"  Fixed answer index: {task_config.fixed_answer_index}")
    print()

    # ========================================================================
    # Create Causal Model
    # ========================================================================
    print("=" * 80)
    print("CREATING CAUSAL MODEL")
    print("=" * 80)

    causal_model = get_causal_model(task_config)
    print(f"\nCausal model: {causal_model.id}")
    print()

    # ========================================================================
    # Load Language Model
    # ========================================================================
    print("=" * 80)
    print("LOADING LANGUAGE MODEL")
    print("=" * 80)

    pipeline_config = get_pipeline_config(args.config)
    pipeline = LMPipeline(
        args.model,
        max_new_tokens=pipeline_config["max_new_tokens"],
        device=DEVICE,
        dtype=torch.bfloat16 if DEVICE == "cuda" else torch.float32,
        max_length=pipeline_config["max_length"],
    )
    pipeline.tokenizer.padding_side = "left"

    print(f"\nModel loaded: {args.model}")
    print(f"  Device: {pipeline.model.device}")
    print(f"  Layers: {pipeline.model.config.num_hidden_layers}")
    print()

    # ========================================================================
    # Generate Counterfactual Dataset
    # ========================================================================
    print("=" * 80)
    print("GENERATING COUNTERFACTUAL DATASET")
    print("=" * 80)

    # Get the counterfactual generator function
    cf_generator_fn = COUNTERFACTUAL_GENERATORS[args.cf_type]

    # Create generator that uses the task config
    def generator():
        # Pass change_answer flag for swap_query_group
        if args.cf_type == "swap_query_group" and args.change_answer:
            return cf_generator_fn(task_config, change_answer=True)
        return cf_generator_fn(task_config)

    print(f"\nGenerating {args.size} counterfactual pairs using {args.cf_type}...")
    dataset = CounterfactualDataset.from_sampler(args.size, generator)
    print(f"Generated {len(dataset)} pairs")

    # Show example
    if len(dataset) > 0:
        print("\nExample pair:")
        example = dataset[0]
        example_input = example["input"]
        example_cf = example["counterfactual_inputs"][0]

        # Get raw_input from the example (should be set by generator)
        if "raw_input" in example_input:
            print(f"  Input:  {example_input['raw_input']}")
        else:
            # Run forward to get raw_input
            setting = causal_model.run_forward(example_input)
            print(f"  Input:  {setting['raw_input']}")

        # Get expected output
        setting = causal_model.run_forward(example_input)
        print(f"  Expected output: {setting['raw_output']}")

        if "raw_input" in example_cf:
            print(f"  Counterfactual: {example_cf['raw_input']}")
        else:
            cf_setting = causal_model.run_forward(example_cf)
            print(f"  Counterfactual: {cf_setting['raw_input']}")

        cf_setting = causal_model.run_forward(example_cf)
        print(f"  CF expected output: {cf_setting['raw_output']}")
    print()

    # ========================================================================
    # Filter Dataset
    # ========================================================================
    print("=" * 80)
    print("FILTERING DATASET")
    print("=" * 80)

    print("\nFiltering to examples where model answers both original and counterfactual correctly...")

    checker = get_checker()
    filtered_dataset = filter_dataset(
        dataset,
        pipeline,
        causal_model,
        checker,
        batch_size=args.batch_size,
        verbose=True,
        validate_counterfactuals=True,
    )

    print("\nFiltering results:")
    print(f"  Original: {len(dataset)} examples")
    print(f"  Filtered: {len(filtered_dataset)} examples")
    keep_rate = len(filtered_dataset) / len(dataset) * 100 if len(dataset) > 0 else 0
    print(f"  Keep rate: {keep_rate:.1f}%")

    if keep_rate < 50 and not args.test:
        print(
            f"\nWARNING: Keep rate is low ({keep_rate:.1f}%). "
            "This may indicate the model struggles with this task."
        )
    print()

    # ========================================================================
    # Save Datasets
    # ========================================================================
    print("=" * 80)
    print("SAVING DATASETS")
    print("=" * 80)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Prepare datasets dict for saving
    counterfactual_datasets = {args.cf_type: dataset}
    filtered_datasets = {args.cf_type: filtered_dataset}

    # Save datasets
    dataset_dirs = save_datasets(
        counterfactual_datasets,
        filtered_datasets,
        str(output_dir),
        filtering_enabled=True,
        verbose=True,
    )

    print(f"\nSaved original dataset: {dataset_dirs[args.cf_type]['original_dir']}")
    print(f"Saved filtered dataset: {dataset_dirs[args.cf_type]['filtered_dir']}")

    # Create and save metadata
    metadata = create_metadata(
        counterfactual_datasets,
        filtered_datasets,
        args.size,
        args.batch_size,
        model_name=args.model,
        filtering_enabled=True,
    )

    # Add task-specific metadata
    metadata["task"] = "entity_binding"
    metadata["config_name"] = args.config
    metadata["counterfactual_type"] = args.cf_type
    metadata["change_answer"] = args.change_answer
    metadata["fixed_query_indices"] = list(task_config.fixed_query_indices) if task_config.fixed_query_indices else None
    metadata["fixed_answer_index"] = task_config.fixed_answer_index
    metadata["task_config"] = {
        "max_groups": task_config.max_groups,
        "max_entities_per_group": task_config.max_entities_per_group,
        "entity_roles": task_config.entity_roles,
        "statement_template": task_config.statement_template,
    }

    metadata_path = save_metadata(metadata, str(output_dir), verbose=True)
    print(f"Saved metadata: {metadata_path}")
    print()

    # ========================================================================
    # Summary
    # ========================================================================
    print("=" * 80)
    print("COMPLETE")
    print("=" * 80)
    print(f"\nOutput directory: {args.output_dir}")
    print(f"  - Original dataset: {len(dataset)} examples")
    print(f"  - Filtered dataset: {len(filtered_dataset)} examples")
    print(f"  - Keep rate: {keep_rate:.1f}%")
    print(
        "\nNext step: Run 02_run_interventions.py to perform interventions on filtered datasets"
    )
=======
        default=32,
        help="Batch size for filtering (default: 32)",
    )
    parser.add_argument(
        "--test", action="store_true", help="Test mode: size=8, batch_size=8"
    )

    args = parser.parse_args()

    # Test mode overrides
    if args.test:
        args.size = 8
        args.batch_size = 8
        print("\n*** TEST MODE: size=8, batch_size=8 ***\n")

    # Auto-generate output path if not specified
    if args.output is None:
        test_suffix = "_test" if args.test else ""
        args.output = f"tasks/entity_binding/datasets/{args.config}_{args.counterfactual}{test_suffix}"

    # Configuration
    print("=" * 70)
    print("Entity Binding Dataset Generation and Filtering")
    print("=" * 70)
    print("\nConfiguration:")
    print(f"  Task config: {args.config}")
    print(f"  Counterfactual: {args.counterfactual}")
    print(f"  Model: {args.model}")
    print(f"  Dataset size: {args.size}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Output: {args.output}")
    print(f"  Test mode: {args.test}")
    print()

    # Get task configuration
    try:
        config = get_task_config(args.config)
        print("Task configuration loaded:")
        print(f"  Max groups: {config.max_groups}")
        print(f"  Entities per group: {config.max_entities_per_group}")
        print(f"  Entity roles: {config.entity_roles}")
        print(f"  Template: {config.statement_template}")
        print()
    except ValueError as e:
        print(f"Error: {e}")
        return 1

    # Create causal model
    causal_model = get_causal_model(config)
    print(f"Causal model: {causal_model.id}")
    print()

    # Get counterfactual generator
    try:
        cf_generator = get_counterfactual_generator(args.counterfactual, config)
    except ValueError as e:
        print(f"Error: {e}")
        return 1

    # Generate counterfactual dataset
    print(f"Generating {args.size} counterfactual pairs using {args.counterfactual}...")
    dataset = CounterfactualDataset.from_sampler(
        args.size, cf_generator, id=args.counterfactual
    )
    print(f"✓ Generated {len(dataset)} pairs")
    print()

    # Show example
    print("Example pair:")
    print(f"  Input:  {dataset[0]['input']['raw_input']}")
    print(f"  Counter: {dataset[0]['counterfactual_inputs'][0]['raw_input']}")
    print()

    # Load language model
    print(f"Loading language model: {args.model}...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Device: {device}")

    pipeline = LMPipeline(args.model, max_new_tokens=5, device=device, max_length=256)
    pipeline.tokenizer.padding_side = "left"
    print("✓ Model loaded")
    print()

    # Get checker function
    checker = get_checker()

    # Filter the dataset
    print("Filtering dataset based on model performance...")
    experiment = FilterExperiment(pipeline, causal_model, checker)

    datasets_dict = {args.counterfactual: dataset}
    filtered_datasets = experiment.filter(
        datasets_dict, verbose=True, batch_size=args.batch_size
    )

    filtered_dataset = filtered_datasets[args.counterfactual]
    print()
    print("Filtering results:")
    print(f"  Original: {len(dataset)} examples")
    print(f"  Filtered: {len(filtered_dataset)} examples")
    print(f"  Keep rate: {len(filtered_dataset) / len(dataset) * 100:.1f}%")
    print()

    # Check if we have enough data
    if len(filtered_dataset) == 0:
        print(
            "⚠ WARNING: No examples passed filtering! Model may not be capable of this task."
        )
        return 1
    elif len(filtered_dataset) < args.size * 0.5 and not args.test:
        print(
            f"⚠ WARNING: Only {len(filtered_dataset)}/{args.size} examples passed. "
            f"Consider increasing dataset size or checking model capability."
        )
        print()

    # Save datasets
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save original (unfiltered) dataset
    original_path = output_path / "original_dataset"
    print(f"Saving original dataset to {original_path}...")
    dataset.dataset.save_to_disk(str(original_path))
    print(f"✓ Original dataset saved ({len(dataset)} examples)")

    # Save filtered dataset
    filtered_path = output_path / "filtered_dataset"
    print(f"Saving filtered dataset to {filtered_path}...")
    filtered_dataset.dataset.save_to_disk(str(filtered_path))
    print(f"✓ Filtered dataset saved ({len(filtered_dataset)} examples)")
    print()

    # Save metadata
    metadata = {
        "config_name": args.config,
        "counterfactual_type": args.counterfactual,
        "model": args.model,
        "original_size": len(dataset),
        "filtered_size": len(filtered_dataset),
        "keep_rate": len(filtered_dataset) / len(dataset),
        "task_config": {
            "max_groups": config.max_groups,
            "entities_per_group": config.max_entities_per_group,
            "entity_roles": config.entity_roles,
            "template": config.statement_template,
        },
        "test_mode": args.test,
    }

    metadata_path = output_path / "dataset_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"✓ Metadata saved to {metadata_path}")
    print()

    print("=" * 70)
    print("✓ Dataset generation and filtering complete!")
    print("=" * 70)
    print(f"\nOutputs saved to: {output_path}")
    print(f"  - Original dataset: {original_path} ({len(dataset)} examples)")
    print(f"  - Filtered dataset: {filtered_path} ({len(filtered_dataset)} examples)")
    print(f"  - Metadata: {metadata_path}")
>>>>>>> origin/main

    return 0


if __name__ == "__main__":
<<<<<<< HEAD
    exit(main())
=======
    sys.exit(main())
>>>>>>> origin/main
