"""
Utility functions for entity binding experiments.

Provides helper functions for task configuration, causal models, and evaluation.
"""

from typing import Any, Dict

from tasks.entity_binding.config import (
    EntityBindingTaskConfig,
    create_sample_action_config,
    create_sample_love_config,
)
from tasks.entity_binding.causal_models import create_positional_entity_causal_model


def get_task_config(config_name: str) -> EntityBindingTaskConfig:
    """Get task configuration by name."""
    if config_name == "love":
        config = create_sample_love_config()
        config.max_groups = 2
    elif config_name == "action":
        config = create_sample_action_config()
        config.max_groups = 3
    else:
        raise ValueError(
            f"Unknown config name: {config_name}. Use 'love' or 'action'"
        )

    # Add instruction wrapper for better performance
    config.prompt_prefix = "We will ask a question about the following sentences.\n\n"
    config.statement_question_separator = "\n\n"
    config.prompt_suffix = "\nAnswer:"

    return config


def get_causal_model(config: EntityBindingTaskConfig, model_type: str = "positional_entity"):
    """
    Get causal model for the given configuration.

    Args:
        config: Task configuration
        model_type: Model type (currently only 'positional_entity' is supported)

    Returns:
        CausalModel object
    """
    if model_type == "positional_entity":
        return create_positional_entity_causal_model(config)
    else:
        raise ValueError(
            f"Unknown model type: {model_type}. Use 'positional_entity'"
        )


def get_pipeline_config(config_name: str) -> Dict[str, Any]:
    """Get pipeline configuration for a given config."""
    return {"max_new_tokens": 5, "max_length": 256}


def get_checker():
    """Get checker function for verifying model outputs."""

    def checker(neural_output, causal_output):
        return (
            causal_output in neural_output["string"]
            or neural_output["string"].strip() in causal_output
        )

    return checker
