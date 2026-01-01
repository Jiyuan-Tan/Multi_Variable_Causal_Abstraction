"""
Counterfactual dataset generation for entity binding tasks.

This module provides functions to generate counterfactual examples by
swapping entity groups while keeping the query the same.
"""

import random
from typing import Dict, Any
from .causal_models import create_positional_entity_causal_model, sample_valid_entity_binding_input
from .config import EntityBindingTaskConfig


def _set_query_entities_from_group(
    sample: Dict[str, Any],
    group_idx: int,
    config: EntityBindingTaskConfig,
) -> None:
    """Set query_e{e} variables from the entities of the specified group."""
    for e in range(config.max_entities_per_group):
        sample[f"query_e{e}"] = sample.get(f"entity_g{group_idx}_e{e}")


def swap_query_group(
    config: EntityBindingTaskConfig, change_answer: bool = False
) -> Dict[str, Any]:
    """
    Generate a counterfactual by swapping the queried entity group with another group.

    This tests whether the model correctly retrieves information based on which
    entity group is queried, rather than relying on positional information.

    Example with 3 groups:
        Input:
            Entities: g0=(Pete, jam), g1=(Ann, pie), g2=(Bob, cake)
            Query: group 1, entity 0 -> asking about Ann
            Prompt:  "Pete loves jam, Ann loves pie, and Bob loves cake. What does Ann love?"
            Answer:  "pie"

        Counterfactual (swapped g1 with g2):
            Entities: g0=(Pete, jam), g1=(Bob, cake), g2=(Ann, pie)
            Query: group 1, entity 0 -> now asking about Bob (who moved to g1)
            Prompt:  "Pete loves jam, Bob loves cake, and Ann loves pie. What does Bob love?"
            Answer:  "cake"

    The counterfactual swaps the entity groups but keeps the SAME QUERY POSITION.
    This means:
    - We're querying the same POSITION in the binding matrix (e.g., group 1, entity 0)
    - But different ENTITIES now occupy that position
    - The model must retrieve the binding at that position, not memorize entity names

    If config.fixed_query_indices is set, query_indices will be fixed to that value.

    Parameters
    ----------
    config : EntityBindingTaskConfig
        The task configuration
    change_answer : bool, optional
        If True, replace the answer entity in the counterfactual with a new entity
        from the same pool (different from all entities in the sample). This creates
        a counterfactual with a different expected answer. Default is False.

    Returns
    -------
    Dict[str, Any]
        Dictionary with:
        - "input": The original input sample
        - "counterfactual_inputs": List containing one counterfactual sample
    """
    # Create causal model
    model = create_positional_entity_causal_model(config)

    # Sample a valid input (will use config.fixed_query_indices if set)
    input_sample = sample_valid_entity_binding_input(config)

    # Regenerate raw_input for the input sample
    model.new_raw_input(input_sample)

    # Get query_group directly from the input sample
    query_group = input_sample["query_group"]
    active_groups = input_sample["active_groups"]

    # Choose a different group to swap with
    other_groups = [g for g in range(active_groups) if g != query_group]
    if not other_groups:
        # Only one group active, return random counterfactual instead
        counterfactual_input = sample_valid_entity_binding_input(config)
        model.new_raw_input(counterfactual_input)
        return {"input": input_sample, "counterfactual_inputs": [counterfactual_input]}

    swap_group = random.choice(other_groups)

    # Create counterfactual by swapping the entity groups
    counterfactual = input_sample.copy()

    # Swap entities between query_group and swap_group
    entities_per_group = input_sample["entities_per_group"]
    for e in range(entities_per_group):
        key_query = f"entity_g{query_group}_e{e}"
        key_swap = f"entity_g{swap_group}_e{e}"

        # Swap the entities
        counterfactual[key_query] = input_sample[key_swap]
        counterfactual[key_swap] = input_sample[key_query]

    # KEY: Update query_group and query_e{e} to follow where the original query entity moved
    # After the swap, the original query entities are now at swap_group
    counterfactual["query_group"] = swap_group
    _set_query_entities_from_group(counterfactual, swap_group, config)

    # This keeps the SAME QUESTION (asking about the same entity)
    # but the statement has changed (entities in different positions)

    # Optionally change the answer entity to a new one
    if change_answer:
        answer_index = counterfactual["answer_index"]
        # The answer entity is at position answer_index in the queried group
        # After swap, the queried group is swap_group
        answer_key = f"entity_g{swap_group}_e{answer_index}"

        # Collect all entities currently in the sample
        used_entities = set()
        for g in range(counterfactual["active_groups"]):
            for e in range(counterfactual["entities_per_group"]):
                entity = counterfactual.get(f"entity_g{g}_e{e}")
                if entity:
                    used_entities.add(entity)

        # Get available entities from the same pool (same entity role)
        available = [
            ent for ent in config.entity_pools[answer_index]
            if ent not in used_entities
        ]

        if available:
            new_answer = random.choice(available)
            counterfactual[answer_key] = new_answer
            # Also update query_e{answer_index} if it was the answer position
            counterfactual[f"query_e{answer_index}"] = new_answer

    # Remove raw_input if copied
    if "raw_input" in counterfactual:
        del counterfactual["raw_input"]

    model.new_raw_input(counterfactual)

    return {"input": input_sample, "counterfactual_inputs": [counterfactual]}

def random_counterfactual(config: EntityBindingTaskConfig) -> Dict[str, Any]:
    """
    Generate a completely random counterfactual by sampling two independent inputs.

    This is a baseline condition - the counterfactual is unrelated to the input.

    Parameters
    ----------
    config : EntityBindingTaskConfig
        The task configuration

    Returns
    -------
    Dict[str, Any]
        Dictionary with:
        - "input": The original input sample
        - "counterfactual_inputs": List containing one counterfactual sample
    """
    model = create_positional_entity_causal_model(config)

    # Sample two independent inputs
    input_sample = sample_valid_entity_binding_input(config)
    model.new_raw_input(input_sample)

    counterfactual = sample_valid_entity_binding_input(config)
    model.new_raw_input(counterfactual)

    return {"input": input_sample, "counterfactual_inputs": [counterfactual]}
