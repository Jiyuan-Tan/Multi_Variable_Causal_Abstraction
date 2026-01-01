"""
Counterfactual generator functions for the task.
"""

from .causal_models import causal_model


def is_valid(setting) -> bool:
    """Check that the input is semantically valid."""
    raise NotImplementedError("TODO: implement")


def sample_valid_input():
    """Sample a valid input"""
    input_sample = causal_model.sample_input(filter_func=is_valid)
    return input_sample


def random_counterfactual():
    """
    Generate a completely random counterfactual by sampling two independent inputs.
    """
    input_sample = sample_valid_input()
    counterfactual = sample_valid_input()

    return {"input": input_sample, "counterfactual_inputs": [counterfactual]}


# TODO: implement generation functions for more counterfactual datasets


COUNTERFACTUAL_GENERATORS = {
    "random_counterfactual": random_counterfactual,
}
