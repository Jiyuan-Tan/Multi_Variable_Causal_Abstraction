# Counting

This task will be counting how many times a target object appears in a list of objects.

For example: "Count how many bananas are in the list, returning only an integer. [lime, apple, banana, banana, orange] How many bananas are in the list? Answer:"

## Behavioral Causal Model

The behavioral causal model should have input variables for the target object and each object in the list (default list length of 5). The model should include intermediate variables for a running count that accumulates the matches thus far. The output variable should be the final count as a string.

## Input Samplers

Random input samplers where objects are sampled from a list of possible objects (e.g., fruits) that all have the same token length.

Ensure that the target object appears at least once in the list so the question is answerable.

### Causal Model Hypotheses with Intermediate Structure

No other causal models are necessary.

## Counterfactuals

Counterfactuals for changing from target to non-target, or vice versa, for each position i in the list. Random counterfactual.

## Language Model

```yaml
models:
  - meta-llama/Meta-Llama-3.1-8B-Instruct
```

# ⚠️ STOP HERE FOR STEP 1 ⚠️

## Token Positions

Consider the token positions for each object in the list, the token after each item in the list, and the last token in the input.

## Experiments

Full vector residual stream patching and attention head DBM.

## Output File

- Via iterative residual stream patching across each of the positional counterfactual datasets, where does the representation for each intermediate variable appear?
- Is there a pattern across all variables?
- If there is a pattern: can you identify the attention heads that carry the information to (presumably) the last token for outputting?

No schema on this one - just answer in prose.