
# Hierarchical Equality

This will be an in context learning task. The function demonstrated in context has two pairs of inputs (four inputs total) and outputs 1 if both pairs are equal or both pairs are unequal, and outputs 0 if exactly one pair is equal. This would mean that AABB and ABCD are both assigned 1 while ABCC and AABC are both assigned 0. In fact, AABB, ABCD, ABCC, and AABC are the four patterns that define the entire input space, i.e., systematically replacing A, B, C, and D with other letters can generate any input example.

## Behavioral Causal Model

Use the following code template:

"def double_equality(a, b, c, d):\n    x = (a == b)\n    y = (c == d)\n    return x == y\nThe function call double_equality({VAR1}, {VAR2}, {VAR3}, {VAR4}) returns the value "

Use **60 in-context examples** for all experiments.

You should write out several different templates and have a causal model variable that allows you to switch between them.

The four variables will populate the final example and the output variable will be 1 or 0 (not True or False) depending on the pattern provided.

### Example Input with Two In-Context Learning Examples

```
def double_equality(a, b, c, d):
    x = (a == b)
    y = (c == d)
    return x == y

The function call double_equality("A", "A", "B", "B") returns the value 1
The function call double_equality("C", "D", "E", "E") returns the value 0
The function call double_equality("F", "G", "H", "I") returns the value
```

In this example:
- First ICL example: ("A", "A", "B", "B") → 1 (both pairs equal, AABB pattern)
- Second ICL example: ("C", "D", "E", "E") → 0 (first pair unequal, second equal, ABCC pattern)
- Test query: ("F", "G", "H", "I") → model should output 1 (both pairs unequal, ABCD pattern) 

## Input Samplers

Random input samplers that is balanced among the four classes defined above. 

### Causal Model Hypotheses with Intermediate Structure

There should be two intermediate variables. The first computes whether Var1 and Var2 are equal and the second computes whether Var3 and Var4 are equal. The output will be computed by checking whether these two intermediate variables are equal. 

## Counterfactuals
One counterfactual dataset of random counterfactuals using the balanced input sampler.

## Language Model
```yaml
models:
  - meta-llama/Llama-3.1-8B-Instruct
```

# ⚠️ STOP HERE FOR STEP 1 ⚠️

## Token Positions

I want token positions for the last token and each of the four input tokens.

## Experiments
Use a batch size of 4 and a dataset size of 32

Generate datasets and evaluate the Llama model using the filter_experiment to ensure the model can reliably solve the task.

Run full vector residual stream patching, then compute the heatmaps for each of the four input variables and the two intermediate variables.

Run attention head localization for the two intermediate variables.

## Output File

- What is the first and last layer with a >75% signal for the two equality variables?
- What is the first and last layer with a >90% signal for the four input variables?
- What is the first layer with a >90% signal for the output? 
- Which attention heads were selected by the masks you learned for the left equality variable?
- Which attention heads were selected by the masks you learned for the right equality variable?


{
    "input": [$FIRST_INPUT, $LAST_INPUT],
    "equality": [$FIRST_EQUALITY, $LAST_EQUALITY],
    "output": $FIRST_OUTPUT,
    "left_equality_heads": [
        [$LAYER_0, $HEAD_IDX_0],
        [$LAYER_1, $HEAD_IDX_1],
        ...
    ],
    "right_equality_heads": [
        [$LAYER_0, $HEAD_IDX_0],
        [$LAYER_1, $HEAD_IDX_1],
        ...
    ]
}

