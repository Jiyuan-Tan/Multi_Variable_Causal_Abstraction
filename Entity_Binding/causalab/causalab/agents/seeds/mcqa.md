# Multiple Choice Question Answering (MCQA)

This task examines how models answer simple multiple choice questions about object colors, e.g., "The ball is white. What color is the ball?\nH. orange\nX. white\nAnswer:"

## Behavioral Causal Model

The causal model should have input variables for the object, the target color, the letters corresponding to each choice, and the candidate answers. It should have an intermediate variable for the answer position. The output variable should be the letter symbol corresponding to the correct answer.

## Input Samplers

Random input samplers that ensure the question is answerable (the correct color appears in the choices) and all symbols are distinct.

## Counterfactual Datasets

Three counterfactual datasets:

1. **different_symbol**: Same question, choices, and positions, but different letter symbols
2. **same_symbol_different_position**: Same symbols and choices, but swapped positions
3. **random_counterfactual**: Completely random pairs

## Language Model

```yaml
models:
  - meta-llama/Llama-3.2-1B-Instruct
```

# ⚠️ STOP HERE FOR STEP 1 ⚠️

## Token Positions

Consider token positions for each answer symbol, the correct answer symbol, and the last token.

## Experiments

Run full vector residual stream patching using all the counterfactual datasets. Get results for the answer position variable and the output variable. Train attention head masks to localize the output variable. You should decide which dataset(s) to train masks on by viewing the results from the residual stream patching.

## Output File

- What's the last layer before representation of the answer starts to depart from its original position (ie, the last layer with a nearly maximum signal for the input variables)?
- At which layer does it fully arrive at the final destimation (ie, the first layer with nearly maximum signal for the output variable)?
- Which attention heads move it?

```json
{
    "departure": $LAYER,
    "arrival": $LAYER,
    "heads": [
        [$LAYER_0, $HEAD_IDX_0],
        [$LAYER_1, $HEAD_IDX_1],
        ...
    ]
}
```
