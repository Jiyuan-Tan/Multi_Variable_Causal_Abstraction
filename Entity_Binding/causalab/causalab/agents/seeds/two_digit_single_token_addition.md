
# Two-digit Addition

This task will be solving two digit addition problems, i.e., "The sum of 12 and 19 is 31"

## Behavioral Causal Model

The behavioral causal model should have two input variables, one for each of the input digits. The raw input should have the form "The sum of x and y is ".

The output variable should store the sum of the two inputs.

## Input Samplers
Random input samplers where all of the inputs are between 0 and 20. 

### Causal Model Hypotheses with Intermediate Structure
No other causal models are necessary.

## Counterfactuals
One counterfactual dataset of random counterfactuals.

## Language Model
```yaml
models:
  - allenai/OLMo-2-0425-1B
```

# ⚠️ STOP HERE FOR STEP 1 ⚠️

## Token Positions
Consider three token positions, each of the two input numbers and the last token in the input.

## Experiments
Run full vector residual stream patching and get the results for each of the two input variables and the output variable.

Train attention head masks to localize the output variable. 

## Output File

- What's the last layer before representation of the answer starts to depart from its original position (ie, the last layer with a nearly maximum signal for the input variables)?
- At which layer does it fully arrive at the final destimation (ie, the first layer with nearly maximum signal for the output variable)?
- Which attention heads move it?

{
    "departure": $LAYER_A,
    "arrival": $LAYER_B,
    "heads": [
        [$LAYER_X, $HEAD_IDX_W],
        [$LAYER_Y, $HEAD_IDX_V],
        ...
    ]
}
