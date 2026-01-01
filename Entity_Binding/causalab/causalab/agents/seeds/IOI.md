# Indirect-Object Identification

This task examines how models learn to select the proper indirect object in contexts where a repeated entity causes another entity to be more likely. For example:

"After the lunch, Susy and Karen went to the movies. Karen gave popcorn to"

Karen appearing in the subject position makes it more likely for Susy to appear in the indirect object position of the sentence.

## Behavioral Causal Model

The IOI causal model generates prompts with the following structure:
- Three names appear: name_A and name_B are introduced first, then name_C (which matches one of them) appears in a subject position before "gave a/an [object] to"
- The model should predict the other name (not name_C) as the indirect object

Generate a list of templates and a list of filler names to populate the templates.

- **raw_input**: The filled template string (the actual prompt)

- **template**: A string template with placeholders for names, place, and object
- **name_A**: First name (from names list)
- **name_B**: Second name (from names list)
- **name_C**: Third name that matches either name_A or name_B

- **logits**: Dictionary mapping each name to its logit value

- **raw_output**: Dictionary containing the predicted token string and logits


## Input Samplers

The input sampler must ensure that:
- name_A and name_B are distinct (name_A ≠ name_B)
- name_C matches either name_A or name_B to create well-formed IOI examples

## Counterfactual Datasets

Generate the **random_ABC** counterfactual dataset where:
- Original inputs are well-formed (name_C matches name_A or name_B)
- Counterfactual inputs have three completely distinct names (name_A, name_B, name_C all different)
- This creates ambiguous scenarios with no definitive answer

## Language Model

```yaml
models:
  - openai-community/gpt2
```

# ⚠️ STOP HERE FOR STEP 1 ⚠️

## Experiments

Run full vector patching on the output of attention heads at the last token position.

For each intervention, compute a custom scoring metric that calculates:
1. The logit difference for the correct name minus the incorrect name in the intervention output
2. The logit difference for the correct name minus the incorrect name in the baseline (no intervention) output
3. Return the difference between these two logit differences, i.e., the intervention value minus the actual value 

This metric measures how much each attention head intervention changes the model's confidence in predicting the correct indirect object.

Filter the dataset to include only examples where the model predicts correctly on the baseline input.

Generate a heatmap showing the scores for each attention head (layer × head) across the dataset.

## Output File

- Which two attention heads had the top three positive scores in the heatmap?

```json
{
    "top_two_heads": [
        [$LAYER_0, $HEAD_0],
        [$LAYER_1, $HEAD_1]
    ]
}
```
