For context, please review the prompt for the prior step: STEP_1.md. Similarly, review any code that the prior Claude made for that step. This code will be the uncommitted changes show by git status.

Like Step 1, you will be filling out template files, based on a seed at {{PATH_TO_SEED}}. This time, there are multiple scripts to run in order to complete the task; each remaining script in the agents/coder/outputs/experiments folder should be run. Like Step 1, you should use the files in agents/coder/outputs/ as templates, and you can use the script at agents/coder/outputs/tests/test_with_model.py for feedback. Feel free to revise your prior work as you complete this phase, as you may find that it is incorrect or needs adaptation.

When your instructions differ from what is in the template, default to the instructions in provided in the seed document.

# Token Positions

## What are Token Positions?

Token positions tell the intervention framework **where** in the input sequence to perform interventions. When you patch a residual stream or attention head, you're modifying activations at specific token positions.

**Examples:**
- Patching at the "last token position" modifies the final prediction
- Patching where variable `x` appears tests if that position carries information about x
- Patching tokens before/after a variable tests information flow

**Why this matters:** By systematically intervening at different positions and layers, we can map out where causal variables are encoded, transformed, and decoded in the network.

## Declarative Specification System (PREFERRED)

We provide a declarative system that handles most common cases without writing Python code. This eliminates the complexity of manual substring matching, tokenization handling, and padding logic.

**Supported patterns:**

1. **Fixed positions**: First, last, or nth token in the sequence
2. **Variable positions**: Where a template variable appears
3. **Indexed positions**: Nth token within a variable's tokenization
4. **Relative positions**: Tokens before/after a variable

## Fallback: Python Functions

If the declarative system can't express your token position (rare!), use the legacy substring token positions functions in neural/LM_units.py.

Be careful using get_substring_token_ids when there might be multiple occurrences of the same substring. **The best approach is to use a substring with enough context to be unique.** For example, in "The sum of 5 and 5 is", use "of 5 and" for the first 5 and "and 5 is" for the second 5, rather than just "5". You can use `strict=True` to catch ambiguity errors during development, and the `occurrence` parameter exists for rare cases where you know the occurrence index is stable, but designing around uniqueness is more robust.

# Language Model experiments

First, I want you to read neural/pipeline.py for details on how to use the language models, and look at experiments/intervention_experiment.py for details on the experiments. These will be essential and is required.

You should also verify that the models you are testing are capable of solving the task, meaning that they pass the filtering stage with reasonably high accuracy, i.e., at least passing a random baseline. If this doesn't work, there is a problem, because we have already verified that the model is capable of solving the task with reasonably high accuracy.

You will be asked to conduct multiple experiments. Make a plan and make sure you run them all and save all the results. You can use text printout of the results to see the information yourself so you can summarize it back in a RESULTS.md. Everytime you do a text printout, you should also save a heatmap. **DO NOT try to read out the results dictionary directly, ALWAYS use text printouts and heatmaps from premade functions to analyze the results**

**When debugging your experiment scripts, you can run the experiment on less data or less layers to more quickly iterate before a full run**

Run the experiments and consolidate your results.

**Make a plan first, then implement that plan. Wait for all of the experiments to complete, you are not finished with your job until the results are saved.**

# Managing Long-Running Experiments

Some experiments can take a long time to complete (30 minutes to several hours). Run experiments in the foreground with a generous timeout (up to 4 hours = 14,400,000 milliseconds):

```python
Bash("uv run my_long_experiment.py", timeout=14400000)
```

Put the result object in agents/coder/outputs/output.json.