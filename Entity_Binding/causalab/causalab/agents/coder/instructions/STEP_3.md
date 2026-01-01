For context, please review the prompts for the prior steps: STEP_1.md and STEP_2.md. Similarly, review any code that the prior Claude made for those step(s). It is in the output folder.

The prior steps created a number of scripts. Read them all, and then summarize the results in a demo jupyter notebook. Make it clean, readable, concise, clear, and minimal.

Instead of writing the notebook directly, I first want you to write a script containing everything the notebook will do. That way, you can run the script and test it, to ensure the outputs are as expected. Then, manually rewrite the script as a notebook, and use the cli tools to run it once, populating the outputs. You should then view those outputs to make sure the conversion worked. unlike the prior steps, which output graphs as text objects, you should produce heatmaps in the notebook. good luck! and remember - methodical, scientific, comprehensive, rigorous.

Crucially, this new script and notebook should **load in the results that already exist**, and load in **heatmaps and other plots** not the text print outs describing results. They should not rerun all the experiments, this would take too long. It is fine to load in and display examples, causal models, token positions, counterfactual datasets, and anything not involving a language model. Be sure to find the filtering results and print those out in addition to the various plots from the other experiments.

When executing a notebook from the command line, there are often issues with the working directory of the notebook. You should have a cell at the top of your notebook that explicitly sets the working directory of the notebook.

```
import os
os.chdir("/path/to/working/directory")
```

**Make a plan first and then implement that plan**