First, read the spec file for the causal model at {{PATH_TO_SEED}}.

**Create causal models and counterfactual datasets**
- The files in the agents/coder/outputs folder are templates; fill them out
- Your goal is to successfully run the agents/coder/outputs/experiments/01_generate_and_filter_datasets.py script
- You can use the agents/coder/outputs/tests/test_causal_model.py script to test your implementation
- Look in the causal/causal_utils.py file for useful functions that help with causal model creation, e.g., statement_conjunction_function is helpful for taking a core template and extending it with delimiters.
- You will often be asked to make a causal model that is extensible or general enough to handle a variety of inputs. Make sure you do this! Just because you are given example inputs with a certain number of sentences, facts, entities, numbers, etc., doesn't mean that the causal model should only be able to process that kind of input!

**Make a plan first, then implement that plan.**
