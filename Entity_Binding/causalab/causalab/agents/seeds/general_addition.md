# Addition
The most general task will be addition with any number of inputs which each have any number of digits. 

You should design the code base for the generic case where K numbers are being added together and each number had D digits. Use the template "The sum of {num1} and {num2} is ", but design so that a template can be provided in the future.

## Behavioral Causal Model
The first input-output causal model should contain K x D input variables, one for each digit of each number. Each variable takes on a value between 0 and 9. 

There should be D + 1 output variables, each storing a digit of the output.

The causal model should be capable of handling blank inputs, i.e., the model can take in less than k numbers and still function. 

There should be an extensible template that the causal model fills with available inputs, i.e., a function mapping a list of numbers to a string like "The sum of {num1} and {num2} ... and {numj} is " 

The raw_output should concatenate the output variables as strings.

Include multiple options for templates that can be specified when generating the datasets, where a set of templates can be provided.

## Input Samplers

Random input samplers that generate inputs where *k* numbers are being added.

An input sampler that takes in a binary vector B of length D and generates two numbers X and Y such that for some index i if B[i] is 1, then the ith digit of X and the ith digit of Y sum to 9.

### Causal Model Hypotheses with Intermediate Structure
This causal model is only for addiion with two inputs.

Next we want a causal model with three classes of intermediate variables which each have D members (one for each input digit):
- The first carry-the-one C_D variable stores 1 if the ones digit of X and the ones digit of Y sum to 10 or more and 0 otherwise
- Other carry-the-one variables: The variable C_i will store a 1 if the ith digit of X, the ith digit of Y, and the previous carry-the-one variable C_{i-1} sum to 10 or more, and will store a 0 otherwise. 
- The first output O_1 variable is simple the sum of the ones digit of X and the ones digit of Y mod 10
- Other output variables: The variable O_i is the sum of the ith digit of X, the ith digit of Y, and the previous carry-the-one variable C_{i-1} all mod 10

## Counterfactuals
One counterfactual dataset of random counterfactuals.

One counterfactual dataset where the counterfactual example is totally random and original example is sampled such that the tens digits from all of inputs sums to 9. 

## Language Model
```yaml
models:
  - meta-llama/Meta-Llama-3.1-8B-Instruct
  - google/gemma-2-9b
  - allenai/OLMo-2-1124-13B
```

# ⚠️ STOP HERE FOR STEP 1 ⚠️

## Token Positions
I want you to write a token position function that returns the *p*th token of the number in the *q*th input position. Also write token position functions for the delimiter tokens using an off set from the number token postions. 

## Experiments
For each of the three models and you will:
- Generate a random counterfactual datasets using two inputs each with two digits 
- load in a specified model 
- filter the dataset with that model and save the results
- Perform residual patching on the last token position and the tokens for each digit in the input.
- Save the heatmaps and text analysis for each 
  - input variable 
  - every group of input variables that form a single number, i.e., target a list of input variable 
  - every subset of input variables going from left two right
  - the raw output
  - Individual output variables

eote that the OLMo and Llama and gemma model all tokenize numbers differently, so you'll need to figure out how to handle that properly, i.e., define the right number of token positions for each model and dataset.

## Output File

I want you to write a file, output.json that answers the following questions:

Two digit Gemma:

- Which layer do individual digits drop below 90%? 3
- At which layer does the two digits of the first number get the strongest signal above the ones digit? 27
- In the later layers, the signal for the tens digit of the second number peaks again in the residual stream. What layer is that at? 26

Two digit LLama:

- What is the last layer that the signal for the first two digits together is above 90% in the residual stream of the first number token? 15
- What is the last layer that the signal for the second two digits together is above 90% in the residual stream of the second number token? 14

Two digit OLMo:

- What is the last layer that the signal for the first two digits together is above 90% in the residual stream of the first number token? 18
- What is the last layer that the signal for the second two digits together is above 90% in the residual stream of the second number token? 18


It must have the following format where all the values are lists:

```json
{
    "gemma_individual_digits_drop_below_90": [$LAYER],
    "gemma_two_digits_strongest_above_ones": [$LAYER],
    "gemma_tens_digit_second_number_peak": [$LAYER],
    "llama_first_two_digits_above_90_last_layer": [$LAYER],
    "llama_second_two_digits_above_90_last_layer": [$LAYER],
    "olmo_first_two_digits_above_90_last_layer": [$LAYER],
    "olmo_second_two_digits_above_90_last_layer": [$LAYER]
}
```