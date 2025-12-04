'''
Shell command to run the script: python das.py --train --hf-cache-dir /vision/u/puyinli/Multi_Variable_Causal_Abstraction/.hf_cache
'''
import os
import sys

# Set HuggingFace cache directory BEFORE importing transformers
# This must happen before any HuggingFace library is imported
for i, arg in enumerate(sys.argv):
    if arg == "--hf-cache-dir" and i + 1 < len(sys.argv):
        os.environ["HF_HOME"] = sys.argv[i + 1]
        print(f"Using HuggingFace cache directory: {sys.argv[i + 1]}")
        break

import util_model 
import util_data 
import torch
import random

# ========== Register Qwen3 support for pyvene ==========
# Qwen3 has the same architecture as Qwen2, so we reuse the same mappings
def register_qwen3_for_pyvene():
    """Register Qwen3 model types with pyvene's type mappings."""
    try:
        import transformers.models.qwen3.modeling_qwen3 as qwen3_modeling
        from pyvene.models.intervenable_modelcard import type_to_module_mapping, type_to_dimension_mapping
        from pyvene.models.qwen2.modelings_intervenable_qwen2 import (
            qwen2_type_to_module_mapping,
            qwen2_type_to_dimension_mapping,
            qwen2_lm_type_to_module_mapping,
            qwen2_lm_type_to_dimension_mapping,
            qwen2_classifier_type_to_module_mapping,
            qwen2_classifier_type_to_dimension_mapping,
        )
        
        # Register Qwen3 models using Qwen2 mappings (same architecture)
        if hasattr(qwen3_modeling, 'Qwen3Model'):
            type_to_module_mapping[qwen3_modeling.Qwen3Model] = qwen2_type_to_module_mapping
            type_to_dimension_mapping[qwen3_modeling.Qwen3Model] = qwen2_type_to_dimension_mapping
        
        if hasattr(qwen3_modeling, 'Qwen3ForCausalLM'):
            type_to_module_mapping[qwen3_modeling.Qwen3ForCausalLM] = qwen2_lm_type_to_module_mapping
            type_to_dimension_mapping[qwen3_modeling.Qwen3ForCausalLM] = qwen2_lm_type_to_dimension_mapping
        
        if hasattr(qwen3_modeling, 'Qwen3ForSequenceClassification'):
            type_to_module_mapping[qwen3_modeling.Qwen3ForSequenceClassification] = qwen2_classifier_type_to_module_mapping
            type_to_dimension_mapping[qwen3_modeling.Qwen3ForSequenceClassification] = qwen2_classifier_type_to_dimension_mapping
        
        print("Successfully registered Qwen3 support for pyvene")
    except ImportError as e:
        print(f"Warning: Could not register Qwen3 for pyvene: {e}")
    except Exception as e:
        print(f"Warning: Error registering Qwen3 for pyvene: {e}")

register_qwen3_for_pyvene()
# ========================================================
from tqdm.auto import tqdm
from tqdm import trange
from torch.utils.data import DataLoader
from pyvene import (
    IntervenableModel,
    VanillaIntervention,
    LowRankRotatedSpaceIntervention,
    RepresentationConfig,
    IntervenableConfig,
)
import json 
import argparse
import numpy as np

def compute_metrics(eval_preds, eval_labels):
    ''' This function is used to compute the accuracy of the predictions. '''
    total_count = 0
    correct_count = 0
    for eval_pred, eval_label in zip(eval_preds, eval_labels):
        total_count += 1
        correct_count += eval_pred == eval_label
    accuracy = float(correct_count) / float(total_count)
    return {"accuracy": accuracy}

def compute_loss(outputs, labels):
    ''' This function is used to compute the loss of the predictions. We will use cross entropy loss. '''
    CE = torch.nn.CrossEntropyLoss()
    return CE(outputs, labels)

def batched_random_sampler(data, batch_size):
    batch_indices = [_ for _ in range(int(len(data) / batch_size))]
    random.shuffle(batch_indices)
    for b_i in batch_indices:
        for i in range(b_i * batch_size, (b_i + 1) * batch_size):
            yield i

def set_random_seed(seed: int):
    """Set random seed for python, numpy and torch (CPU and CUDA).

    This helps reproducibility for dataset shuffling, model init and training.
    """
    import os
    # Python
    random.seed(seed)
    # OS-level hash seed
    os.environ['PYTHONHASHSEED'] = str(seed)
    # NumPy
    np.random.seed(seed)
    # Torch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Deterministic cuDNN (may slow down training)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def find_positional_indices(sample_input_id, tokenizer, sub_token = 'logic_function2('):
    ''' This function is used to find the positional indices of the first token in the input_ids after sub_token.
    This returns the position where {t0} appears in the formatted prompt.
    Input:
        sample_input_id: the input tensor of token ids, tensor with shape (1, k)
        tokenizer: the tokenizer used to tokenize the input, transformers.PreTrainedTokenizer
        sub_token: the sub_token to be found, str (default: 'logic_function2(')
    Output:
        pos_indices: the positional indices of the first token after the sub_token (i.e., position of {t0}). 
                     If there are multiple occurrences, return the last one. int and the length of input_ids'''
    
    # Handle shape (1, k) by squeezing or indexing
    if len(sample_input_id.shape) == 2 and sample_input_id.shape[0] == 1:
        input_ids = sample_input_id.squeeze(0)
    elif len(sample_input_id.shape) == 1:
        input_ids = sample_input_id
    else:
        raise ValueError(f"Unexpected shape for sample_input_id: {sample_input_id.shape}")
    
    # Decode the full sequence to find the text position
    full_text = tokenizer.decode(input_ids, skip_special_tokens=False)
    
    # Find the last occurrence of the sub_token in the decoded text
    text_pos = full_text.rfind(sub_token)
    
    if text_pos == -1:
        print(f"Warning: '{sub_token}' not found in decoded text")
        print(f"Decoded text: {full_text}")
        return -1, len(input_ids)
    
    # Now find which token position corresponds to the character position after sub_token
    target_char_pos = text_pos + len(sub_token)
    
    # Decode token by token to find the position
    current_text = ""
    last_occurrence_pos = -1
    
    for i in range(len(input_ids)):
        current_text = tokenizer.decode(input_ids[:i+1], skip_special_tokens=False)
        if len(current_text) >= target_char_pos:
            last_occurrence_pos = i
            break
    
    # print(f"Position of first argument (t0): {last_occurrence_pos}")
    return last_occurrence_pos, len(input_ids)

def DAS_training(intervenable, train_dataset, optimizer, pos, device, epochs = 10, batch_size = 64, gradient_accumulation_steps = 1):
    '''Main code for training the model with DAS intervention.
    Input:
        intervenable: the model with the intervention, pyvene.IntervenableModel
        train_dataset: the training dataset, contain input_ids, source_input_ids, labels
        optimizer: the optimizer to be used, torch.optim.Adam
        pos: the position of the intervention, int
        device: the device id to be used, int
        epochs: the number of epochs to train, int
        batch_size: the batch size to be used, int
        gradient_accumulation_steps: the number of steps to accumulate gradients, int
    Output:
        None, the model will be trained in-place.
    This function will train the model with the intervention, and compute the loss and accuracy.
    '''
    
    intervenable.model.train()  # set the module to train mode, which enables drop-off but no grads
    print("intervention trainable parameters: ", intervenable.count_parameters()) # count the number of trainable parameters in the intervention

    train_iterator = trange(0, int(epochs), desc="Epoch")  # create a progress bar for the epochs
    total_step = 0
    for epoch in train_iterator:
        epoch_iterator = tqdm( # create a progress bar for the batches
            DataLoader(
                train_dataset,
                batch_size=batch_size,
                sampler=batched_random_sampler(train_dataset, batch_size),
            ),
            desc=f"Epoch: {epoch}",
            position=0,
            leave=False,
            dynamic_ncols=True,
        )

        for batch in epoch_iterator:
            # Jiyuan: Need to verify the shape of input_ids and source_input_ids. The code should be correct, but I don't remember the exact shape.
            batch["input_ids"] = batch["input_ids"].squeeze(1).squeeze(1)
            batch["source_input_ids"] = batch["source_input_ids"].squeeze(1).squeeze(1)
            #print(batch["input_ids"].shape, batch["source_input_ids"].shape)
            batch_size = batch["input_ids"].shape[0]
            for k, v in batch.items():
                if v is not None and isinstance(v, torch.Tensor):
                    batch[k] = v.to(f"cuda:{device}")

            # Interchange intervention: Please pay attention to the shape. It can be tricky.
            _, counterfactual_outputs = intervenable(
                {"input_ids": batch["input_ids"]},
                [
                    {"input_ids": batch["source_input_ids"]},
                ],
                {
                    "sources->base": (
                        [[[pos]] * batch_size], [[[pos]] * batch_size],
                    )
                },
                subspaces=[
                    [[0]] * batch_size,
                ],
            )
            # compute metrics
            eval_metrics = compute_metrics(
            counterfactual_outputs.logits[:,-1,:].argmax(dim=-1), batch["labels"].squeeze()
            )

            # loss and backprop
            loss = compute_loss(
                counterfactual_outputs.logits[:,-1,:], batch["labels"].squeeze()
            )

            epoch_iterator.set_postfix(
                {"loss": f"{loss.item():.4f}", "acc": f"{eval_metrics['accuracy']:.4f}"},
                refresh=True
            )
            train_iterator.set_postfix(
                {"loss": f"{loss.item():.4f}", "acc": f"{eval_metrics['accuracy']:.4f}"},
                refresh=True
            )

            if gradient_accumulation_steps > 1:
                loss = loss / gradient_accumulation_steps
            loss.backward()

            if total_step % gradient_accumulation_steps == 0:
                optimizer.step()
                intervenable.set_zero_grad()
            total_step += 1

        epoch_iterator.close()  # Close inner progress bar after each epoch

def das_test(intervenable, pos, test_dataset, device, batch_size = 64, intervention_type = 'das'):
    ''' This function is used to test the model with the intervention.
    Input:
        intervenable: the model with the intervention, pyvene.IntervenableModel
        pos: the position of the intervention, int
        test_dataset: the testing dataset, contain input_ids, source_input_ids, labels
        device: the device id to be used, int
        batch_size: the batch size to be used, int
    Output:
        acc: the accuracy of the model, float
    This function will test the model with the intervention, and compute the accuracy.'''
    eval_labels = []
    eval_preds = []
    with torch.no_grad():
        epoch_iterator = tqdm(
            DataLoader(
                test_dataset,
                batch_size=batch_size,
                sampler=batched_random_sampler(test_dataset,batch_size),
            ),
            desc=f"Testing",
            position=0,
            leave=False,
        )

        for batch in epoch_iterator:
            batch["input_ids"] = batch["input_ids"].squeeze(1).squeeze(1)
            batch["source_input_ids"] = batch["source_input_ids"].squeeze(1).squeeze(1)
            #print(batch["input_ids"].shape, batch["source_input_ids"].shape)
            batch_size = batch["input_ids"].shape[0]
            for k, v in batch.items():
                if v is not None and isinstance(v, torch.Tensor):
                    batch[k] = v.to(f"cuda:{device}")
            
            if intervention_type == 'das':
                _, counterfactual_outputs = intervenable(
                    {"input_ids": batch["input_ids"]},
                    [
                        {"input_ids": batch["source_input_ids"]},
                    ],
                    {
                        "sources->base": (
                            [[[pos]] * batch_size], [[[pos]] * batch_size],
                        )
                    },
                    subspaces=[
                        [[0]] * batch_size,
                    ],
                )
            elif intervention_type == 'vanilla':
                _, counterfactual_outputs = intervenable(
                    {"input_ids": batch["input_ids"]},
                    [
                        {"input_ids": batch["source_input_ids"]},
                    ],
                    {
                        "sources->base": (
                            [[[pos]] * batch_size], [[[pos]] * batch_size],
                        )
                    }
                )
            else:
                raise ValueError("intervention_type must be 'das' or 'vanilla'")
            eval_labels += [batch["labels"].squeeze()]
            eval_preds += [counterfactual_outputs.logits[:,-1,:].argmax(dim=-1)]
   
    eval_labels = torch.cat(eval_labels)
    eval_preds = torch.cat(eval_preds)
    acc = compute_metrics(eval_preds, eval_labels)["accuracy"]
    return acc

def config_das(model, layer, device, weight=None, subspace_dimension=1):
    '''The function is used to set up the configuration for DAS intervention and wrap the model as an IntervenableModel.
    Input: 
        model: the model to be used
        layer: the layer to be used for intervention
        device: the device to be used
        weight: the weight of the intervention, optional
        weight: the weight of the intervention, optional
    Output:
        intervenable: the model with the intervention
    This function will create an IntervenableModel with the given configuration.'''
    config = IntervenableConfig(
            model_type = type(model),
            representations=[
                RepresentationConfig(
                    layer,              # layer
                    "block_output",          # component
                    "pos",              # intervention unit
                    1,                  # max number of unit
                    low_rank_dimension = subspace_dimension, # low rank dimension
                    subspace_partition = [[0, subspace_dimension]],
                ),
            ],
            intervention_types=LowRankRotatedSpaceIntervention,
        )
    intervenable = IntervenableModel(config, model)
    if weight is not None:
        # Set the weight of the intervention (single layer, always index #0)
        # Use load_state_dict if weight is a state_dict (OrderedDict)
        intervenable.interventions[f"layer_{layer}_comp_block_output_unit_pos_nunit_1#0"].rotate_layer.load_state_dict(weight)
    
    intervenable.set_device(device)
    intervenable.disable_model_gradients()
    return intervenable

def config_vanilla(model, layer, device):
    '''The function is used to set up the configuration for vanilla intervention and wrap the model as an IntervenableModel.
    Input: 
        model: the model to be used
        layer: the layer to be used for intervention
        device: the device to be used
    Output:
        intervenable: the model with the intervention
    This function will create an IntervenableModel with the given configuration.'''
    config = IntervenableConfig(
            model_type = type(model),
            representations=[
                {
                    "layer": layer,              # layer
                   "component": "block_output",          # component
                    "unit": "pos",              # intervention unit
                    "max_number_of_units": 1,
                },
            ],
            intervention_types=VanillaIntervention,
        )
    intervenable = IntervenableModel(config, model)
    intervenable.set_device(device)
    intervenable.disable_model_gradients()
    return intervenable

def save_weight(weights, name: str, path: str):
    # Save the model state dict
    if not os.path.exists(path):
        os.mkdir(path)
    name = os.path.join(path, name)
    # Check that all values are dicts of state_dicts (not modules)
    for key, subdict in weights.items():
        if not isinstance(subdict, dict):
            raise RuntimeError(f"weights['{key}'] is not a dict (got {type(subdict)})")
        for subkey, value in subdict.items():
            if not isinstance(value, dict):
                raise RuntimeError(f"weights['{key}']['{subkey}'] is not a state_dict (got {type(value)})")
    # Save state dict of interventions
    torch.save(weights, name)
    print(f"Model saved to {name}")

def load_weight(name):
    # Load the model state dict
    if os.path.exists(name):
        weights = torch.load(name)
        print("Weights loaded successfully!")
    else:
        print(f"Did not find existing model from {name}")
    return weights

def find_candidate_alignments(
    model,
    dataset,
    poss,
    layers,
    batch_size,
    device,
    n_candidates = 10,
    subspace_dimension = 1
):
    ''' This function is used to find the candidate alignments for the intervention.
    Input: 
        model: the model with the intervention
        dataset: the dataset
        poss: the positions of the intervention
        device: the device to be used
        batch_size: the batch size
        n_candidates: the number of candidates to be found
    Output:
        candidates: the candidates for the intervention
        weights: the weights of the candidates
    '''
    candidates = {}
    weights = {}
    # split dataset into training and testing
    train_dataset = dataset[:int(len(dataset) * 0.6)]
    test_dataset = dataset[int(len(dataset) * 0.6):]

    # Create directory for partial results if it doesn't exist
    os.makedirs("results", exist_ok=True)
    
    total_iterations = len(layers) * len(list(poss))
    current_iteration = 0

    for layer in layers:
        intervenable = config_das(model, layer, device, subspace_dimension=subspace_dimension)
        for pos in poss:
            current_iteration += 1
            print(f"\n[{current_iteration}/{total_iterations}] Processing Layer {layer}, Position {pos}")
            
            # create optimizer
            optimizer_params = []
            for k, v in intervenable.interventions.items():
                optimizer_params += [{"params": v.rotate_layer.parameters()}]
            optimizer = torch.optim.Adam(optimizer_params, lr=0.001)
            # train the model
            DAS_training(intervenable, train_dataset, optimizer, pos=pos, device=device, epochs=5, batch_size=batch_size)
            # test the model
            intervenable.disable_model_gradients()
            acc = das_test(intervenable, pos, test_dataset, device=device, batch_size=batch_size)
            candidates[(layer, pos)] = acc
            print(f"Layer {layer}, Position {pos}: Accuracy = {acc:.4f}")
            
            # Take a safe snapshot of the rotate_layer state_dict so later in-place
            # changes to the intervenable don't mutate previously stored weights.
            sd = intervenable.interventions[f"layer_{layer}_comp_block_output_unit_pos_nunit_1#0"].rotate_layer.state_dict()
            weights[(layer, pos)] = {k: v.clone().detach().cpu() for k, v in sd.items()}
            
            # Save partial results after each iteration
            partial_candidates = {f"L{k[0]}_P{k[1]}": v for k, v in candidates.items()}
            partial_weights = {f"L{k[0]}_P{k[1]}": v for k, v in weights.items()}
            
            with open("results/candidates_partial.json", "w") as f:
                json.dump(partial_candidates, f, indent=4)
            
            torch.save(partial_weights, "results/weights_partial.pt")
            print(f"Partial results saved ({current_iteration}/{total_iterations} completed)")

    # sort the candidates by accuracy
    candidates = sorted(candidates.items(), key=lambda x: x[1], reverse=True)
    # keep only the top n_candidates
    candidates = candidates[:n_candidates]
    # convert to dict
    candidates = {f"L{k[0]}_P{k[1]}": v for k, v in candidates}

    # keep the corresponding weights of the candidates
    weights = {f"L{k[0]}_P{k[1]}": v for k, v in weights.items() if f"L{k[0]}_P{k[1]}" in candidates.keys()}

    return candidates, weights

def extract_layer_pos(string):
    layer, pos = string.split('_')  # Split on underscore -> ["L5", "P78"]
    layer_num = int(layer[1:])      # Remove "L" and convert to int -> 5
    pos_num = int(pos[1:])          # Remove "P" and convert to int -> 78
    return layer_num, pos_num

def select_candidates(node, candidates, causal_model,dataset_generator, weights):
    ''' This function is used to select the candidates for the intervention.
    Input: 
        candidates: the candidates for the intervention
        weights: the weights of the candidates
        dataset_generator: the dataset generator input: intervention
    Output:
        selected_candidates: the selected candidates for the intervention
    More work needs to be done here.
    '''
    children = causal_model.paraents[node]
    if len(children) == 0:
        candidates = candidates[node]
        # return the candidiate with highest accuracy
        selected_candidates = sorted(candidates.items(), key=lambda x: x[1], reverse=True)
        selected_candidates = selected_candidates[0]
        return selected_candidates
    
    dataset = dataset_generator(children)

    rec_score = {}

    for candidate in candidates[node].keys():
        # extract layer and pos from candidate
        layer, pos = extract_layer_pos(candidate)

        # TBD 

def test_with_weights(model, layer, device, pos, test_dataset, batch_size=64, intervention_type = 'das', weight = None, subspace_dimension=1):
    ''' This function is used to test the model with pre-trained intervention weights.
    Input:
        model: the model to be used
        layer: the layer to be used for intervention
        device: the device to be used
        pos: the position of the intervention, int
        test_dataset: the testing dataset, contain input_ids, source_input_ids, labels
        weight: the pre-trained weight (state_dict) of the intervention
        batch_size: the batch size to be used, int
    Output:
        acc: the accuracy of the model, float
    This function will create an intervenable model with pre-trained weights and test its accuracy.'''
    
    # Create intervenable model with the pre-trained weight
    if intervention_type == 'das':
        intervenable = config_das(model, layer, device, weight, subspace_dimension=subspace_dimension)
    elif intervention_type == 'vanilla':
        intervenable = config_vanilla(model, layer, device)
    else:
        raise ValueError("intervention_type must be 'das' or 'vanilla'")
    
    # Test the model using the existing das_test function
    acc = das_test(intervenable, pos, test_dataset, device=device, batch_size=batch_size, intervention_type=intervention_type)
    
    return acc



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run DAS training or testing with selectable causal model")
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument("--train", action="store_true", help="Run training to find candidate alignments")
    mode_group.add_argument("--test", action="store_true", help="Run testing using precomputed weights and candidates")
    parser.add_argument("--model-id", type=str, default="Qwen/Qwen3-14B", help="HuggingFace model ID to use")
    parser.add_argument("--causal-model", choices=["1", "2"], default="1", help="Which causal model to use: 1 (default) or 2")
    parser.add_argument("--intervention-type", type=str, default='das', help="Type of intervention to use (e.g., 'das')")
    parser.add_argument("--weights-path", type=str, default=None, help="Path to das weights (.pt) for test mode")
    parser.add_argument("--candidates-path", type=str, default=None, help="Path to candidates JSON for test mode")
    parser.add_argument("--data-size", type=int, default=1024, help="Number of examples to generate per dataset")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for dataset creation and evaluation")
    parser.add_argument("--subspace-dimension", type=int, default=1, help="Dimension of the subspace for intervention")
    parser.add_argument("--device", type=int, default=0, help="Device to use (0 refers to cuda:0, -2 refer auto, -1 refers cpu). If not set, auto-detects.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--local-model-dir", type=str, default=None, help="Directory to load/save model snapshot (speeds up subsequent runs)")
    parser.add_argument("--hf-cache-dir", type=str, default=None, help="HuggingFace cache directory (sets HF_HOME env var, must be set before imports)")
    args = parser.parse_args()

    # Set random seed early for reproducibility
    set_random_seed(args.seed)
    print(f"Using random seed: {args.seed}")

    # construct causal model based on selection
    if args.causal_model == "1":
        or_causal_model = util_data.build_causal_model()
    elif args.causal_model == "2":        
        or_causal_model = util_data.build_causal_model2()
    else:
        raise RuntimeError(f"Unsupported causal model selection: {args.causal_model}")
    
    # load trained model
    
    # Device selection: -2 for auto (all GPUs), -1 for CPU, >= 0 for specific CUDA device
    if args.device == -2:
        # Use device_map='auto' for model loading, but set device to 0 for intervention operations
        device_map = "auto" if torch.cuda.is_available() else None
        device = 0 if torch.cuda.is_available() else -1
    else:
        device_map = None
        device = args.device
    
    model, tokenizer = util_model.get_model_and_tokenizer(args.model_id, hf_token=os.environ.get("HF_TOKEN"), local_dir=args.local_model_dir, device=args.device)
    print(f"Using device: {device}")
    
    if device_map == "auto" and torch.cuda.is_available():
        # Model will be automatically distributed across all available GPUs
        num_gpus = torch.cuda.device_count()
        print(f"Model distributed across all {num_gpus} available GPUs with device_map='auto'")
        print(f"Using device {device} for intervention operations")
    elif device >= 0 and torch.cuda.is_available():
        model = model.to(f"cuda:{device}")
        print(f"Current GPU: {torch.cuda.current_device()}")

    # create dataset params
    data_size = args.data_size
    batch_size = args.batch_size

    weights = {}

    # Train/test common params
    poss = range(76, 82) 
    # Handle different model configs: Qwen uses num_hidden_layers, GPT-2 uses n_layer
    num_layers = getattr(model.config, 'num_hidden_layers', getattr(model.config, 'n_layer', None))
    if num_layers is None:
        raise ValueError("Could not determine number of layers from model config")
    layers = range(num_layers)

    if args.causal_model == "1":
        op_list = ["op1", "op2", "op3"]
        data_generator = "all"
        out_op = "op3"
    elif args.causal_model == "2":
        op_list = ["op4a", "op5a"]
        data_generator = "all2"
        out_op = "op6a"
    causal_model_tag = f"or_model_{args.causal_model}"
    intervention_type = args.intervention_type
    subspace_dimension=args.subspace_dimension

    if args.train:
        print("Starting training (finding candidate alignments)")
        candidates_total = {}
        das_weights = {}

        for intervention in op_list:
            candidates_total[intervention] = {}
            das_weights[intervention] = {}
            dataset = util_data.make_counterfactual_dataset(
                data_generator,
                intervention,
                out_op,
                or_causal_model,
                model,
                tokenizer,
                data_size,
                device,
                batch_size=batch_size,
            )
            pos_after_sub_token, input_length = find_positional_indices(
                sample_input_id=dataset[0]["input_ids"],
                tokenizer=tokenizer,
                sub_token='Please evaluate: logic_function2('
            )
            pos_after_sub_token2, input_length = find_positional_indices(
                sample_input_id=dataset[1]["input_ids"],
                tokenizer=tokenizer,
                sub_token='Please evaluate: logic_function2('
            )
            if pos_after_sub_token != pos_after_sub_token2:
                raise RuntimeError("The position after sub_token is not consistent across samples.")
            poss = range(pos_after_sub_token+2, input_length)
            print(f"Searching positions from {pos_after_sub_token+2} to {input_length-1} for intervention {intervention}")

            print(f"Dataset created for {intervention}")
            print(f"Finding candidates for {intervention}")
            candidate, weight = find_candidate_alignments(
                model,
                dataset,
                poss,
                layers,
                batch_size,
                device,
                n_candidates=len(layers)*len(poss),
                subspace_dimension=args.subspace_dimension
            )
            candidates_total[intervention].update(candidate)
            das_weights[intervention].update(weight)

        # persist results
        os.makedirs("training_results", exist_ok=True)
        intervention_type = 'das'
        with open(f"training_results/candidates_{intervention_type}_{causal_model_tag}_dim{subspace_dimension}.json", "w") as f:
            json.dump(candidates_total, f, indent=4)
        print(f"Candidate alignments saved to training_results/candidates_{intervention_type}_{causal_model_tag}_dim{subspace_dimension}.json")

        with open(f"training_results/das_weights_{intervention_type}_{causal_model_tag}_dim{subspace_dimension}.pt", "wb") as f:
            torch.save(das_weights, f)
        print(f"DAS weights saved to training_results/das_weights_{intervention_type}_{causal_model_tag}_dim{subspace_dimension}.pt")

    elif args.test:
        print("Starting testing using provided weights and candidates")
        if args.intervention_type == 'das':
            if not args.weights_path:
                args.weights_path = f"training_results/das_weights_{intervention_type}_{causal_model_tag}_dim{subspace_dimension}.pt"
            if not args.candidates_path:
                args.candidates_path = f"training_results/candidates_{intervention_type}_{causal_model_tag}_dim{subspace_dimension}.json"

            # load provided artifacts
            das_weights = load_weight(args.weights_path)
            with open(args.candidates_path, "r") as f:
                candidates_total = json.load(f)

        test_results = {}

        if args.causal_model == "1":
            data_generator = "exhaustive"

        elif args.causal_model == "2":
            data_generator = "exhaustive2"

        for intervention in op_list:
            types = util_data.corresponding_intervention(intervention)
            test_results[intervention] = {}
            for source_code, base_code in types:
                print(f"Creating dataset for {intervention}, source: {source_code}, base: {base_code}")
                dataset = util_data.make_counterfactual_dataset(
                    data_generator,
                    intervention,
                    out_op,
                    or_causal_model,
                    model,
                    tokenizer,
                    data_size,
                    device,
                    batch_size=batch_size,
                    source_code=source_code,
                    base_code=base_code,
                )
                print(f"Dataset created for {intervention}, source: {source_code}, base: {base_code}\r")

                # get the candidates for this intervention
                candidates = candidates_total.get(intervention, {})
                weights_for_intervention = das_weights.get(intervention, {}) if isinstance(das_weights, dict) else das_weights

                results = {}
                for candidate in candidates.keys():
                    layer, pos = extract_layer_pos(candidate)
                    weight = weights_for_intervention.get(candidate)
                    if weight is None:
                        print(f"Warning: weight for candidate {candidate} not found; skipping")
                        continue
                    acc = test_with_weights(
                        model,
                        layer,
                        device,
                        pos,
                        dataset,
                        batch_size=batch_size,
                        intervention_type=intervention_type,
                        weight=weight,
                        subspace_dimension=args.subspace_dimension
                    )
                    print(f"Source: {source_code}, Base: {base_code}, Candidate: {candidate}, Accuracy: {acc:.4f} \r")
                    results[candidate] = acc

                test_results[intervention]["s" + source_code + "_b" + base_code] = results

                # Save partial results after each evaluation
                os.makedirs("test_results", exist_ok=True)
                with open(f"test_results/test_results_partial_{intervention_type}_{causal_model_tag}.json", "w") as f:
                    json.dump(test_results, f, indent=4)
                print(f"Partial test results saved after {intervention}")

        # Save final test results
        with open(f"test_results/test_results_{intervention_type}_{causal_model_tag}_dim{subspace_dimension}.json", "w") as f:
            json.dump(test_results, f, indent=4)
        print(f"Final test results saved to test_results/test_results_{intervention_type}_{causal_model_tag}_dim{subspace_dimension}.json")

    # Release the GPU memory
    model.cpu()
    torch.cuda.empty_cache()
    print("GPU memory cleared")




# The end
# Parallel intervention code that may be useful later:
# def config_das_parallel(model, layers, device, weights=None):
#     ''' This function is used to set up the configuration for parallel interchange intervention.
#     Input: 
#         model: the model to be used
#         locs: list of layer, the locations of the intervention
#         device: the device to be used
#         weights: list the weights of the intervention, in the same order as layers
#     Output:
#         intervenable: the model with the intervention
#     '''
#     representations = []
#     for layer in layers:
#         representations.append(
#             RepresentationConfig(
#                 layer,              # layer
#                 "block_output",          # component
#                 "pos",              # intervention unit
#                 1,                  # max number of unit
#                 low_rank_dimension = 1, # low rank dimension
#                 subspace_partition = [[0, 1]],
#                 intervention_link_key=0
#             )
#         )
#     config = IntervenableConfig(
#             model_type = type(model),
#             representations=representations,
#             intervention_types=LowRankRotatedSpaceIntervention,
#         )
    
#     intervenable = IntervenableModel(config, model)
#     if weights is not None:
#         # Set the weights of the intervention
#         rec_layer = {}
#         for i, layer in enumerate(layers):
#             if layer not in rec_layer:
#                 rec_layer[layer] = 0
#             else:
#                 rec_layer[layer] += 1
#             # Use load_state_dict if weights[i] is a state_dict (OrderedDict)
#             intervenable.interventions[f"layer_{layer}_comp_block_output_unit_pos_nunit_1#{rec_layer[layer]}"].rotate_layer.load_state_dict(weights[i])
#             # Use load_state_dict if weights[i] is a state_dict (OrderedDict)
#             intervenable.interventions[f"layer_{layer}_comp_block_output_unit_pos_nunit_1#{rec_layer[layer]}"].rotate_layer.load_state_dict(weights[i])
        
#     intervenable.set_device(device)
#     intervenable.disable_model_gradients()
#     return intervenable

# def parallel_intervention(intervenable, poss, test_dataset, batch_size):
#     ''' This function is used to set up the parallel intervention.
#     Input: 
#         intervenable: the model with the intervention
#         pos: the position of the intervention
#         batch_size: the batch size
#     Output:
#         acc: the accuracy of the model
#     '''

#     eval_labels = []
#     eval_preds = []
#     n_blocks = len(poss)
#     with torch.no_grad():
#         epoch_iterator = tqdm(
#             DataLoader(
#                 test_dataset,
#                 batch_size=batch_size,
#                 sampler=batched_random_sampler(test_dataset, batch_size),
#             ),
#             desc=f"Testing",
#             position=0,
#             leave=False,
#         )

#         for batch in epoch_iterator:
#             batch["input_ids"] = batch["input_ids"].squeeze(1).squeeze(1)
#             batch["source_input_ids"] = batch["source_input_ids"].squeeze(1).squeeze(1)
#             #print(batch["input_ids"].shape, batch["source_input_ids"].shape)
#             batch_size = batch["input_ids"].shape[0]
#             for k, v in batch.items():
#                 if v is not None and isinstance(v, torch.Tensor):
#                     batch[k] = v.to("cuda")

#             _, counterfactual_outputs = intervenable(
#                 {"input_ids": batch["input_ids"]},
#                 [
#                     {"input_ids": batch["source_input_ids"]},
#                 ] * n_blocks,
#                 {
#                     "sources->base": tuple(
#                        [[[[pos]] * batch_size] * 2 for pos in poss]
#                     )
#                 },
#                 subspaces=[
#                     [[0]] * batch_size,
#                 ] * n_blocks,
#             )
#             eval_labels += [batch["labels"].squeeze()]
#             eval_preds += [counterfactual_outputs.logits[:,-1,:].argmax(dim=-1)]
#     eval_labels = torch.cat(eval_labels)
#     eval_preds = torch.cat(eval_preds)
#     acc = compute_metrics(eval_preds, eval_labels)["accuracy"]
#     return acc
