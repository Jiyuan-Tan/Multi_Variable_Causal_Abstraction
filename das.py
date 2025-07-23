import util_model as util_model
import util_data as util_data
import torch
import os
import random
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
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # Only GPU 1 will be visible. MAKE SURE TO CHANGE THIS TO YOUR PREFERRED GPU

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

def DAS_training(intervenable, train_dataset, optimizer, pos, epochs = 5, batch_size = 64, gradient_accumulation_steps = 1):
    '''Main code for training the model with DAS intervention.
    Input:
        intervenable: the model with the intervention, pyvene.IntervenableModel
        train_dataset: the training dataset, contain input_ids, source_input_ids, labels
        optimizer: the optimizer to be used, torch.optim.Adam
        pos: the position of the intervention, int
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
                    batch[k] = v.to("cuda")

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

def das_test(intervenable, pos, test_dataset, batch_size = 64):
    ''' This function is used to test the model with the intervention.
    Input:
        intervenable: the model with the intervention, pyvene.IntervenableModel
        pos: the position of the intervention, int
        test_dataset: the testing dataset, contain input_ids, source_input_ids, labels
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
                    batch[k] = v.to("cuda")

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
            eval_labels += [batch["labels"].squeeze()]
            eval_preds += [counterfactual_outputs.logits[:,-1,:].argmax(dim=-1)]
    eval_labels = torch.cat(eval_labels)
    eval_preds = torch.cat(eval_preds)
    acc = compute_metrics(eval_preds, eval_labels)["accuracy"]
    return acc

def config_das(model, layer, device):
    '''The function is used to set up the configuration for DAS intervention and wrap the model as an IntervenableModel.
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
                RepresentationConfig(
                    layer,              # layer
                    "block_output",          # component
                    "pos",              # intervention unit
                    1,                  # max number of unit
                    low_rank_dimension = 1, # low rank dimension
                    subspace_partition = [[0, 1]],
                ),
            ],
            intervention_types=LowRankRotatedSpaceIntervention,
        )
    intervenable = IntervenableModel(config, model)
    intervenable.set_device(device)
    intervenable.disable_model_gradients()
    return intervenable

def config_das_parallel(model, layers, device, weights=None):
    ''' This function is used to set up the configuration for parallel interchange intervention.
    Input: 
        model: the model to be used
        locs: list of layer, the locations of the intervention
        device: the device to be used
        weights: list the weights of the intervention, in the same order as layers
    Output:
        intervenable: the model with the intervention
    '''
    representations = []
    for layer in layers:
        representations.append(
            RepresentationConfig(
                layer,              # layer
                "block_output",          # component
                "pos",              # intervention unit
                1,                  # max number of unit
                low_rank_dimension = 1, # low rank dimension
                subspace_partition = [[0, 1]],
                intervention_link_key=0
            )
        )
    config = IntervenableConfig(
            model_type = type(model),
            representations=representations,
            intervention_types=LowRankRotatedSpaceIntervention,
        )
    
    intervenable = IntervenableModel(config, model)
    if weights is not None:
        # Set the weights of the intervention
        rec_layer = {}
        for i, layer in enumerate(layers):
            if layer not in rec_layer:
                rec_layer[layer] = 0
            else:
                rec_layer[layer] += 1
            intervenable.interventions[f"layer_{layer}_comp_block_output_unit_pos_nunit_1#{rec_layer[layer]}"].rotate_layer.weight = weights[i].to("cpu")
        
    intervenable.set_device(device)
    intervenable.disable_model_gradients()
    return intervenable

def parallel_intervention(intervenable, poss, test_dataset, batch_size):
    ''' This function is used to set up the parallel intervention.
    Input: 
        intervenable: the model with the intervention
        pos: the position of the intervention
        batch_size: the batch size
    Output:
        acc: the accuracy of the model
    '''

    eval_labels = []
    eval_preds = []
    n_blocks = len(poss)
    with torch.no_grad():
        epoch_iterator = tqdm(
            DataLoader(
                test_dataset,
                batch_size=batch_size,
                sampler=batched_random_sampler(test_dataset, batch_size),
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
                    batch[k] = v.to("cuda")

            _, counterfactual_outputs = intervenable(
                {"input_ids": batch["input_ids"]},
                [
                    {"input_ids": batch["source_input_ids"]},
                ] * n_blocks,
                {
                    "sources->base": tuple(
                       [[[[pos]] * batch_size] * 2 for pos in poss]
                    )
                },
                subspaces=[
                    [[0]] * batch_size,
                ] * n_blocks,
            )
            eval_labels += [batch["labels"].squeeze()]
            eval_preds += [counterfactual_outputs.logits[:,-1,:].argmax(dim=-1)]
    eval_labels = torch.cat(eval_labels)
    eval_preds = torch.cat(eval_preds)
    acc = compute_metrics(eval_preds, eval_labels)["accuracy"]
    return acc

def save_weight(weights, name: str, path: str):
    # Save the model state dict
    if not os.path.exists(path):
        os.mkdir(path)
    name = os.path.join(path, name)
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

    for layer in layers:
        intervenable = config_das(model, layer, device)
        for pos in poss:
            # create optimizer
            optimizer_params = []
            for k, v in intervenable.interventions.items():
                optimizer_params += [{"params": v.rotate_layer.parameters()}]
                break
            optimizer = torch.optim.Adam(optimizer_params, lr=0.001)
            # train the model
            DAS_training(intervenable,train_dataset,optimizer,pos=pos,epochs=5,batch_size=batch_size)
            # test the model
            intervenable.disable_model_gradients()
            acc = das_test(intervenable, pos, test_dataset, batch_size)
            candidates[(layer, pos)] = acc
            weights[(layer, pos)] = intervenable.interventions[f"layer_{layer}_comp_block_output_unit_pos_nunit_1#0"].rotate_layer

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

if __name__ == "__main__":
    # load data
    vocab, texts, labels = util_data.get_vocab_and_data()

    # construct causal model
    or_causal_model = util_data.build_causal_model2(vocab)

    # load trained model
    model, tokenizer = util_model.load_model()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    model = model.to(device)
    if torch.cuda.is_available():
        print(f"Current GPU: {torch.cuda.current_device()}")

    # create dataset
    data_size = 1024
    interv = "op4"
    batch_size = 32

    # # Split the dataset into training and testing sets
    # training_size = int(len(dataset) * 0.6)
    # train_dataset = dataset[:training_size]
    # test_dataset = dataset[training_size:]

    weights = {}

    # Train the das
    poss = range(76, 82)
    layers = range(model.config.n_layer)

    candidates = {}

    # #load candidates
    # with open("candidates.json", "r") as f:
    #     candidates = json.load(f)

    # # load weights
    # weights = load_weight("das_weights/das_weights.pt")

    
    # layers = [5, 6]
    # poss = [78, 80]
    # w = [weights["op1"]["L5_P78"], weights["op2"]["L5_P80"]]
    # interventions = ["op1", "op2"]
    # intervenable = config_das_parallel(model, layers, device, w)
    
    # create dataset
    for intervention in ["op4", "op5", "op6"]:
        print(f"Creating dataset for {intervention}")
        dataset = util_data.make_counterfactual_dataset(
            "all2",
            [intervention],
            vocab,
            texts,
            labels,
            "op6",
            or_causal_model,
            model,
            tokenizer,
            data_size,
            device, 
            batch_size=batch_size
        )
        print(f"Dataset created for {intervention}")
        print(f"Finding candidates for {intervention}")
        candidate, weight = find_candidate_alignments(
            model,
            dataset,
            poss,
            layers,
            batch_size,
            device,
            n_candidates=72
        )
        candidates[intervention] = candidate
        weights[intervention] = weight
        
    # save the candidates
    with open(f"candidates2.json", "w") as f:
        json.dump(candidates, f, indent=4)
    print("Candidates saved to candidates2.json")


    #acc = parallel_intervention(intervenable, poss, dataset, batch_size)

    #print(f"Accuracy: {acc:.4f}")

    # Release the GPU memory
    model.cpu()
    torch.cuda.empty_cache()
    print("GPU memory cleared")