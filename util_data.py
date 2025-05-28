"""This file creates data for fine-tining and testing the model for a specific logic task."""
import random
import pandas as pd
import pyvene as pv
import numpy as np
from pyvene import CausalModel

def get_vocab_and_data():
    '''Load the vocabulary and data from the specified files. 
    The vocabulary is loaded from vocab.csv (containing 11162 words) and the training data is loaded from sample_data_zen_20000_0312.tsv. 
    The function returns the vocabulary, texts, and labels. '''
    vocab = pd.read_csv("vocab.csv", header=None)
    vocab = vocab.iloc[:, 1].tolist()
    df = pd.read_csv("./data/sample_data_zen_20000_0312.tsv", sep="\t")
    labels = [str(label) for label in df["labels"].tolist()]
    texts = [str(text) for text in df["texts"].tolist()]
    return vocab, texts, labels

def build_causal_model(vocab, model_type = 'or_model'):

    '''
    Build a causal model based on the specified model type. The function returns a CausalModel object. 
    The defined causal model below is
                    op5(or)
                  /         \
            op4(and)         \
              /   \           \
         op1(!=) op2(!=)     op3(==)
            |       |          |
         t2, t4   t0, t5      t1, t3
    '''

    if model_type == 'or_model':
        variables = ["t0", "t1", "t2", "t3", "t4", "t5", "op1", "op2", "op3", "op4", "op5"]
        reps = vocab
        values = {variable: reps for variable in ["t0", "t1", "t2", "t3", "t4", "t5"]}
        for i in range(1, 8):
            values[f"op{i}"] = [True, False] 

        parents = {
            "t0": [], "t1": [], "t2": [], "t3": [], 
            "t4": [], "t5": [], "t6": [], "t7": [],
            "op1": ["t2", "t4"],
            "op2": ["t0", "t5"],
            "op3": ["t1", "t3"],
            "op4": ["op1", "op2"],
            "op5": ["op3", "op4"],
        }

        def FILLER():
            return reps[0]


        functions = {
            "t0": FILLER, "t1": FILLER, "t2": FILLER, "t3": FILLER, "t4": FILLER, "t5": FILLER,
            "op1": lambda x, y:  x != y,
            "op2": lambda x, y:  x != y,
            "op3": lambda x, y: x == y,
            "op4": lambda x, y: x and y,
            "op5": lambda x, y: x or y,
            #"op7": lambda x, y: x and y,
        }

        # Define positions for visualization
        pos = {
            "t0": (.1,0), "t1": (.3,0), "t2": (.5,0), "t3": (.7,0), "t4": (.9,0), "t5": (1.1,0),
            "op1": (.4,1), "op2": (.6,1), "op3": (.8,1), "op4": (.6,2), "op5": (.6,3),
        }

        # Create the causal model
        causal_model = CausalModel(variables, values, parents, functions, pos=pos)

    else:
        # We only support the or_model for now
        raise ValueError("Model type not supported. We only support 'or_model' for now.")
    return causal_model

# Process input 
def format_input(raw_input, context_texts, context_labels):
    '''
    This function formats the input for the model. 
    It returns a formatted string that in-context learning friendly: 
        context1=label1
        ...
        contextn=labeln
        word0,word1,word2,word3,word4,word5=
    '''
    input = ",".join([
        str(raw_input[var]) for var in ["t0", "t1", "t2", "t3", "t4", "t5"] 
    ])
    contexts = ""
    for i in range(len(context_texts)):
        ##print(context)
        contexts += f"{context_texts[i]}={str(context_labels[i])}\n"
    return contexts + input + "="


# filter dataset
def data_filter(causal_model, model, tokenizer, dataset, device, batch_size = 16):
    new_dataset = []

    for i in range(0, len(dataset), batch_size):
        batch = dataset[i : i + batch_size]

        # Prepare all base inputs
        base_texts = [
            format_input(dp["input_ids"], dp["context_texts"], dp["context_labels"])
            for dp in batch
        ]
        if len(base_texts) == 0:
            continue

        # Tokenize base inputs at once
        base_tokenized = tokenizer(
            base_texts,
            return_tensors="pt",
        ).to(device)

        # Forward pass for the entire batch
        base_outputs = model(base_tokenized["input_ids"], return_dict=True)
        # Get the last token predictions for each item in the batch
        base_logits = base_outputs.logits[:, -1, :]
        base_ids = base_logits.argmax(dim=-1)
        base_decoded = [tokenizer.decode(idx) for idx in base_ids]

        # Prepare all source inputs
        source_texts = [
            format_input(dp["source_input_ids"][0], dp["context_texts_source"], dp["context_labels_source"])
            for dp in batch
        ]
        source_tokenized = tokenizer(
            source_texts,
            return_tensors="pt"
        ).to(device)

        # Forward pass for source
        source_outputs = model(source_tokenized["input_ids"], return_dict=True)
        source_logits = source_outputs.logits[:, -1, :]
        source_ids = source_logits.argmax(dim=-1)
        source_decoded = [tokenizer.decode(idx) for idx in source_ids]

        # Compare outputs with labels
        for dp, b_out, s_out in zip(batch, base_decoded, source_decoded):
            base_label = str(dp["base_labels"]["op5"])
            source_label = str(causal_model.run_forward(dp["source_input_ids"][0])["op5"])
            if b_out == base_label and s_out == source_label:
                # Keep only valid data points
                new_dataset.append(dp)
    
    return new_dataset


# Inetrvention: FALSE to TRUE, SOURCE -> BASE
# OP1: TFF -> FTF
# OP2: FTF -> TFF
# OP3: FFT -> FFF
# OP4: TTF -> FFF
# OP5: TTT -> FFF
def make_counterfactual_dataset_ft(causal_model, vocab, intervention:str, samplesize:int):
    dataset = []

    for _ in range(samplesize):
        # default: FFF
        t0, t1, t2, t3 = random.choice(vocab), random.choice(vocab), random.choice(vocab), random.choice(vocab), 
        t4 = t2 if intervention != "op2" else random.choice(vocab)
        t5 = t0 if intervention != "op1" else random.choice(vocab)

        p, q, r= (t2 != t4), (t0 != t5), (t1 == t3)
    
        base_id = {
            "t0": t0,
            "t1": t1,
            "t2": t2,
            "t3": t3,
            "t4": t4,
            "t5": t5,
        }
        dp = {"input_ids": base_id}
        dp["base_labels"] = {"op1": p, "op2": q, "op3": r, "op4": p and q, "op5": (p and q) or r}
        
        
        t0s, t1s, t2s, t3s, t4s, t5s = random.choice(vocab), random.choice(vocab), random.choice(vocab), random.choice(vocab), random.choice(vocab) , random.choice(vocab)
        # default: TTF
        # t5s = t0s
        # t2s = t4s

        if intervention == "op1" or intervention =="op3": 
            t0s = t5s
        if intervention =="op2" or intervention =="op3": 
            t2s = t4s
        if intervention =="op5" or intervention =="op3": 
            t1s = t3s

        source_id = {
            "t0": t0s,
            "t1": t1s,
            "t2": t2s,
            "t3": t3s,
            "t4": t4s,
            "t5": t5s,
        }

        ps, qs, rs = (t2s != t4s), (t0s != t5s), (t1s == t3s)
        dp["source_input_ids"] = [source_id]
        dp["source_labels"] = [{"op1": ps, "op2": qs, "op3": rs, "op4": ps and qs, "op5": (ps and qs) or rs}]
        base_id[intervention] =  dp["source_labels"][0][intervention]
        dp["labels"] = causal_model.run_forward(base_id)
        #print(f"Base: {base_id} label: {p and q and r}, \nsource: {source_id}, label: {ps and qs and rs}\nlabel after interchange: {dp["labels"]}")
        dataset.append(dp)
    return dataset

def make_counterfactual_dataset_fixed(causal_model, vocab, intervention:str, samplesize:int):
    dataset = []

    for _ in range(samplesize):
        # sample t0, t1, t2, t3, t4 such that only intervention is False
        # t0 = random.choice(vocab) 
        # t5 = t0 if random.random() < 0.5 else random.choice(vocab)
        # t1 = random.choice(vocab)
        # t3 = t1 if random.random() < 0.5 else random.choice(vocab)
        # t4 = random.choice(vocab)
        # t2 = t4 if random.random() < 0.5 else random.choice(vocab)
        # defaylt: TTF
        t0, t1, t2, t3, t4, t5 = random.choice(vocab), random.choice(vocab), random.choice(vocab), random.choice(vocab), random.choice(vocab) , random.choice(vocab)
        if intervention == "op1" or intervention == "op5":
            t2 = t4 
        if intervention == "op2" or intervention == "op5":
            t0 = t5
        if intervention == "op3":
            t2 = t4
            t0 = t5
            t1 = t3
    
        p, q, r= (t2 != t4), (t0 != t5), (t1 == t3)
    
        base_id = {
            "t0": t0,
            "t1": t1,
            "t2": t2,
            "t3": t3,
            "t4": t4,
            "t5": t5,
        }
        dp = {"input_ids": base_id}
        dp["base_labels"] = {"op1": p, "op2": q, "op3": r, "op4": p and q, "op5": (p and q) or r}
        
        
        t0s, t1s, t2s, t3s, t4s, t5s = random.choice(vocab), random.choice(vocab), random.choice(vocab), random.choice(vocab), random.choice(vocab) , random.choice(vocab)

        t5s = t0s = t5
        t2s = t4s = t4

        # t0s, t1s, t2s, t3s, t4s = t0, t1, t2, t3, t4
        # default: TTF
        
        if intervention == "op1" or intervention == "op4" or intervention == "op5":
            t2s = t4s if t2 != t4 else random.choice(vocab)
        if intervention == "op2" or intervention == "op4" or intervention == "op5":
            t0s = t5s if t0 != t5 else random.choice(vocab)
        if intervention == "op3" or intervention == "op5":
            t0s, t2s = random.choice(vocab), random.choice(vocab)
            t1s = t3s if t1 != t3 else random.choice(vocab)

        source_id = {
            "t0": t0s,
            "t1": t1s,
            "t2": t2s,
            "t3": t3s,
            "t4": t4s,
            "t5": t5s,
        }

        ps, qs, rs = (t2s != t4s), (t0s != t5s), (t1s == t3s)
        dp["source_input_ids"] = [source_id]
        dp["source_labels"] = [{"op1": ps, "op2": qs, "op3": rs, "op4": ps and qs, "op5": (ps and qs) or rs}]
        base_id[intervention] =  dp["source_labels"][0][intervention]
        dp["labels"] = causal_model.run_forward(base_id)
        # print(f"Base: {base_id} label: {p and q and r}, \nsource: {source_id}, label: {ps and qs and rs}\nlabel after interchange: {dp["labels"]}")
        dataset.append(dp)
    return dataset

def make_counterfactual_dataset_average(causal_model, vocab, intervention:str, samplesize:int):
    dataset = []

    for _ in range(samplesize):
        # Base input:
        # OP1: FTF
        # OP2: TFF
        # OP3: FFT
        # OP4: TTF
        # OP5: TTF
        t0, t1, t2, t3, t4, t5 = random.choice(vocab), random.choice(vocab), random.choice(vocab), random.choice(vocab), random.choice(vocab) , random.choice(vocab)
        if (intervention == "op1" or intervention == "op5"):
            t2 = t4 
        if (intervention == "op2" or intervention == "op5"):
            t0 = t5
        if intervention == "op3" and random.random() < 0.5:
            t2 = t4
            t0 = t5
            t1 = t3

        if intervention == "op4" and random.random() < 0.5:
            t1 = t3
    
        p, q, r= (t2 != t4), (t0 != t5), (t1 == t3)
    
        base_id = {
            "t0": t0,
            "t1": t1,
            "t2": t2,
            "t3": t3,
            "t4": t4,
            "t5": t5,
        }
        dp = {"input_ids": base_id}
        dp["base_labels"] = {"op1": p, "op2": q, "op3": r, "op4": p and q, "op5": (p and q) or r}
        
        
        t0s, t1s, t2s, t3s, t4s, t5s = random.choice(vocab), random.choice(vocab), random.choice(vocab), random.choice(vocab), random.choice(vocab) , random.choice(vocab)

        t5s = t0s if random.random() < 0.5 else random.choice(vocab)
        t2s = t4s if random.random() < 0.5 else random.choice(vocab)

        # t0s, t1s, t2s, t3s, t4s = t0, t1, t2, t3, t4
        
        if (intervention == "op1" or intervention == "op4" or intervention == "op5"):
            t2s = t4s if t2 != t4 else random.choice(vocab)
        if intervention == "op2" or intervention == "op4" or intervention == "op5":
            t0s = t5s if t0 != t5 else random.choice(vocab)
        if intervention == "op3" or intervention == "op5":
            t0s, t2s = random.choice(vocab), random.choice(vocab)
            t1s = t3s if t1 != t3 else random.choice(vocab)
        if intervention == "op4" and random.random() < 0.5:
            t1s = t3s

        source_id = {
            "t0": t0s,
            "t1": t1s,
            "t2": t2s,
            "t3": t3s,
            "t4": t4s,
            "t5": t5s,
        }

        ps, qs, rs = (t2s != t4s), (t0s != t5s), (t1s == t3s)
        dp["source_input_ids"] = [source_id]
        dp["source_labels"] = [{"op1": ps, "op2": qs, "op3": rs, "op4": ps and qs, "op5": (ps and qs) or rs}]
        base_id[intervention] =  dp["source_labels"][0][intervention]
        dp["labels"] = causal_model.run_forward(base_id)
        # print(f"Base: {base_id} label: {p and q and r}, \nsource: {source_id}, label: {ps and qs and rs}\nlabel after interchange: {dp["labels"]}")
        dataset.append(dp)
    return dataset

def make_counterfactual_dataset_all(causal_model, vocab, interventions:list, samplesize:int):
    dataset = []
    for _ in range(samplesize):
        # Base input:
        # OP1: FTF
        # OP2: TFF
        # OP3: FFT
        # OP4: TTF
        # OP5: TTF
        t0, t1, t2, t3, t4, t5 = random.choice(vocab), random.choice(vocab), random.choice(vocab), random.choice(vocab), random.choice(vocab) , random.choice(vocab)
        if random.random() < 0.5:
            t2 = t4
        if random.random() < 0.5:
            t0 = t5
        if random.random() < 0.5:
            t1 = t3

        p, q, r= (t2 != t4), (t0 != t5), (t1 == t3)
    
        base_id = {
            "t0": t0,
            "t1": t1,
            "t2": t2,
            "t3": t3,
            "t4": t4,
            "t5": t5,
        }
        dp = {"input_ids": base_id}
        dp["base_labels"] = {"op1": p, "op2": q, "op3": r, "op4": p and q, "op5": (p and q) or r}
        
        
        t0s, t1s, t2s, t3s, t4s, t5s = random.choice(vocab), random.choice(vocab), random.choice(vocab), random.choice(vocab), random.choice(vocab) , random.choice(vocab)

        t5s = t5 if random.random() < 0.5 else random.choice(vocab)
        t4s = t4 if random.random() < 0.5 else random.choice(vocab)
        t3s = t3 if random.random() < 0.5 else random.choice(vocab)

        t5s = t0s if random.random() < 0.5 else random.choice(vocab)
        t2s = t4s if random.random() < 0.5 else random.choice(vocab)
        t1s = t3s if random.random() < 0.5 else random.choice(vocab)

        source_id = {
            "t0": t0s,
            "t1": t1s,
            "t2": t2s,
            "t3": t3s,
            "t4": t4s,
            "t5": t5s,
        }

        ps, qs, rs = (t2s != t4s), (t0s != t5s), (t1s == t3s)
        dp["source_input_ids"] = [source_id]
        dp["source_labels"] = [{"op1": ps, "op2": qs, "op3": rs, "op4": ps and qs, "op5": (ps and qs) or rs}]
        for intervention in interventions:
            base_id[intervention] =  dp["source_labels"][0][intervention]
        dp["labels"] = causal_model.run_forward(base_id)
        # print(f"Base: {base_id} label: {p and q and r}, \nsource: {source_id}, label: {ps and qs and rs}\nlabel after interchange: {dp["labels"]}")
        dataset.append(dp)
    return dataset

def make_counterfactual_dataset(
    dataset_type, 
    interv,
    vocab,
    texts,
    labels,
    equality_model,
    model,
    tokenizer,
    data_size,
    device, 
    batch_size = 32):
    '''This function generates a counterfactual tokenized dataset. The output dataset is already filtered and tokenized.'''

    if dataset_type == "fixed":
        make_raw_data = make_counterfactual_dataset_fixed
    elif dataset_type == "average":
        make_raw_data = make_counterfactual_dataset_average
    elif dataset_type == "fixed_f2t":
        make_raw_data = make_counterfactual_dataset_ft
    elif dataset_type == "all":
        make_raw_data = make_counterfactual_dataset_all
    else:
        raise ValueError("dataset_type should be one of ['fixed', 'average', 'fixed_f2t']")

    dataset =  make_raw_data(equality_model, vocab, interv, data_size)

    # create context for base and source data
    for dp in dataset:
        indices = random.sample(range(len(texts)), 5)
        dp["context_texts"] = [texts[j] for j in indices]
        dp["context_labels"] = [labels[j] for j in indices]
        indices = random.sample(range(len(texts)), 5)
        dp["context_texts_source"] = [texts[j] for j in indices]
        dp["context_labels_source"] = [labels[j] for j in indices]

    dataset = data_filter(equality_model, model, tokenizer, dataset, device, batch_size=batch_size)
    print(f"The sample size of dataset is {len(dataset)}, batch size {batch_size}")

    # Attach the context for each entry in the dataset and tokenize
    data_tokenized = []
    for i in range(len(dataset)):
        dp = dataset[i]
        base_texts = format_input(dp["input_ids"], dp["context_texts"], dp["context_labels"])

        source_texts = format_input(dp["source_input_ids"][0], dp["context_texts_source"], dp["context_labels_source"])
        data_tokenized.append(
            {
                "input_ids": tokenizer(base_texts, return_tensors="pt")["input_ids"].to(device),
                "source_input_ids": tokenizer(source_texts, return_tensors="pt")["input_ids"].to(device),
                "labels": tokenizer(str(dp["labels"]['op5']), return_tensors="pt")["input_ids"].to(device),
                "source_labels": tokenizer(str(dp["source_labels"][0]['op5']), return_tensors="pt")["input_ids"].to(device),
            }
        )
    return data_tokenized