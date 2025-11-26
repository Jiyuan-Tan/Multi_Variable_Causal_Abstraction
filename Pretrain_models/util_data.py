from typing import List
import random
import pandas as pd
import pyvene as pv
import numpy as np
from pyvene import CausalModel
import re
import torch
from tqdm import tqdm

PROMPT_TEMPLATE1 = (
    "Task: Evaluate the Python expression and return the result.\n\n"
    "Examples:\n"
    "Logic function: \n\n"
    "def logic_function1(t0, t1, t2, t3):\n"
    "    return (t0 or t1) and (t2 and t3)\n\n"
    "logic_function1(True, True, False, True)=false\n"
    "logic_function1(False, True, True, True)=true\n"
    "logic_function1(True, False, True, False)=false\n"
    "Now we have a different logic function to evaluate. Return only 'true' or 'false' (lowercase, no punctuation). Don't use step by step reasoning or any additional text.\n\n"
    "def logic_function2(t0,t1,t2,t3):\n"
    "    return (t0 or t1) and t2 or t3\n\n"
    "Please evaluate: logic_function2({t0},{t1},{t2},{t3})="
)

SYSTEM_PROMPT = ("Provide your final answer as only the word true or false in lowercase. Do NOT use step-by-step reasoning or any additional text.")


def build_vocabulary(tokenizer) -> List[str]:
    """Return a richer list of token ids for sampling.

    Categories collected (all must be single-token encodings):
      - ASCII letters (a-z, A-Z) of any length
      - ASCII digits (0-9) of any length  
      - Selected punctuation/operators: + - * / 
      - Alphanumeric strings of any reasonable length (up to 20 chars)

    Exclusions:
      - Tokens that decode to 'true' or 'false' (to avoid answer leakage)
      - Tokens containing whitespace or non-ascii characters
      - Tokens that include internal special token markers

    The function includes strings of various lengths to provide a richer vocabulary.
    """
    vocab_size = getattr(tokenizer, "vocab_size", None)
    if vocab_size is None:
        vocab = tokenizer.get_vocab()
        vocab_size = max(vocab.values()) + 1

    valid_vocabulary: List[str] = []

    allowed_punct = set(["+", "-", "*", "/"])
    excluded_words = {"true", "false", "="}

    for idx in range(vocab_size):
        try:
            token = tokenizer.convert_ids_to_tokens(idx)
            # Quick reject for special tokens containing brackets or angle markers
            if any(mark in token for mark in ["<", ">", "[", "]"]):
                continue
            enc = tokenizer.encode(token, add_special_tokens=False)
            if len(enc) != 1:
                continue  # must correspond to a single token
            decoded = tokenizer.decode([idx], clean_up_tokenization_spaces=False).strip()
            if not decoded or not decoded.isascii():
                continue
            if any(ch.isspace() for ch in decoded):
                continue
            lower_decoded = decoded.lower()
            if lower_decoded in excluded_words:
                continue

            # Accept strings up to reasonable length (20 chars)
            if len(decoded) > 20:
                continue

            # Categorize and accept various types of tokens
            is_valid = False
            
            # Accept alphabetic strings of any length
            if decoded.isalpha():
                is_valid = True
            # Accept numeric strings of any length  
            elif decoded.isdigit():
                is_valid = True
            # Accept single character punctuation
            elif len(decoded) == 1 and decoded in allowed_punct:
                is_valid = True
            # Accept alphanumeric strings of any length
            elif all(ch.isalnum() for ch in decoded):
                is_valid = True
            
            if is_valid:
                valid_vocabulary.append(decoded)
                
        except Exception:
            continue

    return valid_vocabulary

def build_causal_model(model_type = 'or_model'):
    '''Build a causal model based on the specified model type. 
    The function returns a CausalModel object. 
    task: (op1 and op2) or op3'''
    if model_type == 'or_model':
        variables = ["t0", "t1", "t2", "t3", "op1", "op2", "op3"]
        reps = [True, False]
        values = {variable: reps for variable in ["t0", "t1", "t2", "t3"]}
        values["op1"] = [True, False]
        values["op2"] = [True, False]
        values["op3"] = [True, False]

        parents = {
            "t0": [],
            "t1": [],
            "t2": [],
            "t3": [],
            "op1": ["t0", "t1"],
            "op2": ["op1", "t2"],
            "op3": ["op2", "t3"],
        }


        def FILLER():
            return reps[0]


        functions = {
            "t0": FILLER,
            "t1": FILLER,
            "t2": FILLER,
            "t3": FILLER,
            "op1": lambda x, y:  x or y,
            "op2": lambda x, y:  x and y,
            "op3": lambda x, y: x or y,
        }

        pos = {
            "t0": (.1,0),
            "t1": (.3,0),
            "t2": (.5,0),
            "t3": (.7,0),
            "op1": (.4,1),
            "op2": (.6,1),
            "op3": (.8,1),
        }
        causal_model = CausalModel(variables, values, parents, functions, pos=pos)
    else:
        # We only support the or_model for now
        raise ValueError("Model type not supported. We only support 'or_model' for now.")
    return causal_model

def build_causal_model2(model_type = 'or_model'):
    '''Build a causal model based on the specified model type. 
    The function returns a CausalModel object. 
    task: (op1a or op3a) and (op2a or op3a)'''
    if model_type == 'or_model':
        variables = ["t0", "t1", "t2", "t3", "t4", "t5", "op1a", "op2a", "op3a", "op4a", "op5a", "op6a"]
        reps = vocab
        values = {variable: reps for variable in ["t0", "t1", "t2", "t3", "t4", "t5"]}
        values["op1a"] = [True, False]
        values["op2a"] = [True, False]
        values["op3a"] = [True, False]
        values["op4a"] = [True, False]
        values["op5a"] = [True, False]
        values["op6a"] = [True, False]

        parents = {
            "t0": [],
            "t1": [],
            "t2": [],
            "t3": [],
            "t4": [],
            "t5": [],
            "op1a": ["t2", "t4"],
            "op2a": ["t0", "t5"],
            "op3a": ["t1", "t3"],
            "op4a": ["op1a", "op3a"],
            "op5a": ["op3a", "op2a"],
            "op6a": ["op4a", "op5a"],
        }


        def FILLER():
            return reps[0]


        functions = {
            "t0": FILLER,
            "t1": FILLER,
            "t2": FILLER,
            "t3": FILLER,
            "t4": FILLER,
            "t5": FILLER,
            "op1a": lambda x, y:  x != y,
            "op2a": lambda x, y:  x != y,
            "op3a": lambda x, y: x == y,
            "op4a": lambda x, y: x or y,
            "op5a": lambda x, y: x or y,
            "op6a": lambda x, y: x and y,
        }

        pos = {
            "t0": (.1,0),
            "t1": (.3,0),
            "t2": (.5,0),
            "t3": (.7,0),
            "t4": (.9,0),
            "t5": (1.1,0),
            "op1a": (.4,1),
            "op2a": (.6,1),
            "op3a": (.8,1),
            "op4a": (.6,2),
            "op5a": (.6,3),
            "op6a": (.6,4),
        }
        causal_model = CausalModel(variables, values, parents, functions, pos=pos)
    else:
        # We only support the or_model for now
        raise ValueError("Model type not supported. We only support 'or_model' for now.")
    return causal_model

# Process input 
def format_input(raw_input, tokenizer):
    
    # Convert boolean values to lowercase strings
    formatted_input = {k: str(v).lower() for k, v in raw_input.items()}
    user_prompt = PROMPT_TEMPLATE1.format(**formatted_input)
    # add SYSTEM prompt using tokenizer.apply_chat_template
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt}
    ]
    formatted_input = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    return formatted_input

# filter dataset
def data_filter(op_out, causal_model, model, tokenizer, dataset, device, batch_size = 16):
    new_dataset = []

    for i in tqdm(range(0, len(dataset), batch_size), desc="Filtering dataset"):
        batch = dataset[i : i + batch_size]

        # Prepare all base inputs with system prompt
        base_texts = [
            format_input(dp["input_ids"], tokenizer) for dp in batch
        ]
        if len(base_texts) == 0:
            continue

        base_tokenized = tokenizer(
            base_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(device)

        # Forward pass for the entire batch (single next token prediction)
        with torch.no_grad():
            base_outputs = model(
                input_ids=base_tokenized["input_ids"],
                return_dict=True
            )
        # Get the last token predictions for each item in the batch
        base_logits = base_outputs.logits[:, -1, :]
        base_ids = base_logits.argmax(dim=-1)
        base_decoded = [tokenizer.decode(idx, skip_special_tokens=True).strip().lower() for idx in base_ids]

        # Prepare all source inputs with system prompt
        source_texts = [ format_input(dp["source_input_ids"][0], tokenizer) for dp in batch ]
        source_tokenized = tokenizer(
            source_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(device)

        # Forward pass for source (single next token prediction)
        with torch.no_grad():
            source_outputs = model(
                input_ids=source_tokenized["input_ids"],
                return_dict=True
            )
        source_logits = source_outputs.logits[:, -1, :]
        source_ids = source_logits.argmax(dim=-1)
        source_decoded = [tokenizer.decode(idx, skip_special_tokens=True).strip().lower() for idx in source_ids]

        # Compare outputs with labels
        for dp, b_out, s_out in zip(batch, base_decoded, source_decoded):
            # Convert boolean labels to expected string format ('true' or 'false')
            base_label = 'true' if dp["base_labels"][op_out] else 'false'
            source_label = 'true' if causal_model.run_forward(dp["source_input_ids"][0])[op_out] else 'false'
            # print(f"Base output: {b_out}, Base label: {base_label}, Source output: {s_out}, Source label: {source_label}")
            b_out = re.findall(r'\b(true|false)\b', b_out.lower())[-1]
            s_out = re.findall(r'\b(true|false)\b', s_out.lower())[-1]
            # compare outputs with labels
            if b_out == base_label and s_out == source_label:
                # Keep only valid data points
                new_dataset.append(dp)
    
    return new_dataset

def influenced_ops(source_code: str, base_code: str):
    """
    Returns the list of operations that are influenced when transitioning from source_code to base_code.
    
    Args:
        source_code: 3-bit binary string representing source state (e.g., "FFF", "FFT", etc.)
        base_code: 3-bit binary string representing base state (e.g., "FFF", "FFT", etc.)

    Returns:
        List of influenced operations, or empty list if no operations are influenced
    """
    interv_op_dict = {
        # src = 000 (FFF)
        ("FFF", "FFF"): [],
        ("FFF", "FFT"): ["op3", "op4a", "op5a"],
        ("FFF", "FTF"): [],
        ("FFF", "FTT"): ["op3", "op4a", "op5a"],
        ("FFF", "TFF"): [],
        ("FFF", "TFT"): ["op3", "op4a", "op5a"],
        ("FFF", "TTF"): ["op1", "op2", "op4", "op4a", "op5a"],
        ("FFF", "TTT"): ["op4a", "op5a"],
        
        # src = 001 (FFT)
        ("FFT", "FFF"): ["op3"],
        ("FFT", "FFT"): [],
        ("FFT", "FTF"): ["op3", "op4a"],
        ("FFT", "FTT"): [],
        ("FFT", "TFF"): ["op3", "op5a"],
        ("FFT", "TFT"): [],
        ("FFT", "TTF"): ["op1", "op2", "op4"],
        ("FFT", "TTT"): [],
        
        # src = 010 (FTF)
        ("FTF", "FFF"): [],
        ("FTF", "FFT"): ["op3", "op4a"],
        ("FTF", "FTF"): [],
        ("FTF", "FTT"): ["op3", "op4a"],
        ("FTF", "TFF"): ["op2", "op5a"],
        ("FTF", "TFT"): ["op3", "op4a"],
        ("FTF", "TTF"): ["op1", "op4", "op4a"],
        ("FTF", "TTT"): ["op4a"],
        
        # src = 011 (FTT)
        ("FTT", "FFF"): ["op3"],
        ("FTT", "FFT"): [],
        ("FTT", "FTF"): ["op3", "op4a"],
        ("FTT", "FTT"): [],
        ("FTT", "TFF"): ["op2", "op3", "op5a"],
        ("FTT", "TFT"): [],
        ("FTT", "TTF"): ["op1", "op4"],
        ("FTT", "TTT"): [],
        
        # src = 100 (TFF)
        ("TFF", "FFF"): [],
        ("TFF", "FFT"): ["op3", "op5a"],
        ("TFF", "FTF"): ["op1", "op4a"],  
        ("TFF", "FTT"): ["op3", "op5a"],
        ("TFF", "TFF"): [],
        ("TFF", "TFT"): ["op3", "op5a"],
        ("TFF", "TTF"): ["op2", "op4", "op5a"],
        ("TFF", "TTT"): ["op5a"],
        
        # src = 101 (TFT)
        ("TFT", "FFF"): ["op3"],
        ("TFT", "FFT"): [],
        ("TFT", "FTF"): ["op1", "op3", "op4a"],
        ("TFT", "FTT"): [],
        ("TFT", "TFF"): ["op3", "op5a"],
        ("TFT", "TFT"): [],
        ("TFT", "TTF"): ["op2", "op4"],
        ("TFT", "TTT"): [],
        
        # src = 110 (TTF)
        ("TTF", "FFF"): ["op4"],
        ("TTF", "FFT"): ["op3"],
        ("TTF", "FTF"): ["op1", "op4", "op4a"],
        ("TTF", "FTT"): ["op3"],
        ("TTF", "TFF"): ["op2", "op4", "op5a"],
        ("TTF", "TFT"): ["op3"],
        ("TTF", "TTF"): [],
        ("TTF", "TTT"): [],
        
        # src = 111 (TTT)
        ("TTT", "FFF"): ["op3", "op4"],
        ("TTT", "FFT"): [],
        ("TTT", "FTF"): ["op1", "op3", "op4", "op4a"],
        ("TTT", "FTT"): [],
        ("TTT", "TFF"): ["op2", "op3", "op4", "op5a"],
        ("TTT", "TFT"): [],
        ("TTT", "TTF"): [],
        ("TTT", "TTT"): [],
    }
    
    return interv_op_dict[(source_code, base_code)]

def corresponding_intervention(op:str):
    op_interv_dict = {
        "op1": [("FFF", "TTF"), ("FFT", "TTF"), ("FTF", "TTF"), ("FTT", "TTF"), ("TFF", "FTF"), ("TFT", "FTF"), ("TTF", "FTF"), ("TTT", "FTF")],
        "op2": [("FFF", "TTF"), ("FFT", "TTF"), ("FTF", "TFF"), ("FTT", "TFF"), ("TFF", "TTF"), ("TFT", "TTF"), ("TTF", "TFF"), ("TTT", "TFF")],
        "op3": [("FFF", "FFT"), ("FFF", "FTT"), ("FFF", "TFT"), ("FFT", "FFF"), ("FFT", "FTF"), ("FFT", "TFF"), ("FTF", "FFT"), ("FTF", "FTT"), ("FTF", "TFT"), ("FTT", "FFF"), ("FTT", "FTF"), ("FTT", "TFF"), ("TFF", "FFT"), ("TFF", "FTT"), ("TFF", "TFT"), ("TFT", "FFF"), ("TFT", "FTF"), ("TFT", "TFF"), ("TTF", "FFT"), ("TTF", "FTT"), ("TTF", "TFT"), ("TTT", "FFF"), ("TTT", "FTF"), ("TTT", "TFF")],
        "op4": [("FFF", "TTF"), ("FFT", "TTF"), ("FTF", "TTF"), ("FTT", "TTF"), ("TFF", "TTF"), ("TFT", "TTF"), ("TTF", "FFF"), ("TTF", "FTF"), ("TTF", "TFF"), ("TTT", "FFF"), ("TTT", "FTF"), ("TTT", "TFF")],
        "op4a": [("FFF", "FFT"), ("FFF", "FTT"), ("FFF", "TFT"), ("FFF", "TTF"), ("FFF", "TTT"), ("FFT", "FTF"), ("FTF", "FFT"), ("FTF", "FTT"), ("FTF", "TFT"), ("FTF", "TTF"), ("FTF", "TTT"), ("FTT", "FTF"), ("TFF", "FTF"), ("TFT", "FTF"), ("TTF", "FTF"), ("TTT", "FTF")],
        "op5a": [("FFF", "FFT"), ("FFF", "FTT"), ("FFF", "TFT"), ("FFF", "TTF"), ("FFF", "TTT"), ("FFT", "TFF"), ("FTF", "TFF"), ("FTT", "TFF"), ("TFF", "FFT"), ("TFF", "FTT"), ("TFF", "TFT"), ("TFF", "TTF"), ("TFF", "TTT"), ("TFT", "TFF"), ("TTF", "TFF"), ("TTT", "TFF")]
    }
    return op_interv_dict.get(op, [])

def make_counterfactual_dataset_exhaustive(causal_model, intervention:str, samplesize:int, source_code:str, base_code:str):
    """
    Create counterfactual dataset based on specific source and base binary codes.
    
    Args:
        causal_model: The causal model
        vocab: Vocabulary for token generation
        source_code: 3-bit binary string (e.g., "FFF", "FFT")
        base_code: 3-bit binary string (e.g., "FFF", "FFT") 
        samplesize: Number of samples to generate
    """
    dataset = []
    
    # Convert binary codes to boolean values
    # base_code: "FFF" -> p=False, q=False, r=False
    # base_code: "FFT" -> p=False, q=False, r=True
    # etc.
    p = base_code[0] == 'T'  # First bit
    q = base_code[1] == 'T'  # Second bit  
    r = base_code[2] == 'T'  # Third bit
    
    # Same for source
    ps = source_code[0] == 'T'
    qs = source_code[1] == 'T'
    rs = source_code[2] == 'T'
    
    for _ in range(samplesize):
        t0, t1, t2, t3, t4, t5 = random.choice(vocab), random.choice(vocab), random.choice(vocab), random.choice(vocab), random.choice(vocab) , random.choice(vocab)

        if p == False:
            t2 = t4
        if q == False:
            t0 = t5
        if r == True:
            t1 = t3
        
        base_id = {"t0": t0, "t1": t1, "t2": t2, "t3": t3, "t4": t4, "t5": t5}
        dp = {"input_ids": base_id}
        dp["base_labels"] = {"op1": p, "op2": q, "op3": r, "op4": p and q, "op5": (p and q) or r}  

        t0s, t1s, t2s, t3s, t4s, t5s = random.choice(vocab), random.choice(vocab), random.choice(vocab), random.choice(vocab), random.choice(vocab) , random.choice(vocab)
        if ps == False:
            t2s = t4s
        if qs == False:
            t0s = t5s
        if rs == True:
            t1s = t3s

        source_id = {"t0": t0s, "t1": t1s, "t2": t2s, "t3": t3s, "t4": t4s, "t5": t5s}
        dp["source_input_ids"] = [source_id]
        dp["source_labels"] = [{"op1": ps, "op2": qs, "op3": rs, "op4": ps and qs, "op5": (ps and qs) or rs}]

        intervened_id = base_id.copy()
        intervened_id[intervention] = dp["source_labels"][0][intervention]
        dp["intervened_input_ids"] = intervened_id
        dp["labels"] = causal_model.run_forward(intervened_id)
        dataset.append(dp)
    return dataset

def make_counterfactual_dataset_exhaustive2(causal_model, intervention:str, samplesize:int, source_code:str, base_code:str):
    """
    Create counterfactual dataset based on specific source and base binary codes.
    
    Args:
        causal_model: The causal model
        vocab: Vocabulary for token generation
        source_code: 3-bit binary string (e.g., "000", "001")
        base_code: 3-bit binary string (e.g., "000", "001") 
        samplesize: Number of samples to generate
    """
    dataset = []
    
    # Convert binary codes to boolean values
    # base_code: "FFF" -> p=False, q=False, r=False
    # base_code: "FFT" -> p=False, q=False, r=True
    # etc.
    p = base_code[0] == 'T'  # First bit
    q = base_code[1] == 'T'  # Second bit  
    r = base_code[2] == 'T'  # Third bit
    
    # Same for source
    ps = source_code[0] == 'T'
    qs = source_code[1] == 'T'
    rs = source_code[2] == 'T'
    
    for _ in range(samplesize):
        t0, t1, t2, t3, t4, t5 = random.choice(vocab), random.choice(vocab), random.choice(vocab), random.choice(vocab), random.choice(vocab) , random.choice(vocab)

        if p == False:
            t2 = t4
        if q == False:
            t0 = t5
        if r == True:
            t1 = t3
        
        base_id = {"t0": t0, "t1": t1, "t2": t2, "t3": t3, "t4": t4, "t5": t5}
        dp = {"input_ids": base_id}
        dp["base_labels"] = {"op1a": p, "op2a": q, "op3a": r, "op4a": p or r, "op5a": q or r, "op6a": (p and q) or r}

        t0s, t1s, t2s, t3s, t4s, t5s = random.choice(vocab), random.choice(vocab), random.choice(vocab), random.choice(vocab), random.choice(vocab) , random.choice(vocab)
        if ps == False:
            t2s = t4s
        if qs == False:
            t0s = t5s
        if rs == True:
            t1s = t3s

        source_id = {"t0": t0s, "t1": t1s, "t2": t2s, "t3": t3s, "t4": t4s, "t5": t5s}
        dp["source_input_ids"] = [source_id]
        dp["source_labels"] = [{"op1a": ps, "op2a": qs, "op3a": rs, "op4a": ps or rs, "op5a": qs or rs, "op6a": (ps and qs) or rs}]

        intervened_id = base_id.copy()
        intervened_id[intervention] = dp["source_labels"][0][intervention]
        dp["intervened_input_ids"] = intervened_id
        dp["labels"] = causal_model.run_forward(intervened_id)
        dataset.append(dp)
    return dataset

def make_counterfactual_dataset_all(causal_model, intervention:str, samplesize:int):
    dataset = []
    for _ in range(samplesize):
        # truth values of ops all randomized
        picks = [random.randint(0, 1) for _ in range(4)]
        t0, t1, t2, t3 = [bool(p) for p in picks]

        p = t0 or t1
        q = p and t2
        r = q or t3
        base_id = {"t0": t0, "t1": t1, "t2": t2, "t3": t3}

        dp = {"input_ids": base_id}
        dp["base_labels"] = {"op1": p, "op2": q, "op3": r}
        
        picks = [random.randint(0, 1) for _ in range(4)]
        t0s, t1s, t2s, t3s = [bool(p) for p in picks]



        source_id = {"t0": t0s,"t1": t1s, "t2": t2s, "t3": t3s}

        ps, qs, rs = t0s or t1s, (t0s or t1s) and t2s, (t0s or t1s) and t2s or t3s

        dp["source_input_ids"] = [source_id]
        dp["source_labels"] = [{"op1": ps, "op2": qs, "op3": rs}]
        # Create intervened input by copying base_id and applying interventions
        intervened_id = base_id.copy()
        intervened_id[intervention] = dp["source_labels"][0][intervention]
        
        dp["intervened_input_ids"] = intervened_id
        dp["labels"] = causal_model.run_forward(intervened_id)
        # content of dp: "input_ids" (t0 to t3), "base_labels" (op1 to op3), "source_input_ids" , "source_labels", "intervened_input_ids", "labels". 
        dataset.append(dp)
    return dataset

def make_counterfactual_dataset_all2(causal_model, intervention:str, samplesize:int):
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
    
        base_id = {"t0": t0, "t1": t1, "t2": t2, "t3": t3, "t4": t4, "t5": t5}
        dp = {"input_ids": base_id}
        dp["base_labels"] = {"op1a": p, "op2a": q, "op3a": r, "op4a": p or r, "op5a": q or r, "op6a": (p and q) or r}
        
        
        t0s, t1s, t2s, t3s, t4s, t5s = random.choice(vocab), random.choice(vocab), random.choice(vocab), random.choice(vocab), random.choice(vocab) , random.choice(vocab)

        t5s = t5 if random.random() < 0.5 else random.choice(vocab)
        t4s = t4 if random.random() < 0.5 else random.choice(vocab)
        t3s = t3 if random.random() < 0.5 else random.choice(vocab)

        t5s = t0s if random.random() < 0.5 else random.choice(vocab)
        t2s = t4s if random.random() < 0.5 else random.choice(vocab)
        t1s = t3s if random.random() < 0.5 else random.choice(vocab)

        source_id = {"t0": t0s, "t1": t1s, "t2": t2s, "t3": t3s, "t4": t4s, "t5": t5s}

        ps, qs, rs = (t2s != t4s), (t0s != t5s), (t1s == t3s)
        dp["source_input_ids"] = [source_id]
        dp["source_labels"] = [{"op1a": ps, "op2a": qs, "op3a": rs, "op4a": ps or rs, "op5a": qs or rs,"op6a": (ps and qs) or rs}]
        # Create intervened input by copying base_id and applying interventions
        intervened_id = base_id.copy()
        intervened_id[intervention] = dp["source_labels"][0][intervention]

        dp["intervened_input_ids"] = intervened_id
        dp["labels"] = causal_model.run_forward(intervened_id)
        # print(f"Base: {base_id} label: {p and q and r}, \nsource: {source_id}, label: {ps and qs and rs}\nlabel after interchange: {dp["labels"]}")
        dataset.append(dp)
    return dataset

def make_counterfactual_dataset(
    dataset_type, 
    interv,
    op_out,
    equality_model,
    model,
    tokenizer,
    data_size,
    device, 
    batch_size = 16,
    source_code = 'FFF',
    base_code = 'TTT'):
    '''This function generates a counterfactual tokenized dataset. The output dataset is already filtered and tokenized.'''

    if dataset_type == "all":
        make_raw_data = make_counterfactual_dataset_all
    elif dataset_type == "all2":
        make_raw_data = make_counterfactual_dataset_all2
    elif dataset_type == "exhaustive":
        make_raw_data = lambda equality_model, interv, data_size: make_counterfactual_dataset_exhaustive(equality_model, interv, data_size, source_code, base_code)
    elif dataset_type == "exhaustive2":
        make_raw_data = lambda equality_model, interv, data_size: make_counterfactual_dataset_exhaustive2(equality_model, interv, data_size, source_code, base_code)
    else:
        raise ValueError("dataset_type should be one of ['fixed', 'average', 'fixed_f2t', 'all', 'exhaustive', 'exhaustive2']")

    # vocab = build_vocabulary(tokenizer)
    dataset =  make_raw_data(equality_model, interv, data_size)

    dataset = data_filter(op_out, equality_model, model, tokenizer, dataset, device, batch_size=batch_size)
    print(f"The sample size of dataset is {len(dataset)}, batch size {batch_size}")

    data_tokenized = []
    for i in range(len(dataset)):
        dp = dataset[i]

        base_texts = format_input(dp["input_ids"], tokenizer)

        source_texts = format_input(dp["source_input_ids"][0], tokenizer)
        
        # Add intervened input tokenization
        intervened_texts = format_input(dp["intervened_input_ids"], tokenizer)
        

        data_tokenized.append(
            {
                "input_ids": tokenizer(base_texts, 
                                       return_tensors="pt",
                                       padding=True,
                                        truncation=True)["input_ids"].to(device),
                "source_input_ids": tokenizer(source_texts, return_tensors="pt", padding=True, truncation=True)["input_ids"].to(device),
                "intervened_input_ids": tokenizer(intervened_texts, return_tensors="pt", padding=True, truncation=True)["input_ids"].to(device),
                "labels": tokenizer(str(dp["labels"][op_out]).lower(), return_tensors="pt", padding=True, truncation=True)["input_ids"].to(device),
                "source_labels": tokenizer(str(dp["source_labels"][0][op_out]).lower(), return_tensors="pt", padding=True, truncation=True)["input_ids"].to(device),
            }
        )
    return data_tokenized