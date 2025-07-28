        
'''Unused make_counterfactual_dataset functions stored here for reference'''

def make_counterfactual_dataset_ft(causal_model, vocab, intervention:str, samplesize:int):
    dataset = []
    # Inetrvention: FALSE to TRUE, SOURCE -> BASE
    # OP1: TFF -> FTF
    # OP2: FTF -> TFF
    # OP3: FFT -> FFF
    # OP4: TTF -> FFF
    # OP5: TTT -> FFF
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
        # Create intervened input by copying base_id and applying interventions
        intervened_id = base_id.copy()
        intervened_id[intervention] = dp["source_labels"][0][intervention]
        
        dp["intervened_input_ids"] = intervened_id
        dp["labels"] = causal_model.run_forward(intervened_id)
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
        # Create intervened input by copying base_id and applying interventions
        intervened_id = base_id.copy()
        intervened_id[intervention] = dp["source_labels"][0][intervention]
        
        dp["intervened_input_ids"] = intervened_id
        dp["labels"] = causal_model.run_forward(intervened_id)
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
        # Create intervened input by copying base_id and applying interventions
        intervened_id = base_id.copy()
        intervened_id[intervention] = dp["source_labels"][0][intervention]
        
        dp["intervened_input_ids"] = intervened_id
        dp["labels"] = causal_model.run_forward(intervened_id)
        # print(f"Base: {base_id} label: {p and q and r}, \nsource: {source_id}, label: {ps and qs and rs}\nlabel after interchange: {dp["labels"]}")
        dataset.append(dp)
    return dataset