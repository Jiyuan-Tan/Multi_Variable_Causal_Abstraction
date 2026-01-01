import os
import transformers
import torch

def get_model_and_tokenizer(model_id: str, hf_token: str | None, local_dir: str | None, num_gpus: int = 1):
    """Load AutoModelForCausalLM + AutoTokenizer, optionally from a local snapshot.

    Replaces the previous transformers.pipeline usage so we can directly control batching and decoding.
    """
    auth_token = hf_token if hf_token else None
    use_local = False
    if local_dir and os.path.isdir(local_dir) and os.path.isfile(os.path.join(local_dir, "config.json")):
        use_local = True

    load_source = local_dir if use_local else model_id
    source_desc = f"local directory '{local_dir}'" if use_local else f"model repo '{model_id}'"
    print(f"Loading model + tokenizer from {source_desc} ...")

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        load_source,
        token=auth_token,
        use_fast=True,
    )
    # Ensure pad token exists (Llama often lacks one); map to eos if missing
    if getattr(tokenizer, "pad_token", None) is None and getattr(tokenizer, "eos_token", None) is not None:
        tokenizer.pad_token = tokenizer.eos_token
    # For decoder-only models, use left padding for correct generation alignment
    tokenizer.padding_side = "left"

    if num_gpus > 1 or num_gpus == -1:
        # Use "auto" for multi-GPU - transformers will automatically distribute layers
        # This handles embed_tokens, lm_head, and all layers correctly
        device_map = "auto"
    elif num_gpus == 1:
        device_map = "cuda:0"
    elif num_gpus == 0:
        device_map = "cpu"
    else:
        device_map = "auto"
    
    model = transformers.AutoModelForCausalLM.from_pretrained(
        load_source,
        token=auth_token,
        dtype=torch.bfloat16,
        device_map=device_map
    )

    if not use_local and local_dir:
        try:
            os.makedirs(local_dir, exist_ok=True)
            print(f"Saving model snapshot to '{local_dir}' ...")
            model.save_pretrained(local_dir)
            tokenizer.save_pretrained(local_dir)
        except Exception as e:
            print(f"Warning: failed to save model locally: {e}")

    print("Model + tokenizer ready (bfloat16, device_map=auto).")
    return model, tokenizer