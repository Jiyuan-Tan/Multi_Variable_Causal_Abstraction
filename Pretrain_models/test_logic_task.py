#!/usr/bin/env python3
"""Download a Llama-8B-style model from Hugging Face, run a logical token task, and report accuracy.

Usage examples (PowerShell):
  export HF_TOKEN= (Replace with your token if needed) 
  python3 test_logic_task.py --num-examples 1000 --batch-size 16 --local-model-dir "./model_qwen13b" --prompt-version 3 --model-id Qwen/Qwen3-14B

Notes:
- This script loads the full model in fp16 (no 8-bit/bitsandbytes path).
- Provide a Hugging Face token via the `HF_TOKEN` env var if the model is gated.
"""
import os
import argparse
import random
import time
from typing import List

import torch
import transformers


PROMPT_TEMPLATE1 = (
    "Task: Evaluate the following Python expression and return the result.\n\n"
    "Logic function: \n\n"
    "def logic_function(t0, t1, t2, t3, t4, t5):\n"
    "    return ((t0 != t5) and (t2 != t4)) or (t1 == t3)\n\n"
    "Examples:\n"
    "logic_function(t0='apple', t1='book', t2='cat', t3='dog', t4='egg', t5='fish') → true\n"
    "logic_function(t0='x', t1='y', t2='z', t3='w', t4='z', t5='x') → false\n\n"
    "logic_function(t0='red', t1='blue', t2='green', t3='blue', t4='yellow', t5='red') → true\n"
    "Now evaluate: logic_function(t0='{t0}', t1='{t1}', t2='{t2}', t3='{t3}', t4='{t4}', t5='{t5}')\n\n"
    "Return ONLY 'true' or 'false' (lowercase, no punctuation).\n"
    "Answer:"
)

PROMPT_TEMPLATE2 = (
    "Task: Look at the pattern in these examples and determine the output for the given inputs. The pattern involves comparing strings for equality and inequality. The comparison pair is (t0, t5), (t1, t3), (t2, t4)\n\n"
    "Examples:\n"
    "t0=a, t1=b, t2=c, t3=d, t4=e, t5=f → true\n"
    "t0=g, t1=h, t2=i, t3=h, t4=j, t5=k → true\n"
    "t0=l, t1=m, t2=n, t3=o, t4=n, t5=l → false\n"
    "t0=p, t1=q, t2=r, t3=q, t4=r, t5=s → true\n"
    "t0=t, t1=u, t2=v, t3=u, t4=w, t5=t → true\n"
    "t0=a, t1=b, t2=c, t3=d, t4=c, t5=e → false\n"
    "t0=f, t1=g, t2=h, t3=g, t4=h, t5=f → true\n"
    "t0=i, t1=j, t2=k, t3=l, t4=m, t5=i → false\n"
    "Based on the pattern above, what should the output be for:\n"
    "t0={t0}, t1={t1}, t2={t2}, t3={t3}, t4={t4}, t5={t5}\n\n"
    "Think carefully about which variables are equal or not equal to each other. Respond with only true or false (lowercase, no punctuation).\n"
    "Answer:"
)

PROMPT_TEMPLATE3 = (
    "Task: Evaluate the Python expression and return the result.\n\n"
    "Examples:\n"
    "Logic function: \n\n"
    "def logic_function1(t0, t1, t2, t3):\n"
    "    return (t0 or t1) and (t2 and t3)\n\n"
    "logic_function1(t0=True, t1=True, t2=False, t3=True) → false\n"
    "logic_function1(t0=False, t1=True, t2=True, t3=True) → true\n"
    "logic_function1(t0=True, t1=False, t2=True, t3=False) → false\n"
    "Now we have a different logic function to evaluate:\n\n"
    "def logic_function2(t0, t1, t2, t3):\n"
    "    return (t0 or t1) and t2 or t3\n\n"
    "Please evaluate: logic_function2(t0={t0}, t1={t1}, t2={t2}, t3={t3})\n\n"
    "Return only 'true' or 'false' (lowercase, no punctuation). Do NOT use step-by-step reasoning or any additional text.\n"
    "Answer:"
)


def build_single_token_ids(tokenizer) -> List[int]:
    """Return a richer list of token ids for sampling.

    Categories collected (all must be single-token encodings):
      - ASCII letters (a-z, A-Z) of any length
      - ASCII digits (0-9) of any length  
      - Selected punctuation/operators: + - * / = ! ?
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

    valid_tokens: List[int] = []

    allowed_punct = set(["+", "-", "*", "/", "!", "?"])
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
                valid_tokens.append(idx)
                
        except Exception:
            continue

    return valid_tokens

def generate_examples(single_ids: List[int], tokenizer, n: int):
    examples = []
    for _ in range(n):
        picks = random.sample(single_ids, 6)
        # decode each single id into a readable string
        tokens = [tokenizer.decode([p], clean_up_tokenization_spaces=False) for p in picks]
        t0, t1, t2, t3, t4, t5 = tokens
        
        # uniformly sample the input by potentially forcing equality
        if random.random() < 0.5:  # force t2 == t4 half the time
            t2 = t4
        if random.random() < 0.5:  # force t0 == t5 half the time  
            t0 = t5
        if random.random() < 0.5:  # force t1 == t3 half the time
            t1 = t3
        
        # Update tokens list with potentially modified values
        final_tokens = [t0, t1, t2, t3, t4, t5]
        
        # logical ground truth: ((t2 != t4) and (t0 != t5)) or (t1 == t3)
        gt = ((t2 != t4) and (t0 != t5)) or (t1 == t3)
        examples.append(({f"t{i}": final_tokens[i] for i in range(6)}, gt))
    return examples

def generate_examples2(n: int):
    examples = []
    for _ in range(n):
        
        # sample ti = True/False uniformly
        picks = [random.randint(0, 1) for _ in range(4)]
        t0, t1, t2, t3 = [bool(p) for p in picks]
        
        # Compute ground truth based on PROMPT_TEMPLATE2 formula:
        # (t0 OR t1) AND t2 OR t3
        gt = ((t0 or t1) and t2 or t3)

        # Store as string representations for template formatting
        tokens_map = {f"t{i}": str([t0, t1, t2, t3][i]).lower() for i in range(4)}
        examples.append((tokens_map, gt))
    
    return examples


def parse_bool_from_text(text: str):
    import re
    s = text.lower()
    # Extract standalone boolean tokens (avoid substrings like 'untrue')
    candidates = re.findall(r"\b(true|false|yes|no)\b", s)
    if not candidates:
        return None
    # Prefer the LAST token to get the final answer after chain-of-thought reasoning
    token = candidates[-1]
    if token in ("true", "yes"):
        return True
    if token in ("false", "no"):
        return False
    return None


def get_model_and_tokenizer(model_id: str, hf_token: str | None, local_dir: str | None):
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

    model = transformers.AutoModelForCausalLM.from_pretrained(
        load_source,
        token=auth_token,
        torch_dtype=torch.bfloat16,
        device_map={"": 1},  # Use only GPU 1
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


def run(model_id: str, num_examples: int, hf_token: str | None, seed: int, batch_size: int, local_model_dir: str | None, prompt_version: int = 1):
    random.seed(seed)
    torch.manual_seed(seed)
    
    # Select the appropriate prompt template
    if prompt_version == 1:
        prompt_template = PROMPT_TEMPLATE1
    elif prompt_version == 2:
        prompt_template = PROMPT_TEMPLATE2
    elif prompt_version == 3:
        prompt_template = PROMPT_TEMPLATE3
    else:
        raise ValueError(f"Invalid prompt version: {prompt_version}. Must be 1, 2, or 3.")

    model, tokenizer = get_model_and_tokenizer(model_id, hf_token, local_model_dir)

    print("Building single-token list from tokenizer...")
    single_ids = build_single_token_ids(tokenizer)
    if len(single_ids) < 6:
        raise RuntimeError("Not enough single-token vocabulary items to build examples.")
    print(f"Found {len(single_ids)} single-token vocab entries.")

    print(f"Sampling {num_examples} test examples...")
    if prompt_version == 3:
        examples = generate_examples2(num_examples)
    else:
        examples = generate_examples(single_ids, tokenizer, num_examples)

    correct = 0
    total = 0
    t_start = time.time()
    # Batched generation to reduce sequential GPU pipeline warning
    for start in range(0, len(examples), batch_size):
        batch = examples[start:start + batch_size]
        # Build chat-formatted prompts using tokenizer's chat template if available
        prompt_texts = []
        for tokens_map, _ in batch:
            user_prompt = prompt_template.format(**tokens_map)
            messages = [
                {"role": "system", "content": "Provide your final answer as only the word true or false in lowercase. Do NOT use step-by-step reasoning or any additional text."},
                {"role": "user", "content": user_prompt},
            ]
            if hasattr(tokenizer, "apply_chat_template"):
                prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
            else:
                # Fallback: simple concatenation
                prompt_text = "\n".join([m["content"] for m in messages]) + "\n"
            prompt_texts.append(prompt_text)

        enc = tokenizer(prompt_texts, return_tensors="pt", padding=True)
        enc = {k: v.to(model.device) for k, v in enc.items()}
        input_lengths = [l.item() for l in (enc["attention_mask"].sum(dim=1))]

        gen_out = model.generate(
            **enc,
            max_new_tokens=5,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
        )

        for (tokens_map, gt), seq, in_len in zip(batch, gen_out, input_lengths):
            # Slice generated portion
            gen_tokens = seq[in_len:]
            text_val = tokenizer.decode(gen_tokens, skip_special_tokens=True)
            # print(f"   Raw generation: {text_val!r}")
            pred = parse_bool_from_text(text_val)
            is_correct = (pred is not None) and (pred == gt)
            if is_correct:
                correct += 1
            total += 1

            i = total
            if i % max(1, num_examples // 10) == 0:
                print(f"[{i}/{num_examples}] GT={gt}  PRED={pred}  -> {'OK' if is_correct else 'ERR'}")
            # if not is_correct:
            #     print(f"Input: {tokens_map}  Raw generation: {text_val!r}")

    elapsed = time.time() - t_start
    acc = correct / total if total else 0.0
    print("\n--- Results ---")
    print(f"Examples: {total}")
    print(f"Correct:  {correct}")
    print(f"Accuracy: {acc:.4f}")
    print(f"Elapsed:  {elapsed:.1f}s")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", type=str, default="meta-llama/Llama-3.1-8B-Instruct", help="Hugging Face model repo id")
    parser.add_argument("--num-examples", type=int, default=200, help="Number of test examples to sample")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed for reproducibility")
    # hugging face token from argument
    parser.add_argument("--hf-token", type=str, default=None, help="Hugging Face access token (overrides HF_TOKEN env var)")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for generation to improve GPU utilization")
    parser.add_argument("--local-model-dir", type=str, default=None, help="Directory to load/save model snapshot (speeds up subsequent runs)")
    parser.add_argument("--prompt-version", type=int, default=1, choices=[1, 2, 3], help="Prompt template version: 1=explicit formula, 2=pattern inference from examples, 3=boolean logic")
    args = parser.parse_args()

    hf_token = args.hf_token or os.environ.get("HF_TOKEN")
    run(args.model_id, args.num_examples, hf_token, args.seed, args.batch_size, args.local_model_dir, args.prompt_version)


if __name__ == "__main__":
    main()
    # clear GPU memory
    torch.cuda.empty_cache()
