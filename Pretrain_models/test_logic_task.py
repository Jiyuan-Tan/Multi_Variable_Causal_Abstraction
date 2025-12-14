#!/usr/bin/env python3
"""Download a Llama-8B-style model from Hugging Face, run a logical token task, and report accuracy.

Usage examples (PowerShell):
  export HF_TOKEN= (Replace with your token if needed) 
  python3 test_logic_task.py --num-examples 10 --batch-size 16 --local-model-dir "./model" --prompt-version 3 --model-id Qwen/Qwen3-14B

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
    "Example of equal and not equal comparisons:\n"
    " 'a' == 'a' → true\n"
    " 'a' == 'b' → false\n"
    " 'a' != 'b' → true\n"
    " 'x' != 'x' → false\n"
    "Logic function: \n\n"
    "def logic_function(t0, t1, t2, t3):\n"
    "    return (t0 != t1) or (t2 != t3) and (t0 == t3)\n\n"
    "Examples:\n"
    "logic_function(t0='a', t1='b', t2='c', t3='d') → true\n"
    "logic_function(t0='x', t1='x', t2='y', t3='y') → false\n"
    "logic_function(t0='r', t1='r', t2='g', t3='r') → true\n"
    "logic_function(t0='m', t1='n', t2='p', t3='p') → true\n"
    "logic_function(t0='a', t1='a', t2='b', t3='c') → false\n"
    "Now evaluate the same logic function with new inputs. Return ONLY 'true' or 'false' (lowercase, no punctuation).\n\n"
    "Please Evaluate: logic_function(t0='{t0}', t1='{t1}', t2='{t2}', t3='{t3}')="
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
    "logic_function1(True, True, False, True)=false\n"
    "logic_function1(False, True, True, True)=true\n"
    "logic_function1(True, False, True, False)=false\n"
    "Now we have a different logic function to evaluate. Return only 'true' or 'false' (lowercase, no punctuation). Don't use step by step reasoning or any additional text.\n\n"
    "def logic_function2(t0,t1,t2,t3):\n"
    "    return (t0 or t1) and t2 or t3\n\n"
    "Please evaluate: logic_function2({t0},{t1},{t2},{t3})="
)

# Integer comparison task: comparisons among inputs
# Formula: t0 >= t1 or t1 >= 50 and t3 >= t0
# Same as: (t0 >= t1) or ((t1 >= 50) and (t3 >= t0))
PROMPT_TEMPLATE4 = (
    "Task: Evaluate the logical expression with two-digit integer comparisons.\n\n"
    "Examples of comparisons:\n"
    "45 >= 32 = true\n"
    "12 >= 34 = false\n"
    "23 >= 55 = false\n"
    "80 >= 24 = true\n"
    "Now we introduce the following logic function.\n\n"
    "def check(t0,t1,t2,t3):\n"
    "    return t0 >= t1 or t1 >= 50 and t3 >= t0\n\n"
    "Examples:\n"
    "check(45,32,18,22)=true\n"
    "check(12,34,23,11)=false\n"
    "check(23,55,46,14)=false\n"
    "check(11,24,14,56)=true\n"
    "check(22,35,23,11)=false\n"
    "check(34,62,47,22)=false\n"
    "Now evaluate the following logical function using two-digit integers. Return only 'true' or 'false' (lowercase, no punctuation).\n\n"
    "check({t0},{t1},{t2},{t3})="
)

# Contains Letter task: Check if words contain the letter 'a'
# Formula: ('a' in w0) or ('a' in w1) and ('a' in w2) and ('a' in w3)
# Same as: ('a' in w0) or (('a' in w1) and ('a' in w2) and ('a' in w3))
# Input space: vocab_size^4 possible combinations
PROMPT_TEMPLATE5 = (
    "Task: Check if words contain the letter 'a'.\n\n"
    "def contains_a(w0,w1,w2,w3):\n"
    "    return 'a' in w0 or 'a' in w1 and 'a' in w2 and 'a' in w3\n\n"
    "Examples:\n"
    "contains_a(cat,dog,fish,bird)=true\n"
    "contains_a(dog,fish,bird,sun)=false\n"
    "contains_a(sun,apple,water,banana)=true\n"
    "contains_a(red,blue,green,pink)=false\n"
    "contains_a(tree,bat,map,pan)=true\n"
    "contains_a(cup,box,pen,dot)=false\n"
    "Return only 'true' or 'false' (lowercase, no punctuation).\n\n"
    "contains_a({w0},{w1},{w2},{w3})="
)


def build_single_token_ids(tokenizer) -> List[int]:
    """Return a list of token ids for sampling.

    Categories collected (all must be single-token encodings):
      - ASCII letters (a-z, A-Z) of length 1
      - ASCII digits (0-9) of length 1 

    Exclusions:
      - Tokens that decode to 'true' or 'false' (to avoid answer leakage)
      - Tokens containing whitespace or non-ascii characters
      - Tokens that include internal special token markers
    """
    vocab_size = getattr(tokenizer, "vocab_size", None)
    if vocab_size is None:
        vocab = tokenizer.get_vocab()
        vocab_size = max(vocab.values()) + 1

    valid_tokens: List[int] = []
    excluded_words = {"true", "false", "yes", "no"}

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

            # Only accept single character tokens
            if len(decoded) != 1:
                continue

            # Accept only single ASCII letters (a-z, A-Z) or single digits (0-9)
            if decoded.isalpha() or decoded.isdigit():
                valid_tokens.append(idx)
                
        except Exception:
            continue

    return valid_tokens

def generate_examples(single_ids: List[int], tokenizer, n: int):
    """Generate examples for template 1: (t0 != t1) or (t2 != t3) and (t0 == t3)
    
    Uses 4 tokens with 50% probability of equality for each pair.
    Python precedence: (t0 != t1) or ((t2 != t3) and (t0 == t3))
    """
    examples = []
    for _ in range(n):
        picks = random.sample(single_ids, 4)
        # decode each single id into a readable string
        tokens = [tokenizer.decode([p], clean_up_tokenization_spaces=False) for p in picks]
        t0, t1, t2, t3 = tokens
        
        # uniformly sample the input by potentially forcing equality
        if random.random() < 0.5:  # force t0 == t1 half the time
            t1 = t0
        if random.random() < 0.5:  # force t2 == t3 half the time  
            t3 = t2
        if random.random() < 0.5:  # force t0 == t3 half the time
            t3 = t0
        
        # Update tokens list with potentially modified values
        final_tokens = [t0, t1, t2, t3]
        
        # logical ground truth: (t0 != t1) or (t2 != t3) and (t0 == t3)
        # Python precedence: (t0 != t1) or ((t2 != t3) and (t0 == t3))
        gt = (t0 != t1) or (t2 != t3) and (t0 == t3)
        examples.append(({f"t{i}": final_tokens[i] for i in range(4)}, gt))
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

def generate_examples_int(n: int):
    """Generate examples for integer comparison task.
    Formula: t0 >= t1 or t1 >= 50 and t3 >= t0
    Same as: (t0 >= t1) or ((t1 >= 50) and (t3 >= t0))
    
    Each comparison has 50% probability of being true:
    - t0 >= t1: 50% true (by conditionally swapping)
    - t1 >= 50: 50% true (sample from [10,49] or [50,99])
    - t3 >= t0: 50% true (by conditionally swapping)
    """
    examples = []
    for _ in range(n):
        # Generate base values
        t0 = random.randint(10, 99)
        t2 = random.randint(10, 99)  # t2 is not used in logic but still passed
        
        # t1 >= 50: 50% chance true
        if random.random() < 0.5:
            t1 = random.randint(50, 99)  # t1 >= 50 is true
        else:
            t1 = random.randint(10, 49)  # t1 >= 50 is false
        
        # t0 >= t1: 50% chance true
        # Generate t0 relative to t1 to control this
        if random.random() < 0.5:
            # t0 >= t1 should be true
            t0 = random.randint(t1, 99)
        else:
            # t0 >= t1 should be false (t0 < t1)
            if t1 > 10:
                t0 = random.randint(10, t1 - 1)
            else:
                t0 = 10  # edge case: t1 = 10, so t0 < t1 is impossible with t0 >= 10
        
        # t3 >= t0: 50% chance true
        if random.random() < 0.5:
            # t3 >= t0 should be true
            t3 = random.randint(t0, 99)
        else:
            # t3 >= t0 should be false (t3 < t0)
            if t0 > 10:
                t3 = random.randint(10, t0 - 1)
            else:
                t3 = 10  # edge case
        
        # Ground truth: t0 >= t1 or t1 >= 50 and t3 >= t0
        gt = t0 >= t1 or t1 >= 50 and t3 >= t0
        
        tokens_map = {
            "t0": str(t0), "t1": str(t1), "t2": str(t2), "t3": str(t3)
        }
        examples.append((tokens_map, gt))
    
    return examples


def build_alpha_vocab(tokenizer) -> List[str]:
    """Build vocabulary of single-token words for alphabetical order task.
    
    Filters for:
      - Single-token encodings (word encodes to exactly 1 token)
      - Lowercase alphabetic words (2-10 characters)
      - ASCII only, no special characters
    
    Exclusions:
      - 'true', 'false', 'yes', 'no' (to avoid answer leakage)
      - Words with whitespace or non-alphabetic characters
    """
    vocab_size = getattr(tokenizer, "vocab_size", None)
    if vocab_size is None:
        vocab = tokenizer.get_vocab()
        vocab_size = max(vocab.values()) + 1

    valid_words: List[str] = []
    excluded_words = {"true", "false", "yes", "no"}

    for idx in range(vocab_size):
        try:
            token = tokenizer.convert_ids_to_tokens(idx)
            # Quick reject for special tokens containing brackets or angle markers
            if any(mark in token for mark in ["<", ">", "[", "]", "▁", "Ġ", "##"]):
                continue
            
            enc = tokenizer.encode(token, add_special_tokens=False)
            if len(enc) != 1:
                continue  # must correspond to a single token
            
            decoded = tokenizer.decode([idx], clean_up_tokenization_spaces=False).strip()
            if not decoded or not decoded.isascii():
                continue
            if any(ch.isspace() for ch in decoded):
                continue
            
            # Convert to lowercase for comparison
            word = decoded.lower()
            
            # Skip excluded words
            if word in excluded_words:
                continue
            
            # Accept only lowercase alphabetic words, length 2-10
            if not word.isalpha():
                continue
            if len(word) < 2 or len(word) > 10:
                continue
            
            # Only add if the lowercase version is the same (avoid duplicates)
            if decoded == word:
                valid_words.append(word)
                
        except Exception:
            continue

    # Remove duplicates and sort
    valid_words = sorted(list(set(valid_words)))
    return valid_words


def generate_examples_alpha(vocab: List[str], n: int):
    """Generate examples for contains letter task.
    4 words as input, checking if they contain the letter 'a'.
    Formula: ('a' in w0) or ('a' in w1) and ('a' in w2) and ('a' in w3)
    Same as: ('a' in w0) or (('a' in w1) and ('a' in w2) and ('a' in w3))
    
    This task has:
    - Large input space (vocab_size^4 combinations)
    - Each word is typically a single token
    - Simple check (contains letter 'a')
    - Causal structure with AND/OR logic
    - Each word has 50% probability of containing 'a'
    """
    # Split vocabulary into words with 'a' and words without 'a'
    words_with_a = [w for w in vocab if 'a' in w]
    words_without_a = [w for w in vocab if 'a' not in w]
    
    if len(words_with_a) == 0 or len(words_without_a) == 0:
        raise RuntimeError("Need words both with and without 'a' in vocabulary")
    
    examples = []
    for _ in range(n):
        # For each word position, 50% chance of containing 'a'
        words = []
        for _ in range(4):
            if random.random() < 0.5:
                words.append(random.choice(words_with_a))
            else:
                words.append(random.choice(words_without_a))
        
        w0, w1, w2, w3 = words
        
        # Ground truth: ('a' in w0) or ('a' in w1) and ('a' in w2) and ('a' in w3)
        # Python precedence: and binds tighter than or
        # So this is: ('a' in w0) or (('a' in w1) and ('a' in w2) and ('a' in w3))
        gt = ('a' in w0) or ('a' in w1) and ('a' in w2) and ('a' in w3)
        
        tokens_map = {
            "w0": w0, "w1": w1, "w2": w2, "w3": w3
        }
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
        dtype=torch.bfloat16,
        device_map={"": 0},  # Use only GPU 1 "auto"
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
    elif prompt_version == 4:
        prompt_template = PROMPT_TEMPLATE4
    elif prompt_version == 5:
        prompt_template = PROMPT_TEMPLATE5
    else:
        raise ValueError(f"Invalid prompt version: {prompt_version}. Must be 1, 2, 3, 4, or 5.")

    model, tokenizer = get_model_and_tokenizer(model_id, hf_token, local_model_dir)

    print("Building single-token list from tokenizer...")
    single_ids = build_single_token_ids(tokenizer)
    if len(single_ids) < 6:
        raise RuntimeError("Not enough single-token vocabulary items to build examples.")
    print(f"Found {len(single_ids)} single-token vocab entries.")

    # Build alpha vocabulary for prompt version 5
    alpha_vocab = None
    if prompt_version == 5:
        print("Building alphabetical word vocabulary from tokenizer...")
        alpha_vocab = build_alpha_vocab(tokenizer)
        if len(alpha_vocab) < 8:
            raise RuntimeError("Not enough single-token words to build alphabetical examples.")
        print(f"Found {len(alpha_vocab)} single-token words for alphabetical task.")

    print(f"Sampling {num_examples} test examples...")
    if prompt_version == 3:
        examples = generate_examples2(num_examples)
    elif prompt_version == 4:
        examples = generate_examples_int(num_examples)
    elif prompt_version == 5:
        examples = generate_examples_alpha(alpha_vocab, num_examples)
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
                {"role": "system", "content": "Provide your final answer as only the word true or false in lowercase."},
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
            max_new_tokens=1,
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
    parser.add_argument("--prompt-version", type=int, default=1, choices=[1, 2, 3, 4, 5], help="Prompt template version: 1=string comparison, 2=pattern inference, 3=boolean logic (16 inputs), 4=integer comparison (10^6 inputs), 5=alphabetical order (word pairs)")
    args = parser.parse_args()

    hf_token = args.hf_token or os.environ.get("HF_TOKEN")
    run(args.model_id, args.num_examples, hf_token, args.seed, args.batch_size, args.local_model_dir, args.prompt_version)


if __name__ == "__main__":
    main()
    # clear GPU memory
    torch.cuda.empty_cache()
