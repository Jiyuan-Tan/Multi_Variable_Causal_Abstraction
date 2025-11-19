#!/usr/bin/env python3
"""Download a Llama-8B-style model from Hugging Face, run a logical token task, and report accuracy.

Usage examples (PowerShell):
  $env:HF_TOKEN="<your_hf_token>"; python .\Llama8b\run_test.py --model-id "meta-llama/Llama-2-8b-chat-hf" --num-examples 200

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


PROMPT_TEMPLATE = (
    "Task: given variables t0..t5, compute output = ((t2 != t4) and (t0 != t5)) or (t1 == t3).\n"
    "Respond ONLY with true or false (lowercase, no punctuation).\n"
    "t0={t0}, t1={t1}, t2={t2}, t3={t3}, t4={t4}, t5={t5}. output?"
)


def build_single_token_ids(tokenizer) -> List[int]:
    """Return list of token ids that correspond to a single tokenizer token when encoded alone."""
    vocab_size = getattr(tokenizer, "vocab_size", None)
    if vocab_size is None:
        # fall back to length of vocab mapping
        vocab = tokenizer.get_vocab()
        vocab_size = max(vocab.values()) + 1

    single = []
    for idx in range(vocab_size):
        try:
            token = tokenizer.convert_ids_to_tokens(idx)
            enc = tokenizer.encode(token, add_special_tokens=False)
            if len(enc) == 1 and token.strip() != "":
                single.append(idx)
        except Exception:
            continue
    return single


def generate_examples(single_ids: List[int], tokenizer, n: int):
    examples = []
    for _ in range(n):
        picks = random.sample(single_ids, 6)
        # decode each single id into a readable string
        tokens = [tokenizer.decode([p], clean_up_tokenization_spaces=False) for p in picks]
        t0, t1, t2, t3, t4, t5 = tokens
        # logical ground truth: ((t2 != t4) and (t0 != t5)) or (t1 == t3)
        # uniformaly sample the input
        t2 = t4 if random.random() < 0.5 else t2  # force equality half the time
        t0 = t5 if random.random() < 0.5 else t0  # force equality half the time
        t1 = t3 if random.random() < 0.5 else t1  # force equality half the time
        gt = ((t2 != t4) and (t0 != t5)) or (t1 == t3)
        examples.append(({f"t{i}": tokens[i] for i in range(6)}, gt))
    return examples


def parse_bool_from_text(text: str):
    import re
    s = text.lower()
    # Extract standalone boolean tokens (avoid substrings like 'untrue')
    candidates = re.findall(r"\b(true|false|yes|no)\b", s)
    if not candidates:
        return None
    # Prefer the FIRST token to minimize contamination from chain-of-thought that later includes both
    token = candidates[0]
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
        device_map="auto",
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


def run(model_id: str, num_examples: int, hf_token: str | None, seed: int, batch_size: int, local_model_dir: str | None):
    random.seed(seed)
    torch.manual_seed(seed)

    model, tokenizer = get_model_and_tokenizer(model_id, hf_token, local_model_dir)

    print("Building single-token list from tokenizer...")
    single_ids = build_single_token_ids(tokenizer)
    if len(single_ids) < 6:
        raise RuntimeError("Not enough single-token vocabulary items to build examples.")
    print(f"Found {len(single_ids)} single-token vocab entries.")

    print(f"Sampling {num_examples} test examples...")
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
            user_prompt = PROMPT_TEMPLATE.format(**tokens_map)
            messages = [
                {"role": "system", "content": "You solve logic tasks. Output only the word true or false in lowercase with no explanation."},
                {"role": "user", "content": user_prompt},
            ]
            if hasattr(tokenizer, "apply_chat_template"):
                prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            else:
                # Fallback: simple concatenation
                prompt_text = "\n".join([m["content"] for m in messages]) + "\n"
            prompt_texts.append(prompt_text)

        enc = tokenizer(prompt_texts, return_tensors="pt", padding=True)
        enc = {k: v.to(model.device) for k, v in enc.items()}
        input_lengths = [l.item() for l in (enc["attention_mask"].sum(dim=1))]

        gen_out = model.generate(
            **enc,
            max_new_tokens=10,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
        )

        for (tokens_map, gt), seq, in_len in zip(batch, gen_out, input_lengths):
            # Slice generated portion
            gen_tokens = seq[in_len:]
            text_val = tokenizer.decode(gen_tokens, skip_special_tokens=True)
            pred = parse_bool_from_text(text_val)
            is_correct = (pred is not None) and (pred == gt)
            if is_correct:
                correct += 1
            total += 1

            i = total
            if i % max(1, num_examples // 10) == 0 or i <= 5:
                print(f"[{i}/{num_examples}] GT={gt}  PRED={pred}  -> {'OK' if is_correct else 'ERR'}")
            if not is_correct and i <= 10:
                print(f"   Raw generation: {text_val!r}")

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
    args = parser.parse_args()

    hf_token = args.hf_token
    run(args.model_id, args.num_examples, hf_token, args.seed, args.batch_size, args.local_model_dir)


if __name__ == "__main__":
    main()
