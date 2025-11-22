# Llama 8B local test

This folder contains a script to download a Llama-8B-style model from Hugging Face, load it locally (on an A100 with 8-bit/bitsandbytes if available), run a synthetic logical token-equality task, and report accuracy.

Quick steps (PowerShell):

1. Install dependencies (recommended in a virtual env):

```powershell
python -m pip install -r Llama8b/requirements.txt
```

2. Set your Hugging Face token if the model requires authentication:

```powershell
$env:HF_TOKEN = "<your_hf_token>"
```

3. Run the test script (example using Llama-2 8B chat model as a placeholder):

```powershell
python Llama8b/run_test.py --model-id "meta-llama/Llama-2-8b-chat-hf" --num-examples 200
```

Notes:
- The default `--model-id` is `meta-llama/Llama-2-8b-chat-hf`. Replace with the exact Llama-8B instruct repo id you want.
- The script tries to load the model in 8-bit mode using `bitsandbytes`. If that fails it falls back to fp16.
- The logical task used by the script is: ((t2 != t4) and (t0 != t5)) or (t1 == t3). Tokens t0..t5 are sampled from the model vocabulary (single-token entries).
- Results will be printed to stdout including accuracy.

If you'd like, I can:
- Change the logical expression to match a different spec
- Add saving of the raw predictions and examples to JSON
- Add a small unit test harness or batching for faster inference
