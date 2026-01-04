# Causal Abstraction & Distributed Alignment Search (DAS)

This folder contains tools for studying causal abstraction in language models using Distributed Alignment Search (DAS) and related techniques.

## Overview

The codebase implements:
- **Standard DAS**: Learns fixed-dimension orthogonal subspaces for causal intervention
- **Boundless DAS**: Automatically selects relevant feature dimensions via learnable masks
- **Vanilla Intervention**: Full activation patching without subspace projection
- **Logic Task Evaluation**: Test LLM reasoning on boolean logic expressions
- **Parallel Layer Training**: Train different layers on different GPUs simultaneously

## Files

| File | Description |
|------|-------------|
| `das.py` | Main script for DAS training and testing with multiple intervention types |
| `boundless_das.py` | Boundless DAS implementation with per-feature mask learning |
| `test_logic_task.py` | Standalone script to evaluate LLM accuracy on logic tasks |
| `util_data.py` | Dataset generation, causal model construction, and counterfactual sampling |
| `util_model.py` | Model loading utilities with multi-GPU support |
| `run_parallel_boundless.sh` | Script to launch parallel training across multiple GPUs |
| `merge_parallel_results.py` | Merge results from parallel training runs |

## Installation

```bash
pip install torch transformers pyvene tqdm numpy
```

## Logic Task

The primary task is evaluating the expression:
```python
(t0 != t1) or ((t2 != t3) and (t0 == t3))
```

**Causal structure:**
- `op1`: t0 != t1
- `op2`: t2 != t3  
- `op3`: t0 == t3
- `op4`: op2 and op3
- `op5`: op1 or op4 (final result)

## Usage

### 1. Test Model Accuracy on Logic Task

```bash
# Basic test
python test_logic_task.py --model-id Qwen/Qwen3-14B --num-examples 200

# With local model cache
python test_logic_task.py --model-id Qwen/Qwen3-14B --num-examples 200 \
    --local-model-dir ./model --batch-size 16

# Different prompt versions (1-5 available)
python test_logic_task.py --prompt-version 3  # Boolean logic
python test_logic_task.py --prompt-version 4  # Integer comparison
python test_logic_task.py --prompt-version 5  # Contains letter 'a'
```

### 2. Train DAS Interventions

#### Standard DAS (fixed subspace dimension)
```bash
# Single GPU
python das.py --train --intervention-type das \
    --hf-cache-dir /path/to/.hf_cache \
    --batch-size 8 \
    --subspace-dimension 1

# Multi-GPU
python das.py --train --intervention-type das \
    --hf-cache-dir /path/to/.hf_cache \
    --batch-size 32 --num-gpus 8
```

#### Boundless DAS (automatic feature selection)
```bash
# Single GPU
python das.py --train --intervention-type boundless \
    --hf-cache-dir /path/to/.hf_cache \
    --batch-size 8 \
    --sparsity-coef 0.01

# Multi-GPU with custom temperature annealing
python das.py --train --intervention-type boundless \
    --hf-cache-dir /path/to/.hf_cache \
    --batch-size 32 --num-gpus 8 \
    --temperature-start 1.0 --temperature-end 0.01
```

#### Vanilla Intervention (full activation patching)
```bash
python das.py --train --intervention-type vanilla \
    --hf-cache-dir /path/to/.hf_cache \
    --batch-size 8
```

### 3. Parallel Training (Multi-GPU Layer Parallelism)

Since different layers are independent, you can train them in parallel on different GPUs for **6x speedup**:

#### Option A: Use the provided script
```bash
# Launch parallel training (each GPU handles ~7 layers)
./run_parallel_boundless.sh

# Monitor progress
tail -f logs/gpu_*.log

# After all complete, merge results
python merge_parallel_results.py --intervention op1
```

#### Option B: Manual parallel launch
```bash
# GPU 0: layers 0-9
python das.py --train --intervention-type boundless \
    --gpu-id 0 --layer-start 0 --layer-end 10 \
    --hf-cache-dir /path/to/.hf_cache --batch-size 128

# GPU 1: layers 10-19 (run in another terminal)
python das.py --train --intervention-type boundless \
    --gpu-id 1 --layer-start 10 --layer-end 20 \
    --hf-cache-dir /path/to/.hf_cache --batch-size 128

# ... continue for other GPUs
```

#### Option C: Slurm job array
```bash
for gpu in 0 1 2 3 4 5; do
  layer_start=$((gpu * 7))
  layer_end=$(((gpu + 1) * 7))
  if [ $layer_end -gt 40 ]; then layer_end=40; fi
  
  nlprun -g 1 -a nlp -p sphinx \
    "cd /path/to/Pretrain_models && \
     source ~/.bashrc && conda activate multi-hyp && \
     python das.py --train --intervention-type boundless \
       --gpu-id $gpu --layer-start $layer_start --layer-end $layer_end \
       --hf-cache-dir /path/to/.hf_cache --batch-size 128"
done
```

### 4. Test Pre-trained Interventions

```bash
# Test with saved weights
python das.py --test --intervention-type das \
    --weights-path training_results/das_weights_das_or_model_1_dim1.pt \
    --candidates-path training_results/candidates_das_or_model_1_dim1.json

# Test boundless DAS
python das.py --test --intervention-type boundless \
    --weights-path training_results/das_weights_boundless_or_model_1.pt \
    --candidates-path training_results/candidates_boundless_or_model_1.json
```

## Command-Line Arguments

### das.py

| Argument | Default | Description |
|----------|---------|-------------|
| `--train` / `--test` | Required | Training or testing mode |
| `--model-id` | `Qwen/Qwen3-14B` | HuggingFace model ID |
| `--intervention-type` | `das` | Type: `das`, `vanilla`, or `boundless` |
| `--causal-model` | `1` | Causal model variant (1 or 2) |
| `--subspace-dimension` | `1` | Subspace dimension for standard DAS |
| `--sparsity-coef` | `0.01` | L1 sparsity coefficient for boundless DAS |
| `--temperature-start` | `1.0` | Starting temperature for mask annealing |
| `--temperature-end` | `0.01` | Ending temperature for mask annealing |
| `--batch-size` | `32` | Batch size |
| `--data-size` | `1024` | Number of examples per dataset |
| `--num-gpus` | `1` | Number of GPUs (-1 for auto, 0 for CPU) |
| `--gpu-id` | None | Specific GPU ID for layer parallelism |
| `--layer-start` | None | Start layer index (inclusive) for parallel training |
| `--layer-end` | None | End layer index (exclusive) for parallel training |
| `--seed` | `42` | Random seed |
| `--hf-cache-dir` | None | HuggingFace cache directory |

### test_logic_task.py

| Argument | Default | Description |
|----------|---------|-------------|
| `--model-id` | `meta-llama/Llama-3.1-8B-Instruct` | HuggingFace model ID |
| `--num-examples` | `200` | Number of test examples |
| `--prompt-version` | `1` | Prompt template (1-5) |
| `--batch-size` | `16` | Batch size for generation |
| `--seed` | `42` | Random seed |
| `--local-model-dir` | None | Local model cache directory |

### merge_parallel_results.py

| Argument | Default | Description |
|----------|---------|-------------|
| `--intervention` | `op1` | Intervention name to merge |
| `--all-interventions` | False | Merge all interventions (op1-op5) |
| `--results-dir` | `results` | Directory containing partial results |
| `--output-suffix` | `_merged` | Suffix for merged output files |

## Output Files

### Training
- `training_results/candidates_{type}_{model}_dim{d}.json` - Accuracy scores per (layer, position)
- `training_results/das_weights_{type}_{model}_dim{d}.pt` - Trained intervention weights
- `training_results/feature_counts_boundless_{model}.json` - Selected feature counts (boundless only)
- `training_results/position_token_mapping_{op}_{model}.json` - Token position mappings

### Parallel Training (Boundless DAS)
- `results/candidates_boundless_partial_{op}_L{start}-{end}.json` - Partial results per GPU
- `results/weights_boundless_partial_{op}_L{start}-{end}.pt` - Partial weights per GPU
- `results/feature_counts_partial_{op}_L{start}-{end}.json` - Partial feature counts per GPU
- `logs/gpu_{id}_L{start}-{end}.log` - Training logs per GPU

### After Merging Parallel Results
- `results/candidates_boundless_{op}_merged.json` - Merged accuracy scores
- `results/weights_boundless_{op}_merged.pt` - Merged weights
- `results/feature_counts_{op}_merged.json` - Merged feature counts

### Testing
- `test_results/test_results_{type}_{model}.json` - Test accuracy per candidate
- `test_results/analysis_datasets_{type}_{model}.json` - Per-sample features and correctness

## Intervention Types Comparison

| Type | Description | Parameters | Use Case |
|------|-------------|------------|----------|
| **Standard DAS** | Fixed-dimension orthogonal rotation | Subspace dimension (e.g., 1D) | When you expect a low-dimensional causal variable |
| **Boundless DAS** | Learnable per-feature masks with sparsity | Sparsity coefficient, temperature | When the relevant subspace dimension is unknown |
| **Vanilla** | Full activation interchange | None | Baseline comparison, upper bound on intervention effect |

## Causal Models

### Model 1 (default)
Task: `(t0 != t1) or ((t2 != t3) and (t0 == t3))`
- 4 input tokens: t0, t1, t2, t3
- 5 intermediate operations: op1-op5

### Model 2
Task: `((t2 != t4) or (t1 == t3)) and ((t0 != t5) or (t1 == t3))`
- 6 input tokens: t0-t5
- 6 operations: op1a-op6a

## Tips

1. **Memory**: Use `--num-gpus` for model parallelism with large models
2. **Speed**: Increase `--batch-size` when GPU memory allows
3. **Reproducibility**: Set `--seed` for consistent results
4. **Boundless DAS**: Tune `--sparsity-coef` (higher = sparser feature selection)
5. **Cache**: Use `--hf-cache-dir` to avoid re-downloading models
6. **Parallel Training**: Use `--gpu-id`, `--layer-start`, `--layer-end` to train different layers on different GPUs simultaneously (6x speedup with 6 GPUs)
7. **Layer Independence**: Different layers are completely independent during DAS training, making layer parallelism perfectly efficient

## References

- [pyvene](https://github.com/stanfordnlp/pyvene) - Neural Network Intervention Library
- [Causal Abstraction](https://arxiv.org/abs/2301.04709) - Theoretical Framework
- [Distributed Alignment Search](https://arxiv.org/abs/2303.02536) - DAS Method
