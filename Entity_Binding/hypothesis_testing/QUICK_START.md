# Quick Start Guide

## Prerequisites

Make sure you have the `causalab` package accessible. The script automatically adds it to the Python path.

## Running the Script

### Basic Usage

```bash
# Run with GPT-2 on GPU 0
python run_hypothesis_test.py --model gpt2 --gpu 0

# Test mode (smaller dataset, faster)
python run_hypothesis_test.py --model gpt2 --gpu 0 --test

# Custom dataset size
python run_hypothesis_test.py --model gpt2 --gpu 0 --size 256
```

### What the Script Does

1. **Creates Dataset**: Generates 128 (or custom size) entity binding examples with 10 groups and 3 entities per group
2. **Filters Dataset**: Keeps only examples where the model correctly predicts both base and counterfactual inputs
3. **Checks Accuracy**: Ensures >80% accuracy, otherwise suggests using a larger model
4. **Runs Boundless DAS**: Trains feature alignments for `positional_query_group` across all layers
5. **Saves Results**: Saves weights, IIA accuracies, and visualizations

### Expected Runtime

- **Test mode** (`--test`): ~5-10 minutes (16 examples, smaller batches)
- **Full mode**: ~30-60 minutes depending on model size (128 examples, full training)

### Output Location

Results are saved to: `hypothesis_testing/outputs/{model_name}/`

Key files:
- `summary.json`: Quick summary of results
- `boundless_das_result.pkl`: Full result object
- `boundless_das/heatmaps/`: Visualization images

## Understanding the Results

### IIA Accuracy (Interchange Intervention Accuracy)

- **Range**: 0.0 to 1.0
- **1.0**: Perfect alignment - learned features perfectly substitute the causal variable
- **0.0**: No alignment - learned features cannot substitute the causal variable
- **>0.7**: Good alignment
- **<0.3**: Poor alignment, may indicate the variable is not represented clearly at this layer

### Best Layer

The script identifies which layer has the highest IIA accuracy for `positional_query_group`. This tells you where in the model this variable is most clearly represented.

### Feature Counts

The heatmap shows how many features (out of 32 by default) are selected at each layer. More features may indicate:
- More complex representations
- Less precise alignment
- Need for more features to capture the variable

## Common Issues

### Low Accuracy Error

If you get an error about accuracy being below 80%:

```
⚠ WARNING: Task accuracy (XX%) is below threshold (80%)
```

**Solution**: Use a larger model:
```bash
python run_hypothesis_test.py --model gpt2-medium --gpu 0
```

### Out of Memory

**Solutions**:
- Reduce batch size: `--batch-size 16`
- Use test mode: `--test`
- Reduce features: `--n-features 16`

### Model Not Found

Make sure the model ID is correct:
- `gpt2` - ✅ Works
- `gpt2-medium` - ✅ Works  
- `meta-llama/Llama-3.1-8B-Instruct` - ⚠️ Requires authentication

## Next Steps

1. Check `summary.json` for quick results overview
2. View heatmaps in `boundless_das/heatmaps/` directory
3. Load `boundless_das_result.pkl` for detailed analysis:

```python
import pickle
with open('hypothesis_testing/outputs/gpt2/boundless_das_result.pkl', 'rb') as f:
    result = pickle.load(f)

# Access test scores by layer
print(result['test_scores'])

# Access best layer
print(result['metadata']['best_layer'])
```

## Tips

- Start with `--test` mode to verify everything works
- Use `--size 64` for faster iteration during development
- Check GPU memory usage with `nvidia-smi`
- Larger models (e.g., `gpt2-medium`) may give better accuracy but require more memory

