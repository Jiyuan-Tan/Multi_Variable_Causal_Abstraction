"""
Boundless DAS (Distributed Alignment Search) Implementation

This module provides boundless DAS functionality that learns per-feature masks
instead of fixed-dimension subspaces. Unlike standard DAS which uses a fixed 
subspace dimension, boundless DAS automatically selects relevant feature dimensions
through learnable masks with temperature annealing.

Key differences from standard DAS:
- Standard DAS: Learns a fixed-dimension orthogonal rotation (e.g., 1D subspace)
- Boundless DAS: Learns importance masks for each feature dimension

Usage:
    python das.py --train --intervention-type boundless --hf-cache-dir <cache>
"""

import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm, trange
from torch.utils.data import DataLoader
from pyvene import (
    IntervenableModel,
    RepresentationConfig,
    IntervenableConfig,
)
import pyvene as pv
from pyvene.models import modeling_utils

# Monkey patch pyvene's gather_neurons to fix multi-GPU device mismatch
# The issue: pyvene calls gather_neurons with device=self.get_device() which may
# differ from tensor_input.device when using device_map="auto" on multi-GPU
_original_gather_neurons = modeling_utils.gather_neurons

def _patched_gather_neurons(tensor_input, unit, unit_locations_as_list, device=None):
    """
    Patched version of gather_neurons that always uses tensor_input.device.
    This fixes the multi-GPU device mismatch issue where the caller passes
    device=cuda:0 but tensor_input is on cuda:1.
    """
    # ALWAYS use tensor_input.device, ignore the passed device parameter
    # This ensures torch.gather works correctly on multi-GPU setups
    return _original_gather_neurons(tensor_input, unit, unit_locations_as_list, device=None)

# Apply the monkey patch
modeling_utils.gather_neurons = _patched_gather_neurons


class BoundlessRotatedSpaceIntervention(pv.TrainableIntervention, pv.DistributedRepresentationIntervention):
    """
    Boundless DAS intervention that learns per-feature masks.
    
    Instead of learning a fixed-dimension rotation subspace, this intervention
    learns a mask over all features, allowing the model to select which features
    are relevant for the intervention.
    
    The mask is trained with temperature annealing and L1 sparsity regularization
    to encourage sparse feature selection.
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # self.embed_dim is set by the parent class TrainableIntervention
        embed_dim = self.embed_dim
        
        # Learnable mask weights (one per feature dimension)
        # Initialize to zeros so sigmoid starts at 0.5
        self.mask_weights = nn.Parameter(torch.zeros(embed_dim))
        
        # Temperature for sigmoid (annealed during training)
        self.register_buffer('temperature', torch.tensor(1.0))
        
        # Also keep a learnable rotation layer for the selected subspace
        # This allows learning the optimal direction within the mask
        self.rotate_layer = pv.models.layers.LowRankRotateLayer(embed_dim, embed_dim, init_orth=True)
        self.rotate_layer = torch.nn.utils.parametrizations.orthogonal(self.rotate_layer)
        
    def set_temperature(self, temp: float):
        """Set the temperature for sigmoid mask computation."""
        self.temperature.fill_(temp)
    
    def get_mask(self, training: bool = True) -> torch.Tensor:
        """
        Compute the feature mask.
        
        During training: soft mask via sigmoid with temperature
        During inference: hard binary mask via threshold at 0.5
        """
        if training:
            return torch.sigmoid(self.mask_weights / self.temperature)
        else:
            return (torch.sigmoid(self.mask_weights) > 0.5).float()
    
    def get_sparsity_loss(self) -> torch.Tensor:
        """Compute L1 sparsity regularization loss on the mask."""
        mask = self.get_mask(training=True)
        return torch.norm(mask, p=1)
    
    def get_selected_features(self) -> list:
        """Return indices of features selected by the mask (sigmoid > 0.5)."""
        with torch.no_grad():
            binary_mask = (torch.sigmoid(self.mask_weights) > 0.5).float()
            return torch.nonzero(binary_mask).squeeze(-1).tolist()
    
    def count_selected_features(self) -> int:
        """Return count of selected features."""
        selected = self.get_selected_features()
        if isinstance(selected, int):
            return 1
        return len(selected) if selected else 0
    
    def forward(self, base, source, subspaces=None):
        """
        Forward pass: interchange features between base and source 
        weighted by the learned mask.
        
        Args:
            base: Base activations to be modified
            source: Source activations to take from
            subspaces: Not used (kept for API compatibility)
            
        Returns:
            Modified base activations with masked features from source
        """
        # Get the mask (soft during training, hard during inference)
        mask = self.get_mask(training=self.training)
        
        # Ensure mask is on the same device and dtype
        mask = mask.to(base.device).to(base.dtype)
        
        # Apply rotation to both base and source
        rotated_base = torch.matmul(base.to(self.rotate_layer.weight.dtype), self.rotate_layer.weight)
        rotated_source = torch.matmul(source.to(self.rotate_layer.weight.dtype), self.rotate_layer.weight)
        
        # Interchange: blend rotated features based on mask
        # mask=1 means take from source, mask=0 means keep from base
        blended = (1 - mask) * rotated_base + mask * rotated_source.to(rotated_base.dtype)
        
        # Rotate back to original space
        output = torch.matmul(blended, self.rotate_layer.weight.T)
        
        return output.to(base.dtype)


def config_boundless_das(model, layer, device, weight=None, embed_dim=None):
    """
    Configure an IntervenableModel with BoundlessRotatedSpaceIntervention.
    
    Args:
        model: The language model
        layer: Which layer to intervene on
        device: Device to place the intervention on
        weight: Optional pretrained weights (state_dict)
        embed_dim: Embedding dimension (inferred from model if not provided)
        
    Returns:
        IntervenableModel configured with boundless DAS intervention
    """
    if embed_dim is None:
        embed_dim = getattr(model.config, 'hidden_size', getattr(model.config, 'n_embd', None))
        if embed_dim is None:
            raise ValueError("Could not determine embed_dim from model config")
    
    # Use the same pattern as config_das in das.py
    config = IntervenableConfig(
        model_type=type(model),
        representations=[
            RepresentationConfig(
                layer,              # layer
                "block_output",     # component
                "pos",              # intervention unit
                1,                  # max number of unit
            ),
        ],
        intervention_types=BoundlessRotatedSpaceIntervention,
    )
    
    intervenable = IntervenableModel(config, model)
    
    if weight is not None:
        # Load pretrained weights
        intervention_key = f"layer_{layer}_comp_block_output_unit_pos_nunit_1#0"
        intervenable.interventions[intervention_key].load_state_dict(weight)
    
    # Check if model uses device_map (distributed across GPUs)
    # If so, don't call set_device as it conflicts with accelerate hooks
    is_distributed = hasattr(model, 'hf_device_map') and model.hf_device_map is not None
    if not is_distributed:
        intervenable.set_device(device)
    else:
        # For distributed models, move intervention parameters to the LAYER's device
        # (not input device, since gather_neurons needs tensors on same device as layer output)
        try:
            layer_device = get_layer_device(model, layer)
            for k, v in intervenable.interventions.items():
                v.to(layer_device)
        except Exception as e:
            # Fallback to input device
            try:
                input_device = next(model.get_input_embeddings().parameters()).device
                for k, v in intervenable.interventions.items():
                    v.to(input_device)
            except Exception:
                pass  # If we can't determine device, pyvene will handle it
    
    intervenable.disable_model_gradients()
    return intervenable


def compute_loss(outputs, labels):
    """Cross entropy loss for the predictions."""
    CE = torch.nn.CrossEntropyLoss()
    return CE(outputs, labels)


def get_layer_device(model, layer):
    """
    Get the device where a specific layer is located.
    This is crucial for multi-GPU models with device_map="auto".
    
    Args:
        model: The language model
        layer: Layer index
        
    Returns:
        torch.device for the specified layer
    """
    # Check hf_device_map for layer location
    if hasattr(model, 'hf_device_map') and model.hf_device_map:
        device_map = model.hf_device_map
        # Look for the layer in the device map
        # Common patterns: "model.layers.X", "transformer.h.X", "gpt_neox.layers.X"
        for key, dev in device_map.items():
            # Match layer patterns like "model.layers.27" 
            if f'.layers.{layer}' in key or f'.h.{layer}' in key:
                if isinstance(dev, int):
                    return torch.device(f"cuda:{dev}")
                elif isinstance(dev, str):
                    return torch.device(dev)
        
        # If not found directly, infer from surrounding layers
        # Sort layers to find which device this layer should be on
        layer_devices = {}
        for key, dev in device_map.items():
            import re
            match = re.search(r'\.layers\.(\d+)|\.h\.(\d+)', key)
            if match:
                layer_idx = int(match.group(1) or match.group(2))
                if isinstance(dev, int):
                    layer_devices[layer_idx] = torch.device(f"cuda:{dev}")
                elif isinstance(dev, str):
                    layer_devices[layer_idx] = torch.device(dev)
        
        if layer_devices:
            # Find the closest layer that we know about
            known_layers = sorted(layer_devices.keys())
            for known_layer in reversed(known_layers):
                if layer >= known_layer:
                    return layer_devices[known_layer]
            # If layer is before all known layers, use the first one
            return layer_devices[known_layers[0]]
    
    # Fallback: try to access the layer directly and get its device
    try:
        # Try common model structures
        if hasattr(model, 'model') and hasattr(model.model, 'layers'):
            layer_module = model.model.layers[layer]
        elif hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
            layer_module = model.transformer.h[layer]
        elif hasattr(model, 'gpt_neox') and hasattr(model.gpt_neox, 'layers'):
            layer_module = model.gpt_neox.layers[layer]
        else:
            raise AttributeError("Unknown model structure")
        
        return next(layer_module.parameters()).device
    except (IndexError, StopIteration, AttributeError):
        pass
    
    # Ultimate fallback
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def get_model_input_device(model):
    """Get the device where model inputs should be sent."""
    try:
        embed = model.get_input_embeddings()
        if embed is not None:
            return next(embed.parameters()).device
    except (StopIteration, AttributeError):
        pass
    
    if hasattr(model, 'hf_device_map') and model.hf_device_map:
        for key, dev in model.hf_device_map.items():
            if 'embed' in key.lower():
                return torch.device(f"cuda:{dev}" if isinstance(dev, int) else dev)
        first_device = next(iter(model.hf_device_map.values()))
        return torch.device(f"cuda:{first_device}" if isinstance(first_device, int) else first_device)
    
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def boundless_das_training(
    intervenable,
    train_dataset,
    optimizer,
    pos,
    device,
    tokenizer=None,
    epochs=10,
    batch_size=64,
    gradient_accumulation_steps=1,
    sparsity_coef=0.01,
    temperature_start=1.0,
    temperature_end=0.01,
    temperature_annealing_fraction=0.5,
):
    """
    Train the model with boundless DAS intervention.
    
    This function implements temperature annealing for the mask and L1 sparsity
    regularization to encourage sparse feature selection.
    
    Args:
        intervenable: IntervenableModel with BoundlessRotatedSpaceIntervention
        train_dataset: Training dataset with input_ids, source_input_ids, labels
        optimizer: Optimizer for intervention parameters
        pos: Position of intervention in the sequence
        device: Device to run on
        tokenizer: Tokenizer (optional, for debugging)
        epochs: Number of training epochs
        batch_size: Batch size for training
        gradient_accumulation_steps: Steps to accumulate gradients
        sparsity_coef: Coefficient for L1 sparsity loss
        temperature_start: Starting temperature for sigmoid
        temperature_end: Ending temperature (lower = more discrete)
        temperature_annealing_fraction: Fraction of steps to anneal temperature
    """
    from das import batched_random_sampler
    
    # For distributed models, get the correct input device
    input_device = get_model_input_device(intervenable.model)
    
    intervenable.model.train()
    print("Boundless DAS intervention trainable parameters:", intervenable.count_parameters())
    
    # Calculate total training steps
    total_steps = epochs * (len(train_dataset) // batch_size)
    annealing_steps = int(total_steps * temperature_annealing_fraction)
    
    # Create temperature schedule
    annealing_schedule = np.linspace(temperature_start, temperature_end, annealing_steps + 1)
    constant_schedule = np.full(total_steps - annealing_steps, temperature_end)
    temperature_schedule = np.concatenate([annealing_schedule, constant_schedule])
    
    train_iterator = trange(0, int(epochs), desc="Epoch")
    global_step = 0
    
    for epoch in train_iterator:
        epoch_correct = 0
        epoch_total = 0
        epoch_loss_sum = 0.0
        epoch_sparsity_sum = 0.0
        
        epoch_iterator = tqdm(
            DataLoader(
                train_dataset,
                batch_size=batch_size,
                sampler=batched_random_sampler(train_dataset, batch_size),
            ),
            desc=f"Epoch: {epoch}",
            position=0,
            leave=False,
            dynamic_ncols=True,
        )
        
        for batch in epoch_iterator:
            batch["input_ids"] = batch["input_ids"].squeeze(1).squeeze(1)
            batch["source_input_ids"] = batch["source_input_ids"].squeeze(1).squeeze(1)
            current_batch_size = batch["input_ids"].shape[0]
            
            # Use input_device for distributed models, otherwise use device
            target_device = input_device if input_device else device
            for k, v in batch.items():
                if v is not None and isinstance(v, torch.Tensor):
                    batch[k] = v.to(target_device)
            
            # Update temperature for this step
            current_temp = temperature_schedule[min(global_step, len(temperature_schedule) - 1)]
            for k, v in intervenable.interventions.items():
                v.set_temperature(current_temp)
            
            # Run intervention
            _, counterfactual_outputs = intervenable(
                {"input_ids": batch["input_ids"]},
                [{"input_ids": batch["source_input_ids"]}],
                {
                    "sources->base": (
                        [[[pos]] * current_batch_size],
                        [[[pos]] * current_batch_size],
                    )
                },
            )
            
            # Compute predictions and labels
            preds = counterfactual_outputs.logits[:, -1, :].argmax(dim=-1)
            labels = batch["labels"].squeeze()
            
            # Compute cross-entropy loss
            ce_loss = compute_loss(counterfactual_outputs.logits[:, -1, :], labels)
            
            # Add sparsity regularization
            sparsity_loss = 0.0
            for k, v in intervenable.interventions.items():
                sparsity_loss = sparsity_loss + v.get_sparsity_loss()
            
            total_loss = ce_loss + sparsity_coef * sparsity_loss
            
            # Accumulate epoch stats
            epoch_correct += (preds == labels).sum().item()
            epoch_total += current_batch_size
            epoch_loss_sum += ce_loss.item() * current_batch_size
            epoch_sparsity_sum += sparsity_loss.item() if isinstance(sparsity_loss, torch.Tensor) else sparsity_loss
            
            if gradient_accumulation_steps > 1:
                total_loss = total_loss / gradient_accumulation_steps
            total_loss.backward()
            global_step += 1
            
            # Step optimizer after accumulating gradients
            if global_step % gradient_accumulation_steps == 0:
                optimizer.step()
                intervenable.set_zero_grad()
        
        # Handle remaining gradients
        if global_step % gradient_accumulation_steps != 0:
            optimizer.step()
            intervenable.set_zero_grad()
        
        # Count selected features
        n_selected = 0
        for k, v in intervenable.interventions.items():
            n_selected += v.count_selected_features()
        
        train_iterator.set_postfix(
            loss=epoch_loss_sum / epoch_total if epoch_total > 0 else 0,
            accuracy=epoch_correct / epoch_total if epoch_total > 0 else 0,
            n_features=n_selected,
            temp=current_temp,
        )
        epoch_iterator.close()


def boundless_das_test(intervenable, pos, test_dataset, device, batch_size=64, return_details=False):
    """
    Test the model with boundless DAS intervention.
    
    Args:
        intervenable: IntervenableModel with trained intervention
        pos: Position of intervention
        test_dataset: Test dataset
        device: Device to run on
        batch_size: Batch size for evaluation
        return_details: If True, return per-sample details
        
    Returns:
        Accuracy (and details dict if return_details=True)
    """
    from das import batched_random_sampler
    
    # For distributed models, get the correct input device
    input_device = get_model_input_device(intervenable.model)
    target_device = input_device if input_device else device
    
    eval_labels = []
    eval_preds = []
    sample_indices = []
    
    # Set to eval mode (uses hard threshold for mask)
    for k, v in intervenable.interventions.items():
        v.eval()
    
    with torch.no_grad():
        if return_details:
            data_loader = DataLoader(
                test_dataset,
                batch_size=batch_size,
                shuffle=False,
            )
        else:
            data_loader = DataLoader(
                test_dataset,
                batch_size=batch_size,
                sampler=batched_random_sampler(test_dataset, batch_size),
            )
        
        epoch_iterator = tqdm(
            data_loader,
            desc="Testing",
            position=0,
            leave=False,
        )
        
        batch_idx = 0
        for batch in epoch_iterator:
            batch["input_ids"] = batch["input_ids"].squeeze(1).squeeze(1)
            batch["source_input_ids"] = batch["source_input_ids"].squeeze(1).squeeze(1)
            current_batch_size = batch["input_ids"].shape[0]
            
            for k, v in batch.items():
                if v is not None and isinstance(v, torch.Tensor):
                    batch[k] = v.to(target_device)
            
            _, counterfactual_outputs = intervenable(
                {"input_ids": batch["input_ids"]},
                [{"input_ids": batch["source_input_ids"]}],
                {
                    "sources->base": (
                        [[[pos]] * current_batch_size],
                        [[[pos]] * current_batch_size],
                    )
                },
            )
            
            eval_labels += [batch["labels"].squeeze()]
            eval_preds += [counterfactual_outputs.logits[:, -1, :].argmax(dim=-1)]
            
            if return_details:
                if "idx" in batch:
                    sample_indices.extend(batch["idx"].tolist())
                else:
                    start_idx = batch_idx * batch_size
                    sample_indices.extend(range(start_idx, start_idx + current_batch_size))
            batch_idx += 1
    
    eval_labels = torch.cat(eval_labels)
    eval_preds = torch.cat(eval_preds)
    acc = sum(eval_preds == eval_labels).item() / len(eval_labels)
    
    # Get number of selected features
    n_selected = 0
    selected_features = []
    for k, v in intervenable.interventions.items():
        n_selected += v.count_selected_features()
        selected_features.extend(v.get_selected_features())
    
    print(f"Test accuracy: {acc:.4f}, Selected features: {n_selected}")
    
    if return_details:
        correct = (eval_preds == eval_labels).int()
        details = {
            'eval_labels': eval_labels.cpu().tolist(),
            'eval_preds': eval_preds.cpu().tolist(),
            'correct': correct.cpu().tolist(),
            'indices': sample_indices,
            'n_selected_features': n_selected,
            'selected_features': selected_features,
        }
        return acc, details
    return acc


def find_candidate_alignments_boundless(
    model,
    dataset,
    poss,
    layers,
    batch_size,
    device,
    n_candidates=10,
    intervention_name=None,
    tokenizer=None,
    sparsity_coef=0.01,
    epochs=5,
    layer_suffix=None,  # For layer parallelism: add suffix to output files
):
    """
    Find candidate alignments using boundless DAS.
    
    Similar to the standard find_candidate_alignments but uses boundless DAS
    which learns per-feature masks instead of fixed-dimension subspaces.
    
    Args:
        layer_suffix: Optional suffix for output files (e.g., "_L0-10" for layer parallelism)
    """
    import os
    import json
    
    candidates = {}
    weights = {}
    feature_counts = {}
    
    # Split dataset
    train_dataset = dataset[:int(len(dataset) * 0.6)]
    test_dataset = dataset[int(len(dataset) * 0.6):]
    
    true_count = sum(1 for dp in test_dataset if dp["labels"] == dp["base_labels"])
    print(f"Overall Proportion of True labels: {true_count}/{len(test_dataset)} = {true_count/len(test_dataset):.2f}")
    
    os.makedirs("results", exist_ok=True)
    
    total_iterations = len(layers) * len(list(poss))
    current_iteration = 0
    
    embed_dim = getattr(model.config, 'hidden_size', getattr(model.config, 'n_embd', None))
    
    for layer in layers:
        for pos in poss:
            # Create fresh intervenable for EACH (layer, pos) combination
            # This ensures mask_weights start at 0 (sigmoid=0.5) for each position
            intervenable = config_boundless_das(model, layer, device, embed_dim=embed_dim)
            current_iteration += 1
            print(f"\n[{current_iteration}/{total_iterations}] Processing Layer {layer}, Position {pos}")
            
            # Create optimizer for boundless DAS parameters
            optimizer_params = []
            for k, v in intervenable.interventions.items():
                optimizer_params += [{"params": v.parameters()}]
            optimizer = torch.optim.Adam(optimizer_params, lr=0.001)
            
            # Train with boundless DAS
            boundless_das_training(
                intervenable,
                train_dataset,
                optimizer,
                pos=pos,
                device=device,
                epochs=epochs,
                batch_size=batch_size,
                tokenizer=tokenizer,
                sparsity_coef=sparsity_coef,
            )
            
            # Test
            intervenable.disable_model_gradients()
            acc = boundless_das_test(intervenable, pos, test_dataset, device=device, batch_size=batch_size)
            candidates[(layer, pos)] = acc
            print(f"Layer {layer}, Position {pos}: Test Accuracy = {acc:.4f}")
            
            # Save weights and feature counts
            intervention_key = f"layer_{layer}_comp_block_output_unit_pos_nunit_1#0"
            sd = intervenable.interventions[intervention_key].state_dict()
            weights[(layer, pos)] = {k: v.clone().detach().cpu() for k, v in sd.items()}
            
            n_selected = intervenable.interventions[intervention_key].count_selected_features()
            feature_counts[(layer, pos)] = n_selected
            
            # Save partial results
            partial_candidates = {f"L{k[0]}_P{k[1]}": v for k, v in candidates.items()}
            partial_weights = {f"L{k[0]}_P{k[1]}": v for k, v in weights.items()}
            partial_feature_counts = {f"L{k[0]}_P{k[1]}": v for k, v in feature_counts.items()}
            
            # Build suffix with intervention name and layer range
            suffix = f"_{intervention_name}" if intervention_name else ""
            if layer_suffix:
                suffix += layer_suffix
            
            with open(f"results/candidates_boundless_partial{suffix}.json", "w") as f:
                json.dump(partial_candidates, f, indent=4)
            with open(f"results/feature_counts_partial{suffix}.json", "w") as f:
                json.dump(partial_feature_counts, f, indent=4)
            
            torch.save(partial_weights, f"results/weights_boundless_partial{suffix}.pt")
            print(f"Partial results saved ({current_iteration}/{total_iterations} completed)")
    
    # Sort candidates by accuracy
    candidates = sorted(candidates.items(), key=lambda x: x[1], reverse=True)
    candidates = candidates[:n_candidates]
    candidates = {f"L{k[0]}_P{k[1]}": v for k, v in candidates}
    
    weights = {f"L{k[0]}_P{k[1]}": v for k, v in weights.items() if f"L{k[0]}_P{k[1]}" in candidates.keys()}
    feature_counts = {f"L{k[0]}_P{k[1]}": v for k, v in feature_counts.items() if f"L{k[0]}_P{k[1]}" in candidates.keys()}
    
    return candidates, weights, feature_counts


def test_with_boundless_weights(
    model,
    layer,
    device,
    pos,
    test_dataset,
    batch_size=64,
    weight=None,
    embed_dim=None,
    return_details=False,
):
    """
    Test the model with pre-trained boundless DAS weights.
    
    Args:
        model: The language model
        layer: Layer to intervene on
        device: Device to run on
        pos: Position of intervention
        test_dataset: Test dataset
        batch_size: Batch size for evaluation
        weight: Pre-trained intervention weights (state_dict)
        embed_dim: Embedding dimension (inferred if not provided)
        return_details: If True, return per-sample details
        
    Returns:
        Accuracy (and details dict if return_details=True)
    """
    intervenable = config_boundless_das(model, layer, device, weight=weight, embed_dim=embed_dim)
    return boundless_das_test(intervenable, pos, test_dataset, device=device, batch_size=batch_size, return_details=return_details)

