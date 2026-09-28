#!/usr/bin/env python3
"""
GPT-2 Temporal Steering Validation

Tests whether positive/negative steering strengths move probe predictions
and downstream model behavior in the expected directions.

Workflow:
1. Load the best temporal direction (from find_temporal_direction_fixed.py)
2. Load trained probes from training script
3. For each steering strength (-2.0 to +2.0):
   - Apply steering to layer 11 activations
   - Measure probe predictions (should shift: negative→immediate, positive→long-term)
   - Measure any downstream behavior changes
4. Validate: correlation between steering strength and probe class

Expected result:
- Negative strength: probe predicts immediate (class 0)
- Positive strength: probe predicts long-term (class 1)
- Smooth dose-response curve
"""

import torch
import numpy as np
import json
import pickle
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from transformers import AutoModelForCausalLM, AutoTokenizer
import matplotlib.pyplot as plt

# Paths
REPO_ROOT = Path(__file__).parent.parent.parent
PROBES_DIR = REPO_ROOT / "research/probes"
DIRECTIONS_DIR = REPO_ROOT / "research/results/temporal_directions_fixed"
DATASET_PATH = REPO_ROOT / "data/raw/temporal_scope/temporal_scope_caa.json"
OUTPUT_DIR = REPO_ROOT / "results/gpt2_steering"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Configuration
MODEL_NAME = "gpt2"
LAYER = 11  # Best layer from validation
BEST_POSITION = "punctuation"  # or "final"
STEERING_STRENGTHS = [-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0]

def load_dataset():
    """Load CAA dataset."""
    with open(DATASET_PATH) as f:
        data = json.load(f)
    return data.get("pairs", data)

def load_probe(layer):
    """Load trained probe for layer."""
    probe_file = PROBES_DIR / f"temporal_caa_layer_{layer}_probe.pkl"
    with open(probe_file, "rb") as f:
        return pickle.load(f)

def load_direction(layer, position="final"):
    """Load direction vector."""
    direction_file = DIRECTIONS_DIR / f"layer_{layer}_{position}_direction.npy"
    return np.load(direction_file).astype(np.float32)

def extract_activations_with_steering(
    model, decoder_blocks, tokenizer, prompt, layer, 
    direction, strength
):
    """Extract activations with steering applied."""
    inputs = tokenizer(prompt, return_tensors="pt")
    
    activations = {}
    
    def hook_fn(layer_idx):
        def hook(module, input, output):
            # Get hidden states
            if isinstance(output, tuple):
                hidden = output[0]
            else:
                hidden = output
            
            # Apply steering at this layer
            if layer_idx == layer:
                # Add steering strength * direction
                if hidden.ndim == 3:
                    hidden = hidden.clone()
                    hidden[0, -1, :] += strength * torch.from_numpy(direction).to(hidden.device).to(hidden.dtype)
            
            # Extract activation
            if isinstance(output, tuple):
                activations[layer_idx] = hidden[0, -1, :].detach().cpu().numpy()
            elif isinstance(output, torch.Tensor) and len(output.shape) == 3:
                activations[layer_idx] = hidden[0, -1, :].detach().cpu().numpy()
            
            if isinstance(output, tuple):
                return (hidden,) + output[1:]
            return hidden
        return hook
    
    # Register hooks
    hooks = []
    for i, block in enumerate(decoder_blocks):
        hooks.append(block.register_forward_hook(hook_fn(i)))
    
    with torch.no_grad():
        model(**inputs)
    
    for hook in hooks:
        hook.remove()
    
    return activations

def test_steering_effect():
    """Test steering on probe predictions."""
    print("="*70)
    print("GPT-2 TEMPORAL STEERING TEST")
    print("="*70)
    print()
    
    # Load model
    print("Loading model and components...")
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model.eval()
    
    # Get decoder blocks
    decoder_blocks = model.transformer.h
    
    # Load probe and direction
    probe = load_probe(LAYER)
    direction = load_direction(LAYER, BEST_POSITION)
    
    # Load dataset
    pairs = load_dataset()
    
    print(f"Model: {MODEL_NAME}")
    print(f"Layer: {LAYER}")
    print(f"Probe: {LAYER}")
    print(f"Direction: {BEST_POSITION}")
    print(f"Dataset: {len(pairs)} pairs")
    print()
    
    # Test on all pairs (not just 10)
    test_pairs = pairs
    
    # For each strength, measure probe predictions
    results = []
    
    print("Testing steering strengths...")
    print(f"{'Strength':<12} {'Imm Pred':<12} {'LT Pred':<12} {'Mean Proj':<12} {'Separation':<12}")
    print("-" * 60)
    
    for strength in STEERING_STRENGTHS:
        immediate_preds = []
        longterm_preds = []
        mean_projections = []
        immediate_probs_class1 = []
        longterm_probs_class1 = []
        
        for pair in test_pairs:
            question = pair["question"]
            option_keys = [k for k in pair.keys() if k not in ["question", "category"]]
            
            if len(option_keys) < 2:
                continue
            
            immediate_key = option_keys[0]
            long_term_key = option_keys[1]
            
            # Build prompts
            immediate_prompt = question + "\n\nChoices:\n" + pair[immediate_key]
            long_term_prompt = question + "\n\nChoices:\n" + pair[long_term_key]
            
            # Extract activations with steering
            immediate_acts = extract_activations_with_steering(
                model, decoder_blocks, tokenizer, immediate_prompt, 
                LAYER, direction, strength
            )
            
            long_term_acts = extract_activations_with_steering(
                model, decoder_blocks, tokenizer, long_term_prompt, 
                LAYER, direction, strength
            )
            
            # Get activations at the layer
            imm_act = immediate_acts[LAYER].reshape(1, -1)
            lt_act = long_term_acts[LAYER].reshape(1, -1)
            
            # Probe prediction (class) and probability
            imm_pred = probe.predict(imm_act)[0]
            lt_pred = probe.predict(lt_act)[0]
            imm_prob = probe.predict_proba(imm_act)[0][1]  # prob of class 1
            lt_prob = probe.predict_proba(lt_act)[0][1]    # prob of class 1
            
            immediate_preds.append(imm_pred)
            longterm_preds.append(lt_pred)
            immediate_probs_class1.append(imm_prob)
            longterm_probs_class1.append(lt_prob)
            
            # Direction projection (separation)
            separation = np.dot(lt_act - imm_act, direction)
            mean_projections.append(separation[0])
        
        if not immediate_preds:
            continue
        
        # Summary for this strength
        imm_class_0_rate = sum(1 for p in immediate_preds if p == 0) / len(immediate_preds)
        lt_class_1_rate = sum(1 for p in longterm_preds if p == 1) / len(longterm_preds)
        mean_proj = np.mean(mean_projections)
        mean_separation = np.mean(mean_projections)
        
        # Also use probability for smoother signal
        mean_imm_prob_0 = np.mean([1 - p for p in immediate_probs_class1])
        mean_lt_prob_1 = np.mean(longterm_probs_class1)
        
        results.append({
            "strength": float(strength),
            "immediate_class_0_rate": float(imm_class_0_rate),
            "longterm_class_1_rate": float(lt_class_1_rate),
            "immediate_prob_class_0": float(mean_imm_prob_0),
            "longterm_prob_class_1": float(mean_lt_prob_1),
            "mean_projection": float(mean_proj),
            "mean_separation": float(mean_separation),
            "n_samples": len(immediate_preds),
        })
        
        print(f"{strength:+.2f}         {imm_class_0_rate:.3f}         {lt_class_1_rate:.3f}         {mean_proj:+.4f}    {mean_separation:+.4f}")
    
    print()
    print("="*70)
    print("INTERPRETATION")
    print("="*70)
    print()
    
    print("Key signals:")
    print("1. Immediate (class 0) rate should INCREASE with negative strength")
    print("2. Long-term (class 1) rate should INCREASE with positive strength")
    print("3. Mean projection should correlate with strength")
    print()
    
    # Compute correlations using probabilities (smoother signal)
    strengths_arr = np.array([r["strength"] for r in results])
    imm_prob_0 = np.array([r["immediate_prob_class_0"] for r in results])
    lt_prob_1 = np.array([r["longterm_prob_class_1"] for r in results])
    projections = np.array([r["mean_projection"] for r in results])
    
    # Filter out NaN values
    valid_idx = ~(np.isnan(imm_prob_0) | np.isnan(lt_prob_1) | np.isnan(projections))
    
    if valid_idx.sum() > 2:
        strengths_valid = strengths_arr[valid_idx]
        imm_prob_0_valid = imm_prob_0[valid_idx]
        lt_prob_1_valid = lt_prob_1[valid_idx]
        proj_valid = projections[valid_idx]
        
        if imm_prob_0_valid.std() > 0:
            corr_imm = np.corrcoef(strengths_valid, imm_prob_0_valid)[0, 1]
        else:
            corr_imm = np.nan
            
        if lt_prob_1_valid.std() > 0:
            corr_lt = np.corrcoef(strengths_valid, lt_prob_1_valid)[0, 1]
        else:
            corr_lt = np.nan
            
        if proj_valid.std() > 0:
            corr_proj = np.corrcoef(strengths_valid, proj_valid)[0, 1]
        else:
            corr_proj = np.nan
    else:
        corr_imm = corr_lt = corr_proj = np.nan
    
    print(f"Correlation (strength vs immediate class-0 prob): {corr_imm:.3f}")
    print(f"  Expected: negative (lower strength → higher immediate prediction)")
    print()
    print(f"Correlation (strength vs long-term class-1 prob): {corr_lt:.3f}")
    print(f"  Expected: positive (higher strength → higher long-term prediction)")
    print()
    print(f"Correlation (strength vs projection): {corr_proj:.3f}")
    print(f"  Expected: positive (higher strength → larger projection)")
    print()
    
    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    axes[0].plot(strengths_arr, imm_prob_0, "o-", linewidth=2, markersize=8)
    axes[0].set_xlabel("Steering Strength")
    axes[0].set_ylabel("Immediate (Class 0) Probability")
    axes[0].set_title(f"Immediate Predictions (corr={corr_imm:.3f})")
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim([0, 1])
    
    axes[1].plot(strengths_arr, lt_prob_1, "o-", linewidth=2, markersize=8, color="orange")
    axes[1].set_xlabel("Steering Strength")
    axes[1].set_ylabel("Long-term (Class 1) Probability")
    axes[1].set_title(f"Long-term Predictions (corr={corr_lt:.3f})")
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim([0, 1])
    
    axes[2].plot(strengths_arr, projections, "o-", linewidth=2, markersize=8, color="green")
    axes[2].set_xlabel("Steering Strength")
    axes[2].set_ylabel("Mean Projection onto Direction")
    axes[2].set_title(f"Direction Projection (corr={corr_proj:.3f})")
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "steering_results.png", dpi=150, bbox_inches="tight")
    print(f"Saved plot to {OUTPUT_DIR}/steering_results.png")
    
    # Save results with proper serialization
    output_data = {
        "config": {
            "model": MODEL_NAME,
            "layer": LAYER,
            "position": BEST_POSITION,
            "n_test_pairs": sum(r["n_samples"] for r in results) // len(results) if results else 0,
        },
        "results": results,
        "correlations": {
            "strength_vs_immediate_class_0_prob": float(corr_imm) if not np.isnan(corr_imm) else None,
            "strength_vs_longterm_class_1_prob": float(corr_lt) if not np.isnan(corr_lt) else None,
            "strength_vs_projection": float(corr_proj) if not np.isnan(corr_proj) else None,
        },
    }
    
    with open(OUTPUT_DIR / "steering_results.json", "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"Saved results to {OUTPUT_DIR}/steering_results.json")
    
    print()
    print("="*70)
    if not np.isnan(corr_imm) and not np.isnan(corr_lt):
        if abs(corr_imm) > 0.5 and abs(corr_lt) > 0.5:
            print("✓ STEERING SUCCESSFUL")
            print("Strong correlations indicate steering works as expected!")
        else:
            print("○ WEAK STEERING SIGNAL")
            print("Correlations are weak. Check direction/probe quality.")
    else:
        print("✗ STEERING INCONCLUSIVE")
        print("Could not compute correlations (constant predictions).")
    print("="*70)

if __name__ == "__main__":
    test_steering_effect()
