#!/usr/bin/env python3
"""
Extract detailed analysis information:
1. Weight distribution ranges
2. SVM coefficient distribution ranges
3. Probability distribution for each k
4. Male vs Female favoring neuron counts for each k
"""

import torch
import pickle
import numpy as np
import pandas as pd
import os
from pathlib import Path
from stage1_bias_analysis import SimpleNeuralNetwork, preprocess_data_for_ml_models
from config_utils import Params
import json

def create_ranges(data, num_bins=20):
    """Create range bins and count instances in each range."""
    min_val = data.min()
    max_val = data.max()
    
    # Create bins
    bins = np.linspace(min_val, max_val, num_bins + 1)
    
    # Count in each bin
    counts, bin_edges = np.histogram(data, bins=bins)
    
    # Create range strings
    ranges = []
    for i in range(len(bin_edges) - 1):
        start = round(bin_edges[i], 5)
        end = round(bin_edges[i + 1], 5)
        count = int(counts[i])
        ranges.append((f"{start:.5f} - {end:.5f}", count))
    
    return ranges

def analyze_weights_and_svm(model_path, svm_path, input_size=102):
    """Analyze weight and SVM coefficient distributions."""
    # Load model
    model = SimpleNeuralNetwork(input_size)
    state = torch.load(model_path, map_location='cpu')
    model.load_state_dict(state)
    weights = model.output_layer.weight.data.abs().cpu().numpy().flatten()
    
    # Load SVM
    with open(svm_path, 'rb') as f:
        svm = pickle.load(f)['svm']
    svm_coef_abs = np.abs(svm.coef_[0])
    svm_coef_signed = svm.coef_[0]  # Keep sign for direction analysis
    
    print("=" * 80)
    print("1. OUTGOING WEIGHTS DISTRIBUTION (Second-to-Last to Last Layer)")
    print("=" * 80)
    weight_ranges = create_ranges(weights, num_bins=20)
    for range_str, count in weight_ranges:
        print(f"{range_str}: {count} neurons")
    
    print("\n" + "=" * 80)
    print("2. SVM COEFFICIENTS DISTRIBUTION (with Direction)")
    print("=" * 80)
    
    # Separate negative and positive coefficients
    negative_coef = svm_coef_signed[svm_coef_signed < 0]
    positive_coef = svm_coef_signed[svm_coef_signed > 0]
    zero_coef = svm_coef_signed[svm_coef_signed == 0]
    
    print("\nFemale-Favoring (Negative Coefficients):")
    if len(negative_coef) > 0:
        min_neg = negative_coef.min()
        max_neg = negative_coef.max()
        bins_neg = np.linspace(min_neg, max_neg, 11)  # 10 bins for negatives
        
        for i in range(len(bins_neg) - 1):
            start = round(bins_neg[i], 5)
            end = round(bins_neg[i + 1], 5)
            
            mask = (svm_coef_signed >= bins_neg[i]) & (svm_coef_signed < bins_neg[i + 1])
            if i == len(bins_neg) - 2:
                mask = (svm_coef_signed >= bins_neg[i]) & (svm_coef_signed <= bins_neg[i + 1])
            
            count = mask.sum()
            if count > 0:
                print(f"  {start:.5f} - {end:.5f}: {count} neurons")
    else:
        print("  No negative coefficients")
    
    print(f"\nZero Coefficients: {len(zero_coef)} neurons")
    
    print("\nMale-Favoring (Positive Coefficients):")
    if len(positive_coef) > 0:
        min_pos = positive_coef.min()
        max_pos = positive_coef.max()
        bins_pos = np.linspace(min_pos, max_pos, 11)  # 10 bins for positives
        
        for i in range(len(bins_pos) - 1):
            start = round(bins_pos[i], 5)
            end = round(bins_pos[i + 1], 5)
            
            mask = (svm_coef_signed >= bins_pos[i]) & (svm_coef_signed < bins_pos[i + 1])
            if i == len(bins_pos) - 2:
                mask = (svm_coef_signed >= bins_pos[i]) & (svm_coef_signed <= bins_pos[i + 1])
            
            count = mask.sum()
            if count > 0:
                print(f"  {start:.5f} - {end:.5f}: {count} neurons")
    else:
        print("  No positive coefficients")
    
    return weights, svm_coef_abs

def analyze_probability_distributions(model_path, svm_path, config_path, input_size=102):
    """Analyze probability distributions for each k."""
    from pathlib import Path
    
    # Load config
    with open(config_path) as f:
        config = json.load(f)
    params = Params(config)
    
    # Resolve paths
    project_root = Path(__file__).resolve().parent
    def resolve_path(path_str):
        if os.path.isabs(path_str):
            return path_str
        return str(project_root / path_str)
    
    params.dataset_dir = resolve_path(params.dataset_dir)
    params.ml_models_dir = resolve_path(params.ml_models_dir)
    
    if params.train_file_path and not os.path.isabs(params.train_file_path):
        params.train_file_path = str(Path(params.dataset_dir) / params.train_file_path)
    if params.test_file_path and not os.path.isabs(params.test_file_path):
        params.test_file_path = str(Path(params.dataset_dir) / params.test_file_path)
    
    # Load model
    model = SimpleNeuralNetwork(input_size)
    state = torch.load(model_path, map_location='cpu')
    model.load_state_dict(state)
    
    # Preprocess data
    X_train, X_test, ht, y_train, y_test = preprocess_data_for_ml_models(params)
    X_test_tensor = torch.FloatTensor(X_test.values if hasattr(X_test, 'values') else X_test)
    
    # Load biased neurons
    biased_df = pd.read_csv('ml_models/adult/adult_biased_neurons_stage1.csv')
    
    # Probability bins: 1.0-0.9, 0.9-0.8, ..., 0.1-0.0
    prob_bins = np.linspace(0.0, 1.0, 11)  # [0.0, 0.1, 0.2, ..., 1.0] (ascending for histogram)
    
    masking_levels = [10, 20, 30, 40, 45, 50, 54, 58, 60]
    
    print("\n" + "=" * 80)
    print("3. PROBABILITY DISTRIBUTION FOR EACH K")
    print("=" * 80)
    
    results = {}
    
    for k in masking_levels:
        # Reset mask
        model.reset_neuron_mask()
        
        # Get top k neurons
        top_k_neurons = biased_df.head(k)
        neuron_indices = top_k_neurons['neuron_idx'].tolist()
        
        # Apply mask
        model.set_neuron_mask(neuron_indices, [0.0] * len(neuron_indices))
        
        # Get predictions
        model.eval()
        with torch.no_grad():
            probs = model(X_test_tensor).cpu().numpy().flatten()
        
        # Count in each bin
        counts, _ = np.histogram(probs, bins=prob_bins)
        
        # Store results
        results[k] = {
            'probs': probs,
            'counts': counts
        }
        
        print(f"\n--- k = {k} ---")
        # Print in reverse order (1.0-0.9, 0.9-0.8, etc.)
        for i in range(len(prob_bins) - 1, 0, -1):
            start = prob_bins[i]
            end = prob_bins[i - 1]
            count = int(counts[i - 1])
            print(f"{start:.1f} - {end:.1f}: {count} instances")
    
    return results

def analyze_neuron_directions(biased_df_path):
    """Analyze male vs female favoring neurons for each k."""
    biased_df = pd.read_csv(biased_df_path)
    
    # Count total male and female favoring
    total_male = (biased_df['direction'] == 1).sum()
    total_female = (biased_df['direction'] == -1).sum()
    total_zero = (biased_df['direction'] == 0).sum()
    
    masking_levels = [10, 20, 30, 40, 45, 50, 54, 58, 60]
    
    print("\n" + "=" * 80)
    print("4. MALE vs FEMALE FAVORING NEURONS DROPPED FOR EACH K")
    print("=" * 80)
    print(f"Total Male-favoring neurons: {total_male}")
    print(f"Total Female-favoring neurons: {total_female}")
    print(f"Total Zero-direction neurons: {total_zero}")
    print()
    
    for k in masking_levels:
        top_k = biased_df.head(k)
        male_dropped = (top_k['direction'] == 1).sum()
        female_dropped = (top_k['direction'] == -1).sum()
        zero_dropped = (top_k['direction'] == 0).sum()
        
        male_ratio = (male_dropped / total_male * 100) if total_male > 0 else 0
        female_ratio = (female_dropped / total_female * 100) if total_female > 0 else 0
        male_ratio_decimal = (male_dropped / total_male) if total_male > 0 else 0
        female_ratio_decimal = (female_dropped / total_female) if total_female > 0 else 0
        
        print(f"k = {k}:")
        print(f"  Male-favoring dropped: {male_dropped} ({male_ratio:.2f}% or {male_ratio_decimal:.4f} ratio of total {total_male})")
        print(f"  Female-favoring dropped: {female_dropped} ({female_ratio:.2f}% or {female_ratio_decimal:.4f} ratio of total {total_female})")
        print(f"  Zero-direction dropped: {zero_dropped}")
        print()

if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent
    model_path = project_root / "ml_models/adult/adult__nn_stage1__.pth"
    svm_path = project_root / "ml_models/adult/adult_gender_classifier_stage1.pkl"
    biased_df_path = project_root / "ml_models/adult/adult_biased_neurons_stage1.csv"
    config_path = project_root / "adult_config.json"
    input_size = 102
    
    # 1 & 2: Weight and SVM distributions
    weights, svm_coef = analyze_weights_and_svm(str(model_path), str(svm_path), input_size)
    
    # 3: Probability distributions
    prob_results = analyze_probability_distributions(str(model_path), str(svm_path), str(config_path), input_size)
    
    # 4: Neuron directions
    analyze_neuron_directions(str(biased_df_path))


