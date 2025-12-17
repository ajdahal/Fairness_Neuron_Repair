#!/usr/bin/env python3
"""
Stage 1: Baseline Analysis and Bias Detection

This module implements:
1. Batch sampling (random sampling of males/females per batch)
2. Neural network with activation extraction and neuron masking
3. Training with batches
4. Linear SVM training on activations to predict gender
5. Bias neuron identification
6. Test accuracy evaluation
"""

import os
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
import argparse
import json
from pathlib import Path
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from datetime import datetime

from preprocess import preprocess_data_for_ml_models, patch_test_categories
from logger_utils import get_logger, init_logger
from config_utils import Params, initialize_seed

logger = get_logger(__name__)
DETERMINISTIC_SEED = int(os.getenv("GLOBAL_SEED", 42))


class SimpleNeuralNetwork(nn.Module):
    """Neural network with activation extraction and neuron masking support."""
    
    def __init__(self, input_size):
        super(SimpleNeuralNetwork, self).__init__()
        # Architecture: input -> 256 -> 128 -> 64 -> 1
        self.layer1 = nn.Linear(input_size, 256)
        self.layer2 = nn.Linear(256, 128)
        self.layer3 = nn.Linear(128, 64)
        self.output_layer = nn.Linear(64, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
        # Neuron mask for test-time intervention (all neurons active by default)
        self.register_buffer('neuron_mask', torch.ones(64))    # register_buffer is used to register a buffer in the model, it is a tensor not a parameter of the model, but is used for the model's computation
    
    def forward(self, x, return_activations=False):
        """Forward pass with optional activation extraction."""
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        activations_64 = self.relu(self.layer3(x))
        
        # Apply neuron mask for intervention
        activations_masked = activations_64 * self.neuron_mask          # initially all neurons are masked with 1, so no change in the activations
        
        output = self.sigmoid(self.output_layer(activations_masked))
        
        if return_activations:                                          # redundant for now, but is useful if we need to have an adversarial objective for the neural network
            return output, activations_masked
        return output
    
    
    def extract_activations(self, x):                                   # only return activations, but not the output ... forward method does it as well, is it redundant?
        """Extract activations from layer3 (64 neurons) without computing output."""
        with torch.no_grad():
            x = self.relu(self.layer1(x))
            x = self.relu(self.layer2(x))
            activations_64 = self.relu(self.layer3(x))
            # Apply current mask
            return activations_64 * self.neuron_mask
    
    def set_neuron_mask(self, neuron_indices, values):
        """
        Set mask values for specific neurons.
        neuron_indices: List of neuron indices [0-63]    |    values: List of mask values (0.0=deactivate, 0.0-1.0=scale, 1.0=normal)
        """
        if isinstance(neuron_indices, int):
            neuron_indices = [neuron_indices]
        if isinstance(values, (int, float)):
            values = [values] * len(neuron_indices)
        
        for idx, val in zip(neuron_indices, values):
            self.neuron_mask[idx] = val                                          # the values of the indices are set to the values in the values list
        
        # Print the neuron mask after setting
        logger.info(f"Neuron mask after setting indices {neuron_indices} to {values}:")
        logger.info(f"Mask values: {self.neuron_mask.cpu().numpy()}")
    
    
    def reset_neuron_mask(self):
        """Reset all neurons to active (mask = 1.0)."""
        self.neuron_mask.fill_(1.0)            # in-place operation that sets all the elements of the tensor to 1.0


def train_model(model, train_loader, criterion, optimizer, device, n_epochs=200):       # compare against the training loop in pytorch
    """Train model using PyTorch DataLoader."""
    model.train()
    
    for epoch in range(n_epochs):
        epoch_loss = 0.0
        n_batches = 0
        
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            
            optimizer.zero_grad()
            outputs = model(X_batch)
            outputs = outputs.squeeze(dim=1)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            n_batches += 1
        
        if epoch % 20 == 0:
            avg_loss = epoch_loss / n_batches
            logger.info(f"Epoch {epoch}, Average Loss: {avg_loss:.6f}")


def extract_activations_batch(model, X_tensor, device, batch_size=256):
    """Extract activations in batches to handle large datasets."""
    model.eval()
    activations_list = []
    
    with torch.no_grad():
        for i in range(0, len(X_tensor), batch_size):
            batch = X_tensor[i:i+batch_size].to(device)
            activations = model.extract_activations(batch)
            activations_list.append(activations.cpu())
    
    return torch.cat(activations_list, dim=0).numpy()      # verically stack the activations across the batches - see the shape of the output to understand this


def normalize_activations(activations_train, activations_test=None):
    """
    Normalize activations column-wise (per neuron) to [0, 1] range using sklearn's MinMaxScaler.
    Fits scaler on training data and applies same transformation to test data.
    """
    # Create MinMaxScaler (default feature_range=(0, 1))
    scaler = MinMaxScaler()
    
    # Fit on training data and transform training activations
    activations_train_norm = scaler.fit_transform(activations_train)            # it's per column metric ... for each column, min and max are calculated and then the values are normalized to the range [0, 1] for that column
    
    # Log min/max per column for verification
    min_per_col = scaler.data_min_
    max_per_col = scaler.data_max_
    logger.info(f"Normalized training activations: min per column = {np.round(min_per_col, 3)}, max per column = {np.round(max_per_col, 3)}")   # 64 min values and 64 max values for each column?
    
    # Transform test activations using same scaler (fitted on training data)
    activations_test_norm = scaler.transform(activations_test)
    logger.info(f"Normalized test activations using training min/max")
    
    return activations_train_norm, activations_test_norm, scaler

def train_gender_classifier(activations_train, gender_train):
    """
    Train linear SVM to predict gender from activations.
    Uses weighted loss to penalize misclassification of minority group proportionally.
    """
    # Deterministic mapping: Female -> 0, Male -> 1 (alphabetical standard)
    gender_map = {'Female': 0, 'Male': 1}             # gender mapping for SVM isn't affected by the one hot mapping for the model
    
    # Transform training labels
    unknown = set(str(g) for g in gender_train) - set(gender_map.keys())
    if unknown:
        raise ValueError(f"Unknown gender values: {unknown}. Expected: Male, Female")
    gender_encoded = np.array([gender_map[str(g)] for g in gender_train])
    
    # Count samples
    n_male = np.sum(gender_encoded == 1)
    n_female = np.sum(gender_encoded == 0)
    
    logger.info(f"Gender distribution - Male (1): {n_male}, Female (0): {n_female}")
    
    # Train LinearSVC with balanced class weights
    # This automatically sets weight = n_samples / (n_classes * n_samples_j)
    # So minority class gets higher weight proportionally.
    logger.info("Training SVM with class_weight='balanced' to penalize minority misclassification")
    svm = LinearSVC(random_state=DETERMINISTIC_SEED, max_iter=10000, class_weight='balanced')       # penalizes with higher weights when misclassifying the minority class
    svm.fit(activations_train, gender_encoded)
    
    # Get training accuracy
    train_pred = svm.predict(activations_train)
    train_acc = accuracy_score(gender_encoded, train_pred)
    
    # Per-class accuracy
    male_acc = accuracy_score(gender_encoded[gender_encoded == 1], train_pred[gender_encoded == 1])
    female_acc = accuracy_score(gender_encoded[gender_encoded == 0], train_pred[gender_encoded == 0])
    
    logger.info(f"Gender classifier training accuracy: {train_acc:.4f}")
    logger.info(f" Training - Male accuracy: {male_acc:.4f}")
    logger.info(f" Training - Female accuracy: {female_acc:.4f}")
    
    return svm, gender_map, train_acc


def identify_biased_neurons(svm, model, top_k=10):
    """
    Identify most biased neurons using SVM coefficients AND model output weights.
    Importance = |SVM_coef_relative_wrto_mean| * |Model_Output_Weight_relative_wrto_mean|
    
    This ensures we identify neurons that:
    1. Encode gender information (high SVM coef)
    2. Are actually used by the model for prediction (high output weight)
    """
    # Get SVM coefficients (shape: [1, 64])
    svm_coef = np.abs(svm.coef_[0])
    
    # Get model output layer weights (shape: [1, 64])
    # Detach and move to cpu/numpy
    model_weights = model.output_layer.weight.data.abs().cpu().numpy().flatten()
    
    logger.info(f"Absolute SVM coefficients: {np.array2string(svm_coef, precision=3, suppress_small=True)}")
    logger.info(f"Absolute Model output weights: {np.array2string(model_weights, precision=3, suppress_small=True)}")
    
    # Normalize to get relative strength (handle divide by zero with small epsilon)
    # We divide by the MEAN of ABSOLUTE values to handle negative coefficients correctly
    svm_mean = np.mean(svm_coef) + 1e-10
    weight_mean = np.mean(model_weights) + 1e-10
    
    svm_relative = svm_coef / svm_mean
    weight_relative = model_weights / weight_mean
    
    logger.info(f"Relative SVM coefficients: {np.array2string(svm_relative, precision=3, suppress_small=True)}")
    logger.info(f"Relative Model output weights: {np.array2string(weight_relative, precision=3, suppress_small=True)}")
    
    # Calculate combined importance using relative strengths
    importance = svm_relative * weight_relative     # see what kind of multiplication this one is
    
    logger.info(f"importance for the dropout of the neurons: {np.array2string(importance, precision=3, suppress_small=True)}")
    
    # Get direction from SVM (sign of coefficient)
    direction = np.sign(svm.coef_[0])                      # the magnitude of the coefficient indicates the strength for one class and, the sign indicates the direction of the bias
    
    # Rank by importance
    top_indices = np.argsort(importance)[::-1][:top_k]      # every information needs to be sorted in descending order
    
    biased_neurons = [
        (int(idx), float(importance[idx]), int(direction[idx]))
        for idx in top_indices
    ]
    return biased_neurons


def calculate_counterfactual_discrimination(model, X_test, ht, sensitive_attr, train_data, params, device):
    """
    Calculate counterfactual discrimination by testing gender flip in both directions.
    For Male instances: Change to Female, check if prediction changes (difference = 1)
    For Female instances: Change to Male, check if prediction changes (difference = 1)
    """
    model.eval()
    
    # Create masks from original data
    male_mask = X_test[sensitive_attr] == 'Male'
    female_mask = X_test[sensitive_attr] == 'Female'
    
    # Create counterfactual: flip all genders (Male→Female, Female→Male) in one dataframe
    X_test_cf = X_test.copy()
    X_test_cf.loc[male_mask, sensitive_attr] = 'Female'
    X_test_cf.loc[female_mask, sensitive_attr] = 'Male'
    
    # Prepare train data for patching
    output_column = params.output_column_name
    train_X = train_data.drop(columns=[output_column]) if output_column and output_column in train_data.columns else train_data
    
    # Patch both original and counterfactual data
    categorical_cols = [col for col, typ in params.hypertransformer_config["sdtypes"].items() if typ == "categorical"]
    X_test_patched, dummy_indices_orig = patch_test_categories(train_X, X_test, categorical_cols, "test")
    X_test_cf_patched, dummy_indices_cf = patch_test_categories(train_X, X_test_cf, categorical_cols, "test")
    
    logger.info(f"Dummy indices for original data: {dummy_indices_orig}")
    logger.info(f"Dummy indices for counterfactual data: {dummy_indices_cf}")
    
    # Verify they have the same dummy indices (should be true if only gender changed)
    # If they differ, we cannot guarantee row alignment
    if dummy_indices_orig != dummy_indices_cf:
        raise ValueError(
            f"Dummy indices must match for correct row alignment: "
            f"original={dummy_indices_orig}, counterfactual={dummy_indices_cf}. "
            f"This indicates preprocessing inconsistency that would cause incorrect comparisons."
        )
    logger.info(f"Dummy indices match: {dummy_indices_orig}")
    
    # Use the dummy indices (they should be the same)
    dummy_indices = dummy_indices_orig
    
    logger.info(f"First instance of x_test_patched: {X_test_patched.iloc[0]}")
    logger.info(f"First instance of x_test_cf_patched: {X_test_cf_patched.iloc[0]}")
    
    # Transform both original and counterfactual
    X_test_transformed = ht.transform(X_test_patched)
    X_test_cf_transformed = ht.transform(X_test_cf_patched)
    
    # Drop dummy indices from both
    if dummy_indices:
        X_test_transformed = X_test_transformed.drop(index=dummy_indices).reset_index(drop=True)
        X_test_cf_transformed = X_test_cf_transformed.drop(index=dummy_indices).reset_index(drop=True)
        logger.info(f"Dropped {len(dummy_indices)} dummy rows from both original and counterfactual data")
    
    # Validation: Ensure final data length matches original X_test
    # This verifies that row order is preserved (masks created from X_test will align correctly)
    if len(X_test_transformed) != len(X_test):
        raise ValueError(
            f"Length mismatch: final transformed data has {len(X_test_transformed)} rows, "
            f"but original X_test has {len(X_test)} rows. Row alignment cannot be guaranteed."
        )
    if len(X_test_cf_transformed) != len(X_test):
        raise ValueError(
            f"Length mismatch: final counterfactual data has {len(X_test_cf_transformed)} rows, "
            f"but original X_test has {len(X_test)} rows. Row alignment cannot be guaranteed."
        )
    
    # Validate lengths match between original and counterfactual
    if len(X_test_transformed) != len(X_test_cf_transformed):
        raise ValueError(
            f"Length mismatch after preprocessing: original={len(X_test_transformed)}, "
            f"counterfactual={len(X_test_cf_transformed)}"
        )
    
    # Convert to tensors and get predictions
    X_test_tensor = torch.FloatTensor(X_test_transformed.values).to(device)
    X_test_cf_tensor = torch.FloatTensor(X_test_cf_transformed.values).to(device)
    
    with torch.no_grad():
        original_pred = (model(X_test_tensor).cpu().numpy().flatten() > 0.5).astype(int)
        cf_pred = (model(X_test_cf_tensor).cpu().numpy().flatten() > 0.5).astype(int)
    
    # Calculate discrimination: prediction changed (difference == 1) means discrimination
    total_disc = np.sum(original_pred != cf_pred)               # check the indices of the original and counterfactuals on gender 
    
    return {
        'total_discrimination': total_disc,
        'total_tested': len(X_test)
    }


def main(params, device, timestamp):
    """Main function for Stage 1: Baseline analysis and bias detection."""
    
    start_time = datetime.now()
    logger.info("STAGE 1: BASELINE ANALYSIS AND BIAS DETECTION")
    logger.info(f"Process start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Load original training data for gender labels
    train_data = pd.read_csv(params.train_file_path)
    test_data = pd.read_csv(params.test_file_path)
    sensitive_attr = params.sensitive_attr
    
    # Preprocess data
    X_train, X_test, ht, y_train, y_test = preprocess_data_for_ml_models(params)
    
    # Convert to tensors
    X_train_tensor = torch.FloatTensor(X_train.values if hasattr(X_train, 'values') else X_train)
    X_test_tensor = torch.FloatTensor(X_test.values if hasattr(X_test, 'values') else X_test)
    y_train_tensor = torch.FloatTensor(y_train.values if hasattr(y_train, 'values') else y_train)
    y_test_tensor = torch.FloatTensor(y_test.values if hasattr(y_test, 'values') else y_test)
    
    # Get gender labels aligned with transformed data
    # CRITICAL: Use index-based alignment to ensure one-to-one correspondence.

    if not hasattr(X_train, 'index'):
        raise ValueError("X_train must be a DataFrame with an index to guarantee alignment.")
    
    gender_train = train_data.loc[X_train.index, sensitive_attr].values
    
    # For X_test, preprocessing involves reset_index(drop=True) after handling dummy rows.
    # This resets the index to a RangeIndex (0, 1, ... N-1).
    # Since test_data is loaded fresh, it also has a RangeIndex (0, 1, ... N-1).

    # so the indices align positionally.
    if not hasattr(X_test, 'index'):
         raise ValueError("X_test must be a DataFrame with an index to guarantee alignment.")
         
    gender_test = test_data.loc[X_test.index, sensitive_attr].values
    
    logger.info(f"Training data: {len(X_train)} samples")
    logger.info(f"Gender distribution - Training Data: {pd.Series(gender_train).value_counts().to_dict()}")
    logger.info(f"Test data: {len(X_test)} samples")
    
    # Model setup
    input_size = X_train_tensor.shape[1]
    batch_size = getattr(params, 'batch_size', 256)
    default_nn_lr = 0.001
    
    model = SimpleNeuralNetwork(input_size).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=default_nn_lr, weight_decay=1e-4)
    
    # Create DataLoader
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    logger.info(f"\nCreated DataLoader with {len(train_loader)} batches (batch_size={batch_size})")
    
    # Train model
    logger.info("\nTraining neural network...")
    train_model(model, train_loader, criterion, optimizer, device, n_epochs=300)
    
    # Save model
    model_path = os.path.join(params.ml_models_dir, f"{params.dataset_name}__nn_stage1__{timestamp}.pth")
    torch.save(model.state_dict(), model_path)
    logger.info(f"Model saved to: {model_path}")
    
    # Extract activations from training data
    logger.info("\nExtracting activations from layer3 (64 neurons)...")
    activations_train = extract_activations_batch(model, X_train_tensor, device)
    logger.info(f"Training activations shape: {activations_train.shape}")
    
    # Extract activations from test data
    activations_test = extract_activations_batch(model, X_test_tensor, device)
    logger.info(f"Test activations shape: {activations_test.shape}")
    
    # Normalize activations column-wise (per neuron) to [0, 1] range
    logger.info("\nNormalizing activations column-wise (per neuron) to [0, 1] range...")
    activations_train_norm, activations_test_norm, activation_scaler = normalize_activations(activations_train, activations_test)
    
    logger.info(f"Normalized training activations shape: {activations_train_norm.shape}")
    logger.info(f"Normalized test activations shape: {activations_test_norm.shape}")
    
    # Train gender classifier on normalized activations
    logger.info("\nTraining linear SVM to predict gender from normalized activations...")          # needs to align at all times - gender labels after preprocessing
    svm, gender_map, svm_train_acc = train_gender_classifier(activations_train_norm, gender_train)
    
    # Evaluate gender classifier on normalized test activations
    # Transform test labels using the mapping dictionary
    unknown = set(str(g) for g in gender_test) - set(gender_map.keys())
    if unknown:
        raise ValueError(f"Unknown gender values: {unknown}. Expected: Male, Female")
    gender_test_encoded = np.array([gender_map[str(g)] for g in gender_test])
    gender_test_pred = svm.predict(activations_test_norm)
    svm_test_acc = accuracy_score(gender_test_encoded, gender_test_pred)
    
    logger.info(f"Gender classifier test accuracy: {svm_test_acc:.4f}")
    logger.info(f"Baseline (random): 0.5000")
    logger.info(f"Bias detected: {'Yes' if svm_test_acc > 0.55 else 'No'} (threshold: 0.55)")
    
    # Identify biased neurons (get top 60 for masking experiments)
    logger.info("\nIdentifying most biased neurons (Combined Importance: SVM_Coef * Output_Weight)...")
    biased_neurons = identify_biased_neurons(svm, model, top_k=60)

    
    # Save biased neurons
    biased_neurons_path = os.path.join(params.ml_models_dir, f"{params.dataset_name}_biased_neurons_stage1_{timestamp}.csv")
    biased_df = pd.DataFrame(biased_neurons, columns=['neuron_idx', 'importance', 'direction'])
    biased_df.to_csv(biased_neurons_path, index=False)
    logger.info(f"Biased neurons saved to: {biased_neurons_path}")
    
    # Save activations with gender
    logger.info("\nSaving activations with gender labels...")
    activation_train_df = pd.DataFrame(
        activations_train,
        columns=[f'neuron_{i}' for i in range(64)]
    )
    activation_train_df['gender'] = gender_train
    activation_train_path = os.path.join(params.ml_models_dir, f"{params.dataset_name}_activations_train_stage1_{timestamp}.csv")
    activation_train_df.to_csv(activation_train_path, index=False)
    logger.info(f"Training activations saved to: {activation_train_path}")
    
    activation_test_df = pd.DataFrame(
        activations_test,
        columns=[f'neuron_{i}' for i in range(64)]
    )
    activation_test_df['gender'] = gender_test
    activation_test_path = os.path.join(params.ml_models_dir, f"{params.dataset_name}_activations_test_stage1_{timestamp}.csv")
    activation_test_df.to_csv(activation_test_path, index=False)
    logger.info(f"Test activations saved to: {activation_test_path}")
    
    # Save gender classifier
    svm_path = os.path.join(params.ml_models_dir, f"{params.dataset_name}_gender_classifier_stage1_{timestamp}.pkl")
    with open(svm_path, 'wb') as f:
        pickle.dump({'svm': svm, 'gender_map': gender_map}, f)
    logger.info(f"Gender classifier saved to: {svm_path}")
    
    # Convert y_test to numpy for metrics calculation
    y_test_numpy = y_test.values if hasattr(y_test, 'values') else np.array(y_test)
    if isinstance(y_test_numpy, pd.Series):
        y_test_numpy = y_test_numpy.values
    y_test_numpy = y_test_numpy.astype(int)
    
    # Evaluate income classification accuracy (baseline)
    logger.info("\nEvaluating income classification accuracy...")
    model.eval()
    
    # Log output weights before reset
    weights_before = model.output_layer.weight.data.clone()
    logger.info(f"Weights before reset : {np.round(weights_before.cpu().numpy().flatten(), 3)}")
    
    model.reset_neuron_mask()  # Ensure all neurons are active - check this it might be turning activations to 1 for all the neurons - but mask isn't multiplied with weights - need to reference a tutorial 
    
    # Log output weights after reset and check for changes
    weights_after = model.output_layer.weight.data.clone()
    logger.info(f"Weights after reset : {np.round(weights_after.cpu().numpy().flatten(), 3)}")
    
    with torch.no_grad():
        y_pred_probs = model(X_test_tensor.to(device)).cpu().numpy().flatten()
        
        # Log probabilities > 0.5 and < 0.5 with averages
        probs_high = y_pred_probs[y_pred_probs > 0.5]
        probs_low = y_pred_probs[y_pred_probs < 0.5]
        if len(probs_high) > 0:
            logger.info(f"Baseline - Probabilities > 0.5: {len(probs_high)} instances, values: {np.round(probs_high, 4)}")
            logger.info(f"Baseline - Average probability for > 0.5: {np.mean(probs_high):.4f}")
        else:
            logger.info(f"Baseline - Probabilities > 0.5: 0 instances (all predictions are <= 0.5)")
        if len(probs_low) > 0:
            logger.info(f"Baseline - Probabilities < 0.5: {len(probs_low)} instances, values: {np.round(probs_low, 4)}")
            logger.info(f"Baseline - Average probability for < 0.5: {np.mean(probs_low):.4f}")
        else:
            logger.info(f"Baseline - Probabilities < 0.5: 0 instances (all predictions are >= 0.5)")
        
        y_pred_baseline = (y_pred_probs > 0.5).astype(int)
    
    income_acc_baseline = accuracy_score(y_test_numpy, y_pred_baseline)
    logger.info(f"Income classification test accuracy: {income_acc_baseline:.4f}")
    
    # Calculate baseline counterfactual discrimination
    logger.info("BASELINE COUNTERFACTUAL DISCRIMINATION (No neurons masked)")

    # Prepare test data (drop output column if present)
    X_test_for_cf = test_data.drop(columns=[params.output_column_name]) if params.output_column_name and params.output_column_name in test_data.columns else test_data.copy()
    cf_baseline = calculate_counterfactual_discrimination(model, X_test_for_cf, ht, sensitive_attr, train_data, params, device)
    
    logger.info(f"Income Classification Accuracy: {income_acc_baseline:.4f}")
    logger.info(f"Counterfactual Discrimination:")
    logger.info(f"  Total: {cf_baseline['total_discrimination']}/{cf_baseline['total_tested']} ({100*cf_baseline['total_discrimination']/cf_baseline['total_tested']:.2f}%)")
    
    # Loop for neuron masking experiments
    masking_levels = [10, 20, 30, 40, 45, 50, 54, 58, 60, 62]
    results = {
        'baseline': {
            'accuracy': income_acc_baseline,
            'discrimination': cf_baseline
        }
    }
    
    for k in masking_levels:
        logger.info(f"METRICS WITH TOP {k} BIASED NEURONS MASKED")
        # Get top k (or all available)
        current_top_k_indices = [neuron[0] for neuron in biased_neurons[:k]]
        logger.info(f"Neurons masked: {current_top_k_indices}")
        
        model.eval()
        
        # Log activations for the first row BEFORE masking (all active)  |  We perform reset first to ensure clean state
        model.reset_neuron_mask()
        
        sample_input = X_test_tensor[0:1].to(device)
        with torch.no_grad():
            activations_before = model.extract_activations(sample_input).cpu().numpy().flatten()
        logger.info(f"Activations (first row) BEFORE masking: {np.round(activations_before, 2)}")
        
        # Note: This zeroes out the activations of these neurons, effectively removing their contribution to the next layer (equivalent to zeroing weights for this input)
        model.set_neuron_mask(current_top_k_indices, [0.0] * len(current_top_k_indices))            #  is it applied to top k neurons or just from the top of the layer?
        
        # Log activations for the first row AFTER masking (should see zeros at masked indices)
        with torch.no_grad():
            activations_after = model.extract_activations(sample_input).cpu().numpy().flatten()
        logger.info(f"Activations (first row) AFTER masking top {k}: {np.round(activations_after, 2)}")
        
        # Verify zeros at masked indices
        masked_values = activations_after[current_top_k_indices]
        logger.info(f"Values at masked indices {current_top_k_indices}: {masked_values}")
        
        with torch.no_grad():
            y_pred_probs_k = model(X_test_tensor.to(device)).cpu().numpy().flatten()
            
            # Log probabilities > 0.5 and < 0.5 with averages
            probs_high_k = y_pred_probs_k[y_pred_probs_k > 0.5]
            probs_low_k = y_pred_probs_k[y_pred_probs_k < 0.5]
            if len(probs_high_k) > 0:
                logger.info(f"Top {k} masked - Probabilities > 0.5: {len(probs_high_k)} instances, values: {np.round(probs_high_k, 4)}")
                logger.info(f"Top {k} masked - Average probability for > 0.5: {np.mean(probs_high_k):.4f}")
            else:
                logger.info(f"Top {k} masked - Probabilities > 0.5: 0 instances (all predictions are <= 0.5)")
            if len(probs_low_k) > 0:
                logger.info(f"Top {k} masked - Probabilities < 0.5: {len(probs_low_k)} instances, values: {np.round(probs_low_k, 4)}")
                logger.info(f"Top {k} masked - Average probability for < 0.5: {np.mean(probs_low_k):.4f}")
            else:
                logger.info(f"Top {k} masked - Probabilities < 0.5: 0 instances (all predictions are >= 0.5)")
            
            y_pred_k = (y_pred_probs_k > 0.5).astype(int)
        
        acc_k = accuracy_score(y_test_numpy, y_pred_k)
        cf_k = calculate_counterfactual_discrimination(model, X_test_for_cf, ht, sensitive_attr, train_data, params, device)
        
        logger.info(f"Income Classification Accuracy: {acc_k:.4f}")
        logger.info(f"Counterfactual Discrimination:")
        logger.info(f"  Total: {cf_k['total_discrimination']}/{cf_k['total_tested']} ({100*cf_k['total_discrimination']/cf_k['total_tested']:.2f}%)")
        
        results[k] = {
            'accuracy': acc_k,
            'discrimination': cf_k
        }
    
    # Summary
    logger.info("STAGE 1 SUMMARY")
    logger.info(f"Income classification accuracy (baseline): {results['baseline']['accuracy']:.4f}")
    
    for k in masking_levels:
        if k in results:
            logger.info(f"Income classification accuracy (top {k} masked): {results[k]['accuracy']:.4f}")
            
    logger.info(f"Counterfactual discrimination (baseline): {results['baseline']['discrimination']['total_discrimination']}/{results['baseline']['discrimination']['total_tested']} ({100*results['baseline']['discrimination']['total_discrimination']/results['baseline']['discrimination']['total_tested']:.2f}%)")
    
    for k in masking_levels:
        if k in results:
            logger.info(f"Counterfactual discrimination (top {k} masked): {results[k]['discrimination']['total_discrimination']}/{results[k]['discrimination']['total_tested']} ({100*results[k]['discrimination']['total_discrimination']/results[k]['discrimination']['total_tested']:.2f}%)")
            
    logger.info(f"Gender classification from activations (train): {svm_train_acc:.4f}")
    logger.info(f"Gender classification from activations (test): {svm_test_acc:.4f}")
    logger.info(f"Number of biased neurons identified: {len(biased_neurons)}")
    
    end_time = datetime.now()
    duration = end_time - start_time
    total_seconds = duration.total_seconds()
    minutes = int(total_seconds // 60)
    seconds = int(total_seconds % 60)
    logger.info(f"Process end time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Total execution time: {minutes} minutes {seconds} seconds ({total_seconds:.2f} seconds)")
    logger.info("="*80)
    
    return {
        'results': results,
        'gender_classifier_train_acc': svm_train_acc,
        'gender_classifier_test_acc': svm_test_acc,
        'biased_neurons': biased_neurons,
        'model': model,
        'svm': svm
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 1: Baseline analysis and bias detection")
    parser.add_argument("--config", required=True, help="Path to config file")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size for balanced sampling")
    args = parser.parse_args()
    
    with open(args.config) as f:
        config = json.load(f)
    
    params = Params(config)
    params.batch_size = args.batch_size
    
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
    
    os.makedirs(params.ml_models_dir, exist_ok=True)
    
    _, device = initialize_seed(args.seed)
    
    # Generate timestamp for all files (format: YYYYMMDD_HH_MM)
    start_time = datetime.now()
    timestamp = start_time.strftime('%Y%m%d_%H_%M')
    
    # Init logger
    log_path = project_root / "logs" / f"stage1_bias_analysis_{timestamp}.log"
    init_logger(log_path)
    
    main(params, device, timestamp)