#!/usr/bin/env python3
"""
Stage 1: Baseline Analysis, Bias Detection, and INLP Repair

This module implements:
1. Neural network training on Income data
2. INLP (Iterative Nullspace Projection) to compute a Gender Neutrality Matrix
3. Weight Surgery to permanently "fix" the model
4. Comprehensive evaluation (Before vs. After)
"""

import os
import pickle
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
import argparse
import json
from pathlib import Path
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from datetime import datetime
from torch.utils.data import random_split

# Local imports
from preprocess import preprocess_data_for_ml_models, patch_test_categories
from logger_utils import get_logger, init_logger
from config_utils import Params, initialize_seed

# Import the new INLP core functions
from inlp_core import get_projection_and_svm, perform_weight_surgery

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
        
        # Neuron mask (kept for backward compatibility, though INLP uses weights)
        self.register_buffer('neuron_mask', torch.ones(64))
    
    def forward(self, x, return_activations=False, return_logits=False):
        """Forward pass with optional activation extraction."""
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        activations_64 = self.relu(self.layer3(x))
        
        # Apply neuron mask (default is all 1s)
        activations_masked = activations_64 * self.neuron_mask
        
        logits = self.output_layer(activations_masked)
        
        if return_logits:
            output = logits
        else:
            output = self.sigmoid(logits)
        
        if return_activations:
            return output, activations_masked
        return output
    
    def extract_activations(self, x):
        """Extract activations from layer3 (64 neurons) without computing output."""
        with torch.no_grad():
            x = self.relu(self.layer1(x))
            x = self.relu(self.layer2(x))
            activations_64 = self.relu(self.layer3(x))
            # Apply current mask
            return activations_64 * self.neuron_mask
    
    def reset_neuron_mask(self):
        """Reset all neurons to active (mask = 1.0)."""
        self.neuron_mask.fill_(1.0)            # in-place operation that sets all the elements of the tensor to 1.0


def train_model(model, train_loader, val_loader, criterion, optimizer, device, n_epochs=200, 
                early_stopping_patience=50, min_delta_ratio=0.0001):
    """
    Train model for fixed epochs with validation monitoring. Tracks and restores best validation model.
    """
    best_val_loss = float('inf')
    best_model_state = None
    best_epoch = 0
    
    for epoch in range(n_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        n_train_batches = 0
        
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            
            optimizer.zero_grad()
            # Get logits (before sigmoid) for BCEWithLogitsLoss
            logits = model(X_batch, return_logits=True)
            logits = logits.squeeze(dim=1)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            n_train_batches += 1
        
        avg_train_loss = train_loss / n_train_batches
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        n_val_batches = 0
        
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                
                # Get logits (before sigmoid) for BCEWithLogitsLoss
                logits = model(X_batch, return_logits=True)
                logits = logits.squeeze(dim=1)
                loss = criterion(logits, y_batch)
                
                val_loss += loss.item()
                n_val_batches += 1
        
        avg_val_loss = val_loss / n_val_batches
        
        # Logging
        if epoch % 20 == 0 or epoch == n_epochs - 1:
            logger.info(f"Epoch {epoch:3d} | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}")
        
        # Track best model based on validation loss
        if avg_val_loss < best_val_loss * (1 - min_delta_ratio):
            best_val_loss = avg_val_loss
            best_epoch = epoch
            # Save best model state
            best_model_state = copy.deepcopy(model.state_dict())
            if epoch % 20 != 0:  # Log if we didn't already log this epoch
                logger.info(f"Epoch {epoch:3d} | New best validation loss: {best_val_loss:.6f}")
    
    # Restore best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        logger.info(f"Restored best model from epoch {best_epoch} with validation loss: {best_val_loss:.6f}")
    
    return best_val_loss


def extract_activations_batch(model, X_tensor, device, batch_size):
    """Extract activations in batches to handle large datasets."""
    model.eval()
    activations_list = []
    
    with torch.no_grad():
        for i in range(0, len(X_tensor), batch_size):
            batch = X_tensor[i:i+batch_size].to(device)
            activations = model.extract_activations(batch)
            activations_list.append(activations.cpu())
    
    # Concatenate along batch dimension (dim=0) to stack batches: (batch_size, 64) -> (total_samples, 64)
    return torch.cat(activations_list, dim=0).numpy()


def calculate_probability_statistics(probs, dataset_name, log: bool = True):
    """
    Calculate average probabilities for values > 0.5 and < 0.5.
    """
    probs_high = probs[probs > 0.5]
    probs_low = probs[probs < 0.5]
    
    avg_high = np.mean(probs_high) if len(probs_high) > 0 else np.nan
    avg_low = np.mean(probs_low) if len(probs_low) > 0 else np.nan
    
    if log:
        if len(probs_high) > 0:
            logger.info(f"{dataset_name} - Probabilities > 0.5: {len(probs_high)} instances, Average: {avg_high:.4f}")
        else:
            logger.info(f"{dataset_name} - Probabilities > 0.5: 0 instances")
        
        if len(probs_low) > 0:
            logger.info(f"{dataset_name} - Probabilities < 0.5: {len(probs_low)} instances, Average: {avg_low:.4f}")
        else:
            logger.info(f"{dataset_name} - Probabilities < 0.5: 0 instances")
    
    return {
        'avg_high': avg_high,
        'avg_low': avg_low,
        'count_high': len(probs_high),
        'count_low': len(probs_low)
    }


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
    
    # Verify they have the same dummy indices (should be true if only gender changed)
    # If they differ, we cannot guarantee row alignment
    if dummy_indices_orig != dummy_indices_cf:
        raise ValueError(
            f"Dummy indices must match for correct row alignment: "
            f"original={dummy_indices_orig}, counterfactual={dummy_indices_cf}. "
            f"This indicates preprocessing inconsistency that would cause incorrect comparisons."
        )
    # Use the dummy indices (they should be the same)
    dummy_indices = dummy_indices_orig
    
    # Transform both original and counterfactual
    X_test_transformed = ht.transform(X_test_patched)
    X_test_cf_transformed = ht.transform(X_test_cf_patched)
    
    # Drop dummy indices from both
    if dummy_indices:
        X_test_transformed = X_test_transformed.drop(index=dummy_indices).reset_index(drop=True)
        X_test_cf_transformed = X_test_cf_transformed.drop(index=dummy_indices).reset_index(drop=True)
    
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
    
    # Calculate discrimination: count samples where flipping gender changes the prediction
    # This measures counterfactual unfairness - ideally should be 0 for a fair model
    total_disc = np.sum(original_pred != cf_pred) 
    
    return {
        'total_discrimination': total_disc,
        'total_tested': len(X_test)
    }


def _safe_rate(numer, denom):
    """Return numer/denom if denom>0 else np.nan."""
    return (numer / denom) if denom > 0 else np.nan


def calculate_aod_eod(y_true, y_pred, sensitive, privileged_value=1):
    """
    Equal Opportunity Difference (EOD) = |TPR_priv - TPR_unpriv|
    Average Odds Difference (AOD) = 0.5 * (|TPR_priv - TPR_unpriv| + |FPR_priv - FPR_unpriv|)
    Standard definitions aligned with group fairness literature (Hardt et al., 2016).
    """
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    sensitive = np.asarray(sensitive).astype(int)

    priv_mask = (sensitive == privileged_value)
    unpriv_mask = ~priv_mask

    def _rates(mask):
        yt = y_true[mask]
        yp = y_pred[mask]
        tp = int(np.sum((yt == 1) & (yp == 1)))
        fn = int(np.sum((yt == 1) & (yp == 0)))
        fp = int(np.sum((yt == 0) & (yp == 1)))
        tn = int(np.sum((yt == 0) & (yp == 0)))
        tpr = _safe_rate(tp, tp + fn)
        fpr = _safe_rate(fp, fp + tn)
        return tpr, fpr

    tpr_priv, fpr_priv = _rates(priv_mask)
    tpr_unpriv, fpr_unpriv = _rates(unpriv_mask)

    eod = np.abs(tpr_priv - tpr_unpriv) if (not np.isnan(tpr_priv) and not np.isnan(tpr_unpriv)) else np.nan
    if any(np.isnan(x) for x in [tpr_priv, tpr_unpriv, fpr_priv, fpr_unpriv]):
        aod = np.nan
    else:
        aod = 0.5 * (np.abs(tpr_priv - tpr_unpriv) + np.abs(fpr_priv - fpr_unpriv))

    return {
        'aod': float(aod) if not np.isnan(aod) else np.nan,
        'eod': float(eod) if not np.isnan(eod) else np.nan,
        'tpr_priv': float(tpr_priv) if not np.isnan(tpr_priv) else np.nan,
        'tpr_unpriv': float(tpr_unpriv) if not np.isnan(tpr_unpriv) else np.nan,
        'fpr_priv': float(fpr_priv) if not np.isnan(fpr_priv) else np.nan,
        'fpr_unpriv': float(fpr_unpriv) if not np.isnan(fpr_unpriv) else np.nan,
    }


def _pct_change(before, after):
    """Percent change: (after-before)/before * 100. Returns np.nan if before==0 or nan."""
    if before is None or after is None:
        return np.nan
    if isinstance(before, float) and np.isnan(before):
        return np.nan
    if isinstance(after, float) and np.isnan(after):
        return np.nan
    if before == 0:
        return np.nan
    return 100.0 * (after - before) / before


def main(params, device, timestamp):
    """Main function for Stage 1: INLP Repair Pipeline."""
    
    start_time = datetime.now()
    logger.info("STAGE 1: BIAS ANALYSIS AND INLP REPAIR")
    logger.info(f"Process start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # -------------------------------------------------------------------------
    # 1. Data Loading & Preprocessing
    # -------------------------------------------------------------------------
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
    
    # Align Gender Labels
    if not hasattr(X_train, 'index') or not hasattr(X_test, 'index'):
        raise ValueError("X_train/X_test must have indices for alignment.")
        
    gender_train = train_data.loc[X_train.index, sensitive_attr].values
    gender_test = test_data.loc[X_test.index, sensitive_attr].values
    
    # Map gender to 0/1 for SVM
    gender_map = {'Female': 0, 'Male': 1}
    y_gen_train_encoded = np.array([gender_map[str(g)] for g in gender_train])
    y_gen_test_encoded = np.array([gender_map[str(g)] for g in gender_test])
    
    logger.info(f"Training data: {len(X_train)} samples")
    logger.info(f"Gender distribution (Train): {pd.Series(gender_train).value_counts().to_dict()}")

    # -------------------------------------------------------------------------
    # 2. Train Original Model
    # -------------------------------------------------------------------------
    input_size = X_train_tensor.shape[1]
    batch_size = params.batch_size
    if batch_size is None:
        raise ValueError("batch_size must be specified in the config file")
    val_split_ratio = getattr(params, 'val_split_ratio', 0.15)
    if val_split_ratio is None:
        val_split_ratio = 0.15
    early_stopping_patience = getattr(params, 'early_stopping_patience', 50)
    if early_stopping_patience is None:
        early_stopping_patience = 50
    min_delta_ratio = getattr(params, 'min_delta_ratio', 0.0001)
    if min_delta_ratio is None:
        min_delta_ratio = 0.0001
    
    # Split training data into train and validation sets
    full_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    n_total = len(full_dataset)
    n_val = int(n_total * val_split_ratio)
    n_train = n_total - n_val
    
    train_dataset, val_dataset = random_split(
        full_dataset, 
        [n_train, n_val],
        generator=torch.Generator().manual_seed(DETERMINISTIC_SEED)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    logger.info(f"\nData split: {n_train} training samples, {n_val} validation samples")
    logger.info(f"Created DataLoaders: {len(train_loader)} train batches, {len(val_loader)} val batches (batch_size={batch_size})")
    
    # Calculate class weights for weighted BCELoss (handle class imbalance)
    n_class0 = (y_train_tensor == 0).sum().item()
    n_class1 = (y_train_tensor == 1).sum().item()
    pos_weight = torch.tensor([n_class0 / n_class1]).to(device)
    logger.info(f"Class distribution: Class 0={n_class0}, Class 1={n_class1}, pos_weight={pos_weight.item():.4f}")
    
    model = SimpleNeuralNetwork(input_size).to(device)
    # Use BCEWithLogitsLoss which supports pos_weight directly and is more numerically stable
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-2)
    
    logger.info("\nTraining neural network (Baseline) for fixed epochs with validation monitoring...")
    best_val_loss = train_model(
        model, train_loader, val_loader, criterion, optimizer, device, 
        n_epochs=200, early_stopping_patience=early_stopping_patience, 
        min_delta_ratio=min_delta_ratio
    )
    logger.info(f"Training completed. Best validation loss: {best_val_loss:.6f}")
    
    # Save original model
    model_path = os.path.join(params.ml_models_dir, f"{params.dataset_name}__nn_stage1_original__{timestamp}.pth")
    torch.save(model.state_dict(), model_path)
    logger.info(f"Original Model saved to: {model_path}")

    # -------------------------------------------------------------------------
    # 3. Calculate Baseline Metrics (Before INLP)
    # -------------------------------------------------------------------------
    logger.info("\n--- BASELINE EVALUATION (Before Surgery) ---")
    
    # A. Income Accuracy
    model.eval()
    with torch.no_grad():
        # Test data probabilities
        y_pred_probs_test = model(X_test_tensor.to(device)).cpu().numpy().flatten()
        y_pred_baseline = (y_pred_probs_test > 0.5).astype(int)
        
        # Train data probabilities
        y_pred_probs_train = model(X_train_tensor.to(device)).cpu().numpy().flatten()
    
    y_test_numpy = y_test.values if hasattr(y_test, 'values') else np.array(y_test)
    y_test_numpy = y_test_numpy.astype(int)
    
    acc_before = accuracy_score(y_test_numpy, y_pred_baseline)
    logger.info(f"Baseline Income Accuracy: {acc_before:.4%}")
    
    # Class-wise accuracy
    acc_class_0 = np.nan
    acc_class_1 = np.nan
    class_0_mask = (y_test_numpy == 0)
    class_1_mask = (y_test_numpy == 1)
    if class_0_mask.sum() > 0:
        acc_class_0 = accuracy_score(y_test_numpy[class_0_mask], y_pred_baseline[class_0_mask])
        logger.info(f"Baseline Accuracy (Class 0): {acc_class_0:.4%} ({class_0_mask.sum()} samples)")
    if class_1_mask.sum() > 0:
        acc_class_1 = accuracy_score(y_test_numpy[class_1_mask], y_pred_baseline[class_1_mask])
        logger.info(f"Baseline Accuracy (Class 1): {acc_class_1:.4%} ({class_1_mask.sum()} samples)")
    
    # Probability statistics (used in final summary; avoid redundant logging here)
    stats_test_before = calculate_probability_statistics(y_pred_probs_test, "Test", log=False)
    
    # B. Counterfactual Discrimination
    X_test_for_cf = test_data.drop(columns=[params.output_column_name]) if params.output_column_name and params.output_column_name in test_data.columns else test_data.copy()
    
    cf_before = calculate_counterfactual_discrimination(model, X_test_for_cf, ht, sensitive_attr, train_data, params, device)
    disc_count_before = cf_before['total_discrimination']
    disc_pct_before = (disc_count_before / cf_before['total_tested']) * 100
    logger.info(f"Baseline Counterfactual Discrimination: {disc_count_before}/{cf_before['total_tested']} ({disc_pct_before:.2f}%)")

    # C. AOD / EOD (test set, gender as sensitive attribute)
    aod_eod_before = calculate_aod_eod(
        y_true=y_test_numpy,
        y_pred=y_pred_baseline,
        sensitive=y_gen_test_encoded,
        privileged_value=gender_map.get('Male', 1),
    )
    logger.info(f"Baseline AOD: {aod_eod_before['aod']:.6f}" if not np.isnan(aod_eod_before['aod']) else "Baseline AOD: N/A")
    logger.info(f"Baseline EOD: {aod_eod_before['eod']:.6f}" if not np.isnan(aod_eod_before['eod']) else "Baseline EOD: N/A")

    # -------------------------------------------------------------------------
    # 4. INLP Process (Compute Projection)
    # -------------------------------------------------------------------------
    logger.info("\n--- STARTING INLP PROCESS ---")
    
    # Extract RAW Activations (No normalization needed for weight surgery)
    logger.info("Extracting raw activations from layer 3...")
    activations_train = extract_activations_batch(model, X_train_tensor, device, batch_size)
    activations_test = extract_activations_batch(model, X_test_tensor, device, batch_size)
    
    # Enhancement 2: Add activation statistics before projection
    logger.info("\n--- Activation Statistics (Before Projection) ---")
    logger.info(f"Train activations shape: {activations_train.shape}")
    logger.info(f"Train activations - Mean: {np.mean(activations_train):.6f}, Std: {np.std(activations_train):.6f}")
    logger.info(f"Train activations - L2 norm: {np.linalg.norm(activations_train):.6f}")
    logger.info(f"Train activations - Min: {np.min(activations_train):.6f}, Max: {np.max(activations_train):.6f}")
    logger.info(f"Test activations shape: {activations_test.shape}")
    logger.info(f"Test activations - Mean: {np.mean(activations_test):.6f}, Std: {np.std(activations_test):.6f}")
    logger.info(f"Test activations - L2 norm: {np.linalg.norm(activations_test):.6f}")
    logger.info("=" * 80)
    
    # Extract validation activations and gender labels using same split as NN training
    val_indices = val_dataset.indices
    activations_val = activations_train[val_indices]
    y_gen_val_encoded = y_gen_train_encoded[val_indices]
    
    # Use only training portion (excluding validation) for INLP training
    train_indices = train_dataset.indices
    activations_train_inlp = activations_train[train_indices]
    y_gen_train_inlp_encoded = y_gen_train_encoded[train_indices]
    
    # Compute Projection Matrix & Train Bias Detector
    logger.info("Computing Projection Matrix (P) and Training Gender SVM...")
    logger.info(f"Using same validation set as NN training: {len(activations_train_inlp)} train, {len(activations_val)} val")
    P, gender_svm, mean_activation = get_projection_and_svm(
        activations_train_inlp, y_gen_train_inlp_encoded, activations_val, y_gen_val_encoded, 
        n_iter=params.inlp_iters, alpha=params.inlp_alpha
    )
    
    # -------------------------------------------------------------------------
    # 5. Evaluate Gender Bias (Before vs. After Projection)
    # -------------------------------------------------------------------------
    logger.info("\n--- EVALUATING GENDER INFORMATION (SVM) ---")
    
    # Metric 1: SVM Accuracy on Raw Activations (Before INLP)
    svm_acc_train_before = gender_svm.score(activations_train - mean_activation, y_gen_train_encoded)
    svm_acc_test_before = gender_svm.score(activations_test - mean_activation, y_gen_test_encoded)

    # SVM class-wise accuracy (per gender class) on test
    svm_pred_test_before = gender_svm.predict(activations_test - mean_activation)
    svm_acc_test_female_before = accuracy_score(y_gen_test_encoded[y_gen_test_encoded == 0], svm_pred_test_before[y_gen_test_encoded == 0]) if np.any(y_gen_test_encoded == 0) else np.nan
    svm_acc_test_male_before = accuracy_score(y_gen_test_encoded[y_gen_test_encoded == 1], svm_pred_test_before[y_gen_test_encoded == 1]) if np.any(y_gen_test_encoded == 1) else np.nan
    
    logger.info(f"Gender SVM Accuracy (Before Projection):")
    logger.info(f"   Train: {svm_acc_train_before:.4%}")
    logger.info(f"   Test:  {svm_acc_test_before:.4%} (Indicates bias presence)")
    logger.info(f"   Test Female (class 0): {svm_acc_test_female_before:.4%}" if not np.isnan(svm_acc_test_female_before) else "   Test Female (class 0): N/A")
    logger.info(f"   Test Male (class 1):   {svm_acc_test_male_before:.4%}" if not np.isnan(svm_acc_test_male_before) else "   Test Male (class 1):   N/A")
    
    # Metric 2: SVM Accuracy on Projected Activations (After INLP)
    # We project the activations manually to see if the SVM can still find gender info
    # X_new = X_old @ P.T (Since P acts on column vectors in theory, but here we process row vectors)
    # P_step was symmetric, so P.T is used for row-vector multiplication
    # P was trained on centered data, so we must subtract mean
    activations_train_proj = (activations_train - mean_activation) @ P.T
    activations_test_proj = (activations_test - mean_activation) @ P.T
    
    # Enhancement 2: Add activation statistics after projection
    logger.info("\n--- Activation Statistics (After Projection) ---")
    logger.info(f"Train activations (projected) shape: {activations_train_proj.shape}")
    logger.info(f"Train activations (projected) - Mean: {np.mean(activations_train_proj):.6f}, Std: {np.std(activations_train_proj):.6f}")
    logger.info(f"Train activations (projected) - L2 norm: {np.linalg.norm(activations_train_proj):.6f}")
    logger.info(f"Test activations (projected) shape: {activations_test_proj.shape}")
    logger.info(f"Test activations (projected) - L2 norm: {np.linalg.norm(activations_test_proj):.6f}")
    
    # Compute change in activation magnitudes
    train_norm_before = np.linalg.norm(activations_train - mean_activation)
    train_norm_after = np.linalg.norm(activations_train_proj)
    test_norm_before = np.linalg.norm(activations_test - mean_activation)
    test_norm_after = np.linalg.norm(activations_test_proj)
    logger.info(f"Train activation norm change: {train_norm_before:.6f} -> {train_norm_after:.6f} ({100*(train_norm_after/train_norm_before - 1):.2f}%)")
    logger.info(f"Test activation norm change: {test_norm_before:.6f} -> {test_norm_after:.6f} ({100*(test_norm_after/test_norm_before - 1):.2f}%)")
    logger.info("=" * 80)
    
    svm_acc_train_after = gender_svm.score(activations_train_proj, y_gen_train_encoded)
    svm_acc_test_after = gender_svm.score(activations_test_proj, y_gen_test_encoded)

    svm_pred_test_after = gender_svm.predict(activations_test_proj)
    svm_acc_test_female_after = accuracy_score(y_gen_test_encoded[y_gen_test_encoded == 0], svm_pred_test_after[y_gen_test_encoded == 0]) if np.any(y_gen_test_encoded == 0) else np.nan
    svm_acc_test_male_after = accuracy_score(y_gen_test_encoded[y_gen_test_encoded == 1],svm_pred_test_after[y_gen_test_encoded == 1]) if np.any(y_gen_test_encoded == 1) else np.nan
    
    logger.info(f"Gender SVM Accuracy (After Projection):")
    logger.info(f"   Train: {svm_acc_train_after:.4%}")
    logger.info(f"   Test:  {svm_acc_test_after:.4%} (Should be ~50% if successful)")
    logger.info(f"   Test Female (class 0): {svm_acc_test_female_after:.4%}" if not np.isnan(svm_acc_test_female_after) else "   Test Female (class 0): N/A")
    logger.info(f"   Test Male (class 1):   {svm_acc_test_male_after:.4%}" if not np.isnan(svm_acc_test_male_after) else "   Test Male (class 1):   N/A")

    # -------------------------------------------------------------------------
    # 6. Weight Surgery
    # -------------------------------------------------------------------------
    logger.info("\n--- APPLYING WEIGHT SURGERY ---")
    perform_weight_surgery(model, P, mean_activation)
    
    # Save modified model
    model_surgery_path = os.path.join(params.ml_models_dir, f"{params.dataset_name}__nn_stage1_inlp_repaired__{timestamp}.pth")
    torch.save(model.state_dict(), model_surgery_path)
    logger.info(f"Repaired Model saved to: {model_surgery_path}")

    # -------------------------------------------------------------------------
    # 7. Final Evaluation (After Surgery)
    # -------------------------------------------------------------------------
    logger.info("\n--- FINAL EVALUATION (After Surgery) ---")
    
    # A. Income Accuracy
    model.eval()
    with torch.no_grad():
        # Test data probabilities
        y_pred_probs_after_test = model(X_test_tensor.to(device)).cpu().numpy().flatten()
        y_pred_after = (y_pred_probs_after_test > 0.5).astype(int)
        
        # Train data probabilities
        y_pred_probs_after_train = model(X_train_tensor.to(device)).cpu().numpy().flatten()
    
    acc_after = accuracy_score(y_test_numpy, y_pred_after)
    logger.info(f"Final Income Accuracy: {acc_after:.4%}")
    logger.info(f"Accuracy Drop: {acc_before - acc_after:.4%}")
    
    # Class-wise accuracy
    acc_class_0_after = np.nan
    acc_class_1_after = np.nan
    class_0_mask = (y_test_numpy == 0)
    class_1_mask = (y_test_numpy == 1)
    if class_0_mask.sum() > 0:
        acc_class_0_after = accuracy_score(y_test_numpy[class_0_mask], y_pred_after[class_0_mask])
        logger.info(f"Final Accuracy (Class 0): {acc_class_0_after:.4%} ({class_0_mask.sum()} samples)")
    if class_1_mask.sum() > 0:
        acc_class_1_after = accuracy_score(y_test_numpy[class_1_mask], y_pred_after[class_1_mask])
        logger.info(f"Final Accuracy (Class 1): {acc_class_1_after:.4%} ({class_1_mask.sum()} samples)")
    
    # Probability statistics (used in final summary; avoid redundant logging here)
    stats_test_after = calculate_probability_statistics(y_pred_probs_after_test, "Test", log=False)
    
    # B. Counterfactual Discrimination
    cf_after = calculate_counterfactual_discrimination(model, X_test_for_cf, ht, sensitive_attr, train_data, params, device)
    disc_count_after = cf_after['total_discrimination']
    disc_pct_after = (disc_count_after / cf_after['total_tested']) * 100
    logger.info(f"Final Counterfactual Discrimination: {disc_count_after}/{cf_after['total_tested']} ({disc_pct_after:.2f}%)")

    aod_eod_after = calculate_aod_eod(
        y_true=y_test_numpy,
        y_pred=y_pred_after,
        sensitive=y_gen_test_encoded,
        privileged_value=gender_map.get('Male', 1),
    )
    logger.info(f"Final AOD: {aod_eod_after['aod']:.6f}" if not np.isnan(aod_eod_after['aod']) else "Final AOD: N/A")
    logger.info(f"Final EOD: {aod_eod_after['eod']:.6f}" if not np.isnan(aod_eod_after['eod']) else "Final EOD: N/A")

    # -------------------------------------------------------------------------
    # 8. Calculate Improvement Ratios
    # -------------------------------------------------------------------------
    # Calculate accuracy change ratio: 100 * (acc_before - acc_after) / acc_before
    # Note: acc_before and acc_after are already proportions (0-1)
    
    if acc_before > 0:
        accuracy_change_ratio = 100 * (acc_before - acc_after) / acc_before
    else:
        accuracy_change_ratio = np.nan
        logger.warning("Cannot calculate accuracy change ratio: baseline accuracy is 0")
    
    # Calculate discrimination change ratio: 100 * (disc_count_before - disc_count_after) / disc_count_before
    if disc_count_before > 0:
        discrimination_change_ratio = 100 * (disc_count_before - disc_count_after) / disc_count_before
    else:
        discrimination_change_ratio = np.nan
        logger.warning("Cannot calculate discrimination change ratio: baseline discrimination count is 0")
    
    logger.info(f"\nImprovement Ratios:")
    if not np.isnan(accuracy_change_ratio):
        logger.info(f"  Accuracy Change Ratio: {accuracy_change_ratio:.2f}%")
    else:
        logger.info(f"  Accuracy Change Ratio: N/A (baseline accuracy is 0)")
    
    if not np.isnan(discrimination_change_ratio):
        logger.info(f"  Discrimination Change Ratio: {discrimination_change_ratio:.2f}%")
    else:
        logger.info(f"  Discrimination Change Ratio: N/A (baseline discrimination count is 0)")

    # -------------------------------------------------------------------------
    # 9. Summary & Return
    # -------------------------------------------------------------------------
    aod_change_pct = _pct_change(aod_eod_before['aod'], aod_eod_after['aod'])
    eod_change_pct = _pct_change(aod_eod_before['eod'], aod_eod_after['eod'])

    logger.info("\n" + "=" * 80)
    logger.info("SUMMARY (Requested Ordering)")
    logger.info("Before fix:")
    logger.info(f"  Accuracy: {acc_before:.4%}")
    logger.info(f"  Classwise Accuracy (Class 0): {acc_class_0:.4%}" if not np.isnan(acc_class_0) else "  Classwise Accuracy (Class 0): N/A")
    logger.info(f"  Classwise Accuracy (Class 1): {acc_class_1:.4%}" if not np.isnan(acc_class_1) else "  Classwise Accuracy (Class 1): N/A")
    logger.info(f"  Avg Prob (Pred Class 0): {stats_test_before['avg_low']:.6f}" if not np.isnan(stats_test_before['avg_low']) else "  Avg Prob (Pred Class 0): N/A")
    logger.info(f"  Avg Prob (Pred Class 1): {stats_test_before['avg_high']:.6f}" if not np.isnan(stats_test_before['avg_high']) else "  Avg Prob (Pred Class 1): N/A")
    logger.info(f"  SVM Accuracy (Gender) Test: {svm_acc_test_before:.4%}")
    logger.info(f"  SVM Classwise Accuracy (Female): {svm_acc_test_female_before:.4%}" if not np.isnan(svm_acc_test_female_before) else "  SVM Classwise Accuracy (Female): N/A")
    logger.info(f"  SVM Classwise Accuracy (Male):   {svm_acc_test_male_before:.4%}" if not np.isnan(svm_acc_test_male_before) else "  SVM Classwise Accuracy (Male):   N/A")
    logger.info(f"  Counterfactual Fairness (Discrimination %): {disc_pct_before:.2f}%")
    logger.info(f"  AOD: {aod_eod_before['aod']:.6f}" if not np.isnan(aod_eod_before['aod']) else "  AOD: N/A")
    logger.info(f"  EOD: {aod_eod_before['eod']:.6f}" if not np.isnan(aod_eod_before['eod']) else "  EOD: N/A")

    logger.info("After fix:")
    logger.info(f"  Accuracy: {acc_after:.4%}")
    logger.info(f"  Classwise Accuracy (Class 0): {acc_class_0_after:.4%}" if not np.isnan(acc_class_0_after) else "  Classwise Accuracy (Class 0): N/A")
    logger.info(f"  Classwise Accuracy (Class 1): {acc_class_1_after:.4%}" if not np.isnan(acc_class_1_after) else "  Classwise Accuracy (Class 1): N/A")
    logger.info(f"  Avg Prob (Pred Class 0): {stats_test_after['avg_low']:.6f}" if not np.isnan(stats_test_after['avg_low']) else "  Avg Prob (Pred Class 0): N/A")
    logger.info(f"  Avg Prob (Pred Class 1): {stats_test_after['avg_high']:.6f}" if not np.isnan(stats_test_after['avg_high']) else "  Avg Prob (Pred Class 1): N/A")
    logger.info(f"  SVM Accuracy (Gender) Test: {svm_acc_test_after:.4%}")
    logger.info(f"  SVM Classwise Accuracy (Female): {svm_acc_test_female_after:.4%}" if not np.isnan(svm_acc_test_female_after) else "  SVM Classwise Accuracy (Female): N/A")
    logger.info(f"  SVM Classwise Accuracy (Male):   {svm_acc_test_male_after:.4%}" if not np.isnan(svm_acc_test_male_after) else "  SVM Classwise Accuracy (Male):   N/A")
    logger.info(f"  Counterfactual Fairness (Discrimination %): {disc_pct_after:.2f}%")
    logger.info(f"  AOD: {aod_eod_after['aod']:.6f}" if not np.isnan(aod_eod_after['aod']) else "  AOD: N/A")
    logger.info(f"  EOD: {aod_eod_after['eod']:.6f}" if not np.isnan(aod_eod_after['eod']) else "  EOD: N/A")

    logger.info("Change before and after fix:")
    logger.info(f"  Accuracy (% change): {_pct_change(acc_before, acc_after):.2f}%" if not np.isnan(_pct_change(acc_before, acc_after)) else "  Accuracy (% change): N/A")
    logger.info(f"  Class 0 Accuracy (% change): {_pct_change(acc_class_0, acc_class_0_after):.2f}%" if not np.isnan(_pct_change(acc_class_0, acc_class_0_after)) else "  Class 0 Accuracy (% change): N/A")
    logger.info(f"  Class 1 Accuracy (% change): {_pct_change(acc_class_1, acc_class_1_after):.2f}%" if not np.isnan(_pct_change(acc_class_1, acc_class_1_after)) else "  Class 1 Accuracy (% change): N/A")
    logger.info(f"  Avg Prob (Pred Class 0) (% change): {_pct_change(stats_test_before['avg_low'], stats_test_after['avg_low']):.2f}%" if not np.isnan(_pct_change(stats_test_before['avg_low'], stats_test_after['avg_low'])) else "  Avg Prob (Pred Class 0) (% change): N/A")
    logger.info(f"  Avg Prob (Pred Class 1) (% change): {_pct_change(stats_test_before['avg_high'], stats_test_after['avg_high']):.2f}%" if not np.isnan(_pct_change(stats_test_before['avg_high'], stats_test_after['avg_high'])) else "  Avg Prob (Pred Class 1) (% change): N/A")
    logger.info(f"  SVM Accuracy (% change): {_pct_change(svm_acc_test_before, svm_acc_test_after):.2f}%" if not np.isnan(_pct_change(svm_acc_test_before, svm_acc_test_after)) else "  SVM Accuracy (% change): N/A")
    logger.info(f"  SVM Female (% change): {_pct_change(svm_acc_test_female_before, svm_acc_test_female_after):.2f}%" if not np.isnan(_pct_change(svm_acc_test_female_before, svm_acc_test_female_after)) else "  SVM Female (% change): N/A")
    logger.info(f"  SVM Male (% change):   {_pct_change(svm_acc_test_male_before, svm_acc_test_male_after):.2f}%" if not np.isnan(_pct_change(svm_acc_test_male_before, svm_acc_test_male_after)) else "  SVM Male (% change):   N/A")
    logger.info(f"  Counterfactual Fairness (Discrimination %) (% change): {_pct_change(disc_pct_before, disc_pct_after):.2f}%" if not np.isnan(_pct_change(disc_pct_before, disc_pct_after)) else "  Counterfactual Fairness (Discrimination %) (% change): N/A")
    logger.info(f"  AOD (% change): {aod_change_pct:.2f}%" if not np.isnan(aod_change_pct) else "  AOD (% change): N/A")
    logger.info(f"  EOD (% change): {eod_change_pct:.2f}%" if not np.isnan(eod_change_pct) else "  EOD (% change): N/A")
    logger.info("=" * 80)

    results = {
        'baseline': {
            'income_accuracy': acc_before,
            'discrimination_count': disc_count_before,
            'discrimination_pct': disc_pct_before,
            'gender_svm_acc_test': svm_acc_test_before,
            'gender_svm_acc_test_female': svm_acc_test_female_before,
            'gender_svm_acc_test_male': svm_acc_test_male_before,
            'avg_prob_pred_class0': stats_test_before['avg_low'],
            'avg_prob_pred_class1': stats_test_before['avg_high'],
            'aod': aod_eod_before['aod'],
            'eod': aod_eod_before['eod']
        },
        'repaired': {
            'income_accuracy': acc_after,
            'discrimination_count': disc_count_after,
            'discrimination_pct': disc_pct_after,
            'gender_svm_acc_test_projected': svm_acc_test_after,
            'gender_svm_acc_test_projected_female': svm_acc_test_female_after,
            'gender_svm_acc_test_projected_male': svm_acc_test_male_after,
            'avg_prob_pred_class0': stats_test_after['avg_low'],
            'avg_prob_pred_class1': stats_test_after['avg_high'],
            'aod': aod_eod_after['aod'],
            'eod': aod_eod_after['eod']
        },
        'improvement_ratios': {
            'accuracy_change_ratio_pct': accuracy_change_ratio,
            'discrimination_change_ratio_pct': discrimination_change_ratio,
            'aod_change_ratio_pct': aod_change_pct,
            'eod_change_ratio_pct': eod_change_pct
        },
        'svm_model': gender_svm,
        'projection_matrix': P
    }
    
    end_time = datetime.now()
    duration = end_time - start_time
    total_seconds = duration.total_seconds()
    minutes = int(total_seconds // 60)
    seconds = int(total_seconds % 60)
    logger.info(f"Process end time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Total execution time: {minutes} minutes {seconds} seconds ({total_seconds:.2f} seconds)")
    logger.info("="*80)
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 1: Bias Analysis and INLP Repair")
    parser.add_argument("--config", required=True, help="Path to config file")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--inlp_iters", type=int, default=5, help="Number of INLP iterations")
    parser.add_argument("--inlp_alpha", type=float, default=1.0, help="INLP projection strength (alpha)")
    args = parser.parse_args()
    
    with open(args.config) as f:
        config = json.load(f)
    
    params = Params(config)
    # batch_size must be specified in the config file 
    if not hasattr(params, 'batch_size') or params.batch_size is None:
        raise ValueError("batch_size must be specified in the config file")
    params.inlp_iters = args.inlp_iters
    params.inlp_alpha = args.inlp_alpha
    
    # Path resolution logic 
    project_root = Path(__file__).resolve().parent
    def resolve_path(path_str):
        if os.path.isabs(path_str): return path_str
        return str(project_root / path_str)
    
    params.dataset_dir = resolve_path(params.dataset_dir)
    params.ml_models_dir = resolve_path(params.ml_models_dir)
    if params.train_file_path and not os.path.isabs(params.train_file_path):
        params.train_file_path = str(Path(params.dataset_dir) / params.train_file_path)
    if params.test_file_path and not os.path.isabs(params.test_file_path):
        params.test_file_path = str(Path(params.dataset_dir) / params.test_file_path)
    
    os.makedirs(params.ml_models_dir, exist_ok=True)
    _, device = initialize_seed(args.seed)
    
    # Generate timestamp for all files (format: YYYYMMDD_HH_MM_SS)
    start_time = datetime.now()
    timestamp = start_time.strftime('%Y%m%d_%H_%M_%S')
    
    # Init logger
    log_path = project_root / "logs" / params.dataset_name / f"stage1_bias_analysis_{timestamp}_inlp_iters_{params.inlp_iters}_inlp_alpha_{params.inlp_alpha}.log"
    init_logger(log_path)
    
    main(params, device, timestamp)