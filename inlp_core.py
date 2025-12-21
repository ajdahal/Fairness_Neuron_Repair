import numpy as np
import torch
from sklearn.svm import LinearSVC
from logger_utils import get_logger

logger = get_logger(__name__)

def get_projection_and_svm(X_train, Y_train_gen, X_val, Y_val_gen, n_iter, alpha=1.0):
    """
    Iterative Nullspace Projection (INLP).
    This function trains an initial SVM to detect bias, and then iteratively removes bias by projecting onto the nullspace of the bias direction.
    """
    # Centering data
    mean_vector = np.mean(X_train, axis=0, keepdims=True)
    X_train = X_train - mean_vector
    input_dim = X_train.shape[1]
    
    # Train the initial classifier (Bias Detector) to evaluate bias before/after projection
    logger.info("Training Initial Gender Detector (SVM)...")
    original_svm = LinearSVC(fit_intercept=False, class_weight='balanced', max_iter=5000, dual='auto', random_state=42)
    original_svm.fit(X_train, Y_train_gen)
    
    # INLP Loop: iteratively remove bias by projecting onto the nullspace of the bias direction
    P = np.eye(input_dim)
    
    # Use provided validation set (centered with same mean as training data)
    X_curr_train = X_train.copy()
    X_curr_dev = X_val - mean_vector
    Y_train_inlp = Y_train_gen
    Y_dev_inlp = Y_val_gen
    logger.info(f"Using validation set: {len(X_curr_train)} train, {len(X_curr_dev)} val")
    
    logger.info(f"Starting INLP (n_iter={n_iter}, alpha={alpha})...")
    recent_accs = []  # Track recent accuracies for stability check
    
    for i in range(n_iter):
        svm_iter = LinearSVC(fit_intercept=False, class_weight='balanced', max_iter=2000, dual='auto', random_state=42)
        svm_iter.fit(X_curr_train, Y_train_inlp)
        
        # Evaluate on held-out dev set
        acc = svm_iter.score(X_curr_dev, Y_dev_inlp)
        logger.info(f"Iter {i+1}: SVM Acc = {acc:.4f}")
        
        # Check if accuracy stopped improving (stable for 3 iterations)
        recent_accs.append(acc)
        if len(recent_accs) > 3:
            recent_accs.pop(0)  # Keep only last 3 accuracies
        
        # Converge if: (1) acc near random chance OR (2) accuracy plateau
        if len(recent_accs) >= 2 and all(a <= 0.55 for a in recent_accs):
            logger.info("Converged (random chance reached and stable).")
            break
        
        
        # Get the weight vector (direction of bias) and normalize it to get the unit vector
        w = svm_iter.coef_[0]
        w_norm = np.linalg.norm(w)
        if w_norm == 0:
            break
        
        u = w / w_norm  # Unit vector in direction of bias
        
        # Projection matrix: P_step = I - u u^T
        P_step = np.eye(input_dim) - alpha * np.outer(u, u)
        
        # Update total projection P = P @ P_step (order matters: apply projections in sequence)
        P = P @ P_step
        
        # Project data for next iteration: X_new = X_curr @ P_step.T
        # Note: Since P_step is symmetric, P_step.T == P_step
        X_curr_train = X_curr_train @ P_step.T
        X_curr_dev = X_curr_dev @ P_step.T
    
    # Enhancement 1: Add projection matrix analysis
    logger.info("\n--- Projection Matrix Analysis ---")
    P_rank = np.linalg.matrix_rank(P)
    P_symmetry_error = np.linalg.norm(P - P.T)
    logger.info(f"Projection matrix rank: {P_rank}/{input_dim}")
    logger.info("=" * 50)
    
    return P, original_svm, mean_vector


def perform_weight_surgery(model, P, projection_mean=None):
    """
    Applies the projection matrix P to the model's output layer weights.
    Optionally adjusts bias using projection_mean if data was centered during P computation.
    
    Since P was trained on centered activations (X - mu), but the model uses uncentered
    activations X, we compensate by adjusting the bias: b_new = b - W_new @ mu
    This ensures: output_new = W_new @ X + b_new = W_old @ P @ (X - mu) + b
    """
    logger.info("\n--- Performing Weight Surgery ---")
    with torch.no_grad():
        # Get original weights [Out_dim, In_dim] -> [1, 64]
        W_old = model.output_layer.weight.data
        
        # Convert P to tensor
        P_tensor = torch.tensor(P, dtype=torch.float32).to(W_old.device)
        
        # Enhancement 4: Better logging (shape and norms instead of full matrices)
        logger.info(f"W_old shape: {W_old.shape}, norm: {torch.norm(W_old).item():.6f}")
        logger.info(f"P_tensor shape: {P_tensor.shape}, norm: {torch.norm(P_tensor).item():.6f}")
        logger.info(f"P_tensor symmetry error: {torch.norm(P_tensor - P_tensor.T).item():.2e}")
        
        # Apply projection: W_new = W_old @ P
        # This works because: Output = W_old @ (P @ x) = (W_old @ P) @ x
        W_new = torch.matmul(W_old, P_tensor)
        
        # Enhancement 4: Add sanity checks before updating weights
        assert W_new.shape == W_old.shape, f"Weight shape changed: {W_old.shape} -> {W_new.shape}"
        assert not torch.isnan(W_new).any(), "NaN weights detected after projection!"
        assert not torch.isinf(W_new).any(), "Inf weights detected after projection!"
        
        # Update model weights
        model.output_layer.weight.data = W_new
        logger.info(f"Weights updated successfully. W_new shape: {W_new.shape}, norm: {torch.norm(W_new).item():.6f}")
        logger.info(f"Weight change magnitude: {torch.norm(W_new - W_old).item():.6f}")
        
        # Adjust bias if mean was subtracted during projection training:
        # Since P was trained on centered data (X - mu), but model uses uncentered X,
        # we need to adjust bias to compensate:
        # Desired: output = W_old @ P @ (X - mu) + b = W_old @ P @ X - W_old @ P @ mu + b
        # After surgery: output = W_new @ X + b_new = (W_old @ P) @ X + b_new
        # Therefore: b_new = b - W_new @ mu (where W_new = W_old @ P)
        if projection_mean is not None:
            mu_tensor = torch.tensor(projection_mean.flatten(), dtype=torch.float32).to(W_old.device).reshape(-1, 1)
            bias_correction = torch.matmul(W_new, mu_tensor).squeeze()
            
            # Enhancement 4: Add sanity checks for bias correction
            assert not torch.isnan(bias_correction).any(), "NaN bias correction detected!"
            assert not torch.isinf(bias_correction).any(), "Inf bias correction detected!"
            
            if model.output_layer.bias is not None:
                b_old_norm = torch.norm(model.output_layer.bias.data).item()
                model.output_layer.bias.data -= bias_correction
                b_new_norm = torch.norm(model.output_layer.bias.data).item()
                logger.info(f"Bias updated to account for data centering.")
                logger.info(f"  Bias correction magnitude: {torch.norm(bias_correction).item():.6f}, Bias norm: {b_old_norm:.6f} -> {b_new_norm:.6f}")
            else:
                logger.warning(" Model has no bias term, but projection_mean was provided. Bias correction skipped. It breaks the model.")
