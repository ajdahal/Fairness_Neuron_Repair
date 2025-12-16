import os
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import argparse
import json
from pathlib import Path
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
import numpy as np

# internal utils
from preprocess import preprocess_data_for_ml_models
from logger_utils import get_logger, init_logger
from config_utils import Params, initialize_seed

logger = get_logger(__name__)
DETERMINISTIC_SEED = int(os.getenv("GLOBAL_SEED", 42))


class SimpleNeuralNetwork(nn.Module):
    """ Neural network with better architecture for tabular data."""
    def __init__(self, input_size):
        super(SimpleNeuralNetwork, self).__init__()
        # Architecture: input ->  258 -> 128 -> 64 -> 1
        self.layer1 = nn.Linear(input_size, 256)
        self.layer2 = nn.Linear(256, 128)
        self.layer3 = nn.Linear(128, 64)
        self.output_layer = nn.Linear(64, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.relu(self.layer3(x))
        x = self.sigmoid(self.output_layer(x))
        return x



def main(params, device):
    
    # paths
    MODEL_PATHS = {
        "nn": os.path.join(params.ml_models_dir, f"{params.dataset_name}__nn__.pth"),
    }
    
    # Get simple train/test splits for standard ML training
    X_train, X_test, ht, y_train, y_test = preprocess_data_for_ml_models(params)        # Convert to PyTorch tensors for neural network
    X_train_tensor = torch.FloatTensor(X_train.values if hasattr(X_train, 'values') else X_train)
    X_test_tensor = torch.FloatTensor(X_test.values if hasattr(X_test, 'values') else X_test)
    y_train_tensor = torch.FloatTensor(y_train.values if hasattr(y_train, 'values') else y_train)
    y_test_tensor = torch.FloatTensor(y_test.values if hasattr(y_test, 'values') else y_test)
    
    # Using scikit-learn default settings
    logger.info("="*80)
    logger.info("STANDARD 80/20 TRAIN/TEST ML METHODOLOGY")
    logger.info("="*80)
    logger.info("Step 1: HyperTransformer fitted on TRAINING set only")
    logger.info("Step 2: ML models trained on TRAINING set (80% of data)") 
    logger.info("Step 3: NO hyperparameter tuning - use reasonable defaults")
    logger.info("Step 4: Evaluate models on TEST set only (20% of data)")
    logger.info("="*80)
    
    logger.info(f"Training data size: {len(X_train)} (80%)")
    logger.info(f"Test data size: {len(X_test)} (20%)")
    logger.info("Using reasonable default hyperparameters (no tuning needed)")
    
    # Default hyperparameters (sklearn models use library defaults)
    default_nn_lr = 0.001
    
    input_size = X_train_tensor.shape[1]

    # Neural Network
    logger.info("\n Training Neural Network...")
    nn_final = SimpleNeuralNetwork(input_size).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(nn_final.parameters(), lr=default_nn_lr, weight_decay=1e-4)
    
    X_train_device = X_train_tensor.to(device)
    y_train_device = y_train_tensor.to(device)
    
    nn_final.train()
    for epoch in range(200):  # Train for 200 epochs
        optimizer.zero_grad()
        outputs = nn_final(X_train_device)
        loss = criterion(outputs.squeeze(), y_train_device)
        loss.backward()
        optimizer.step()
        
        if epoch % 20 == 0:
            logger.info(f"Neural Network epoch {epoch}, Loss: {loss.item():.6f}")
    
    torch.save(nn_final.state_dict(), MODEL_PATHS["nn"])
    
    nn_final.eval()
    with torch.no_grad():
        y_pred_probs = nn_final(X_test_tensor.to(device)).cpu().numpy().flatten()
        nn_preds = (y_pred_probs > 0.5).astype(int)
    acc_nn = accuracy_score(y_test, nn_preds)  # Evaluate on test set only
    logger.info(f"Neural Network - learning_rate: {default_nn_lr}, Test accuracy: {acc_nn:.4f}")
    print(f"  - {os.path.basename(MODEL_PATHS['nn'])}: Test={acc_nn:.3f}")
    
    logger.info("="*80)
    logger.info(" STANDARD TRAINING COMPLETED")
    logger.info("="*80)
    logger.info(f"Models saved in: {params.ml_models_dir}")
    logger.info("Summary:")
    logger.info("  HyperTransformer fitted on training data only")
    logger.info("  ML models trained on training data (80%)")
    logger.info("  No hyperparameter tuning (reasonable defaults used)")
    logger.info("  Final evaluation performed on test set only (20%)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    with open(args.config) as f:
        config = json.load(f)

    params = Params(config)

    # Resolve basic paths (relative to project root if not absolute)
    project_root = Path(__file__).resolve().parent

    # Helper to make path absolute relative to project root if it isn't already
    def resolve_path(path_str):
        if os.path.isabs(path_str):
            return path_str
        return str(project_root / path_str)

    params.dataset_dir = resolve_path(params.dataset_dir)
    params.ml_models_dir = resolve_path(params.ml_models_dir)

    # train/test files are usually relative to dataset_dir
    if params.train_file_path and not os.path.isabs(params.train_file_path):
        params.train_file_path = str(Path(params.dataset_dir) / params.train_file_path)
        
    if params.test_file_path and not os.path.isabs(params.test_file_path):
        params.test_file_path = str(Path(params.dataset_dir) / params.test_file_path)

    os.makedirs(params.ml_models_dir, exist_ok=True)
    
    _, device = initialize_seed(args.seed)
    
    # Init logger
    log_path = project_root / "logs" / "ml_model_training.log"
    init_logger(log_path)

    main(params, device)