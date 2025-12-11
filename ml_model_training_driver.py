import os
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import argparse
import json
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
import numpy as np

# internal utils
from preprocess import preprocess_data_for_fader_network
from logger_utils import get_logger, init_logger
from config_utils import Params, initialize_seed

logger = get_logger(__name__)
DETERMINISTIC_SEED = int(os.getenv("GLOBAL_SEED", 42))


class SimpleNeuralNetwork(nn.Module):
    """ Neural network with better architecture for tabular data."""
    def __init__(self, input_size):
        super(SimpleNeuralNetwork, self).__init__()
        # Architecture: input -> 256 -> 128 -> 64 -> 32 -> 1
        # Following good practices for tabular data
        self.layer1 = nn.Linear(input_size, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.dropout1 = nn.Dropout(0.3)
        
        self.layer2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.dropout2 = nn.Dropout(0.3)
        
        self.layer3 = nn.Linear(128, 64)
        self.bn3 = nn.BatchNorm1d(64)
        self.dropout3 = nn.Dropout(0.2)
        
        self.layer4 = nn.Linear(64, 32)
        self.bn4 = nn.BatchNorm1d(32)
        self.dropout4 = nn.Dropout(0.2)
        
        self.output_layer = nn.Linear(32, 1)
        
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x = self.dropout1(self.relu(self.bn1(self.layer1(x))))
        x = self.dropout2(self.relu(self.bn2(self.layer2(x))))
        x = self.dropout3(self.relu(self.bn3(self.layer3(x))))
        x = self.dropout4(self.relu(self.bn4(self.layer4(x))))
        x = self.sigmoid(self.output_layer(x))
        return x



def main(params, device):
    
    # paths
    
    MODEL_PATHS = {
        "logistic": os.path.join(params.ml_models_dir, f"{params.dataset_name}__logistic_regression__.pkl"),
        "rf":       os.path.join(params.ml_models_dir, f"{params.dataset_name}__random_forest__.pkl"),
        "svm":      os.path.join(params.ml_models_dir, f"{params.dataset_name}__svm__.pkl"),
        "nn":       os.path.join(params.ml_models_dir, f"{params.dataset_name}__nn__.pth"),
    }
    
    # Get simple train/test splits for standard ML training
    X_train, X_test, ht, y_train, y_test = preprocess_data_for_fader_network(params)        # Convert to PyTorch tensors for neural network
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

    # Train all models with default hyperparameters on training data
    logger.info("\n Training Logistic Regression...")
    lr_final = LogisticRegression(random_state=DETERMINISTIC_SEED, max_iter=1000)
    lr_final.fit(X_train, y_train)  # Train on training set only
    with open(MODEL_PATHS["logistic"], "wb") as f:
        pickle.dump(lr_final, f)
    acc_lr = accuracy_score(y_test, lr_final.predict(X_test))  # Evaluate on test set only
    logger.info(f"Logistic Regression  - Test accuracy: {acc_lr:.4f}")
    print(f"  - {os.path.basename(MODEL_PATHS['logistic'])}: Test={acc_lr:.3f}")

    # Random Forest
    logger.info("\n Training Random Forest...")
    rf_final = RandomForestClassifier(random_state=DETERMINISTIC_SEED)
    rf_final.fit(X_train, y_train)  # Train on training set only
    with open(MODEL_PATHS["rf"], "wb") as f:
        pickle.dump(rf_final, f)
    acc_rf = accuracy_score(y_test, rf_final.predict(X_test))  # Evaluate on test set only
    logger.info(f"Random Forest - Test accuracy: {acc_rf:.4f}")
    print(f"  - {os.path.basename(MODEL_PATHS['rf'])}: Test={acc_rf:.3f}")

    # SVM
    logger.info("\n Training SVM...")
    svm_final = SVC(random_state=DETERMINISTIC_SEED, probability=True)
    svm_final.fit(X_train, y_train)  # Train on training set only
    with open(MODEL_PATHS["svm"], "wb") as f:
        pickle.dump(svm_final, f)
    acc_svm = accuracy_score(y_test, svm_final.predict(X_test))  # Evaluate on test set only
    logger.info(f"SVM - Test accuracy: {acc_svm:.4f}")
    print(f"  - {os.path.basename(MODEL_PATHS['svm'])}: Test={acc_svm:.3f}")

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
    # code/ml_model_training.py is in code/, so parent.parent is project root
    project_root = Path(__file__).resolve().parent.parent

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