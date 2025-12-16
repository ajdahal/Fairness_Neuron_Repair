# data/preprocess.py
import os
import pickle
import pandas as pd
import numpy as np
from rdt import HyperTransformer
from rdt.transformers import GaussianNormalizer, OneHotEncoder
from sklearn.model_selection import train_test_split

from logger_utils import get_logger
from path_utils import shorten_path_for_logging

logger = get_logger(__name__)

def load_hypertransformer(params):
    """
    Load a saved HyperTransformer from a pickle file.
    """
    if not hasattr(params, 'ml_models_dir') or not params.ml_models_dir:
        raise ValueError("params.ml_models_dir is required but not provided")
    
    if not hasattr(params, 'train_file_path') or not params.train_file_path:
        raise ValueError("params.train_file_path is required but not provided")
    
    ht_file_path = os.path.join(params.ml_models_dir, f"{os.path.splitext(os.path.basename(params.train_file_path))[0]}_hypertransformer.pkl")
    
    if not os.path.exists(ht_file_path):
        raise FileNotFoundError(f"HyperTransformer file not found: {ht_file_path}")
    
    try:
        with open(ht_file_path, "rb") as f:
            ht = pickle.load(f)
    except (pickle.UnpicklingError, EOFError) as e:
        raise RuntimeError(f"Failed to load HyperTransformer from {ht_file_path} - file may be corrupted: {e}")
    except (FileNotFoundError, PermissionError) as e:
        raise FileNotFoundError(f"Cannot access HyperTransformer file {ht_file_path}: {e}")
    except Exception as e:
        raise RuntimeError(f"Unexpected error loading HyperTransformer from {ht_file_path}: {e}")
    
    # Validate that the loaded object is a HyperTransformer
    if not hasattr(ht, 'transform') or not hasattr(ht, 'reverse_transform'):
        raise RuntimeError(f"Loaded object from {ht_file_path} is not a valid HyperTransformer (missing required methods)")
    
    logger.info(f"Successfully loaded HyperTransformer from: {shorten_path_for_logging(ht_file_path)}")
    return ht


def initialize_hypertransformer(params):
    
    ht = HyperTransformer()
    # Set deterministic seed for HyperTransformer to ensure reproducibility
    seed_value = int(os.getenv("GLOBAL_SEED", 42))
    ht.random_state = seed_value
    logger.debug(f"HyperTransformer initialized with deterministic seed: {seed_value}")
    
    # The configuration might store transformer names as strings. 
    # You may need a mapping to convert strings to transformer objects.
    transformer_mapping = {
        "GaussianNormalizer": GaussianNormalizer,
        "OneHotEncoder": OneHotEncoder
    }
    
    # Convert transformer strings to actual objects.
    transformers = {}
    sdtypes = {}
    
    # Get the target column name to exclude it
    target_column = getattr(params, 'output_column_name', None)
    
    for key, val in params.hypertransformer_config["transformers"].items():
        # Skip the target column as it won't be in X_train
        if key == target_column:
            continue
            
        if val in transformer_mapping:
            transformers[key] = transformer_mapping[val]()          # Initialize a new object each time
        else:
            raise ValueError(f"Transformer {val} for column {key} is not recognized.")
    
    # Also exclude target column from sdtypes
    for key, val in params.hypertransformer_config["sdtypes"].items():
        if key == target_column:
            continue
        sdtypes[key] = val
    
    # Update the hypertransformer config with transformer objects.
    config_ht = params.hypertransformer_config.copy()
    config_ht["transformers"] = transformers
    config_ht["sdtypes"] = sdtypes
    ht.set_config(config_ht)
    return ht


def preprocess_data_for_ml_models(params):
    """
    Preprocess data specifically for ML Network training with 80/20 train/test split.
    This function loads the pre-split data and doesn't balance sensitive attributes.
    """
    logger.info("Preprocessing data for ML Network training")
    logger.info("Loading pre-split train/test data created by download_split_dataset_router")
    
    train_file = params.train_file_path
    test_file = params.test_file_path
    
    # Validate that all required files exist
    if not os.path.exists(train_file):
        raise FileNotFoundError(f"Training file not found: {train_file}. Run download_split_dataset_router first.")
    if not os.path.exists(test_file):
        raise FileNotFoundError(f"Test file not found: {test_file}. Run download_split_dataset_router first.")
    
    # Load the pre-split datasets
    train_data = pd.read_csv(train_file)
    test_data = pd.read_csv(test_file)
    
    logger.info(f"Data loaded - Train: {len(train_data)}, Test: {len(test_data)}")
    
    # Drop the target column based on the configuration.
    output_column = params.output_column_name
    if output_column:
        X_train = train_data.drop(columns=[output_column])
        X_test = test_data.drop(columns=[output_column])
        Y_train = train_data[output_column]
        Y_test = test_data[output_column]
    else:
        X_train = train_data
        X_test = test_data
        Y_train = None
        Y_test = None
    
    # Initialize and fit HyperTransformer on features only (X_train, without output column)
    logger.info("Initializing and fitting HyperTransformer on training features...")
    ht = initialize_hypertransformer(params)
    ht.fit(X_train)
    logger.info("HyperTransformer fitted successfully on training features")
    
    ht_file_path = os.path.join(params.ml_models_dir, f"{os.path.splitext(os.path.basename(params.train_file_path))[0]}_hypertransformer.pkl")
    
    os.makedirs(params.ml_models_dir, exist_ok=True)
    with open(ht_file_path, "wb") as f:
        pickle.dump(ht, f)
    logger.info(f"HyperTransformer saved to: {shorten_path_for_logging(ht_file_path)}")
    
    logger.info("Skipping sensitive attribute balancing for Fader Network to ensure consistent comparison with ML models.")
    logger.info(f"X_train length: {len(X_train)}, column headers (original distribution): {X_train.columns}")
    
    # Load the existing hypertransformer from ML training stage (DO NOT retrain)
    logger.info("Loading existing HyperTransformer from ML training stage...")
    ht = load_hypertransformer(params)
    logger.info("Successfully loaded pre-trained HyperTransformer - no retraining needed")
    
    # Patch test data to include all categories from train
    categorical_columns = [col for col, typ in params.hypertransformer_config["sdtypes"].items() if typ == "categorical"]
    
    # Handle test set  
    X_test_patched, test_dummy_indices = patch_test_categories(X_train, X_test, categorical_columns, data_type="test")
    
    # Transform all datasets
    X_train_transformed = ht.transform(X_train)
    X_test_transformed = ht.transform(X_test_patched)
    
    # Drop dummy rows after transformation
    if test_dummy_indices:
        X_test_transformed = X_test_transformed.drop(index=test_dummy_indices).reset_index(drop=True)
    
    logger.info(f"X_train_transformed column headers: {X_train_transformed.columns}")
    
    return X_train_transformed, X_test_transformed, ht, Y_train, Y_test



def patch_test_categories(X_train, X_test, categorical_columns, data_type="test"):
    """
    For each categorical column, add a dummy row for categories present in train but missing in test.
    Uses data from the TEST set for dummy rows 
    Returns the patched test set and a list of indices of the dummy rows - these are dropped after transformation.
    """
    patched_test = X_test.copy()
    dummy_indices = []
    missing_categories_info = {}
    
    # First check for missing categories and build info dictionary
    for col in categorical_columns:
        train_cats = set(X_train[col].unique())
        test_cats = set(X_test[col].unique())
        missing_cats = train_cats - test_cats
        if missing_cats:
            missing_categories_info[col] = len(missing_cats)
            logger.info(f"Column {col} has {len(missing_cats)} categories in training data that are missing in {data_type} data")
            logger.debug(f"Missing categories in {col} ({data_type}): {missing_cats}")
            
            # Add dummy rows using test data as template - will drop later
            if len(X_test) > 0:  # Make sure test set is not empty
                template_row = X_test.iloc[0].copy()  # Use first test row as template
                for cat in missing_cats:
                    dummy_row = template_row.copy()
                    dummy_row[col] = cat
                    patched_test = pd.concat([patched_test, pd.DataFrame([dummy_row])], ignore_index=True)
                    dummy_indices.append(patched_test.index[-1])
    
    # Log summary of patches if any were made
    if missing_categories_info:
        logger.info(f"Added {len(dummy_indices)} dummy rows to {data_type} data for categories missing in {data_type} set")
        for col, count in missing_categories_info.items():
            logger.info(f"  - {col}: {count} categories")
    
    return patched_test, dummy_indices