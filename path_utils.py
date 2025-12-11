"""
Path utilities for consistent ML model directory management.
"""

import os
from pathlib import Path
from typing import Optional


def get_consistent_ml_models_path(dataset_name: str) -> str:
    """
      Get consistent ML models path for a dataset.
    """
    return f"ml_models/{dataset_name}"


def ensure_ml_models_directory(dataset_name: str) -> Path:
    """
      Ensure ML models directory exists for a dataset.
    """
    ml_models_dir = Path("ml_models") / dataset_name
    ml_models_dir.mkdir(parents=True, exist_ok=True)
    return ml_models_dir


def shorten_path_for_logging(file_path: str, run_dir: Optional[str] = None) -> str:
    """
      Shorten file paths for logging to make them more readable 
    """
    
    if not file_path:
        return file_path
    
    # Convert to string if it's a Path object
    file_path = str(file_path)
    
    # Get current working directory (should be fader_network)
    try:
        cwd = os.getcwd()
        cwd_name = os.path.basename(cwd)
    except OSError:
        # Fallback if we can't get current directory
        cwd_name = "fader_network"
    
    # If path doesn't start with /home/, check if it's already relative or starts with cwd
    if not file_path.startswith('/home/'):
        # If it's already a relative path or starts with the current directory name, return as is
        if not os.path.isabs(file_path) or file_path.startswith(cwd_name):
            return file_path
        # If it's an absolute path but doesn't start with /home/, it might be on a different drive
        # In this case, try to make it relative to current directory
        try:
            return os.path.relpath(file_path, cwd)
        except ValueError:
            # If we can't make it relative, return the basename
            return os.path.basename(file_path)
    
    # Look for the current directory name in the path
    if cwd_name in file_path:
        cwd_pos = file_path.find(cwd_name)
        if cwd_pos != -1:
            # Return from the current directory onwards
            return file_path[cwd_pos:]
    
    # Fallback: try to find common patterns in the path
    # Look for fader_counterfactual_runs, cvae_counterfactual_runs, or attribute_flipped_runs
    common_patterns = [
        'fader_counterfactual_runs',
        'cvae_counterfactual_runs', 
        'attribute_flipped_runs',
        'ml_models',
        'dataset',
        'config',
        'results',
        'logs'
    ]
    
    for pattern in common_patterns:
        if pattern in file_path:
            pattern_pos = file_path.find(pattern)
            if pattern_pos != -1:
                return file_path[pattern_pos:]
    
    # If no patterns found, return just the basename to avoid exposing full paths
    return os.path.basename(file_path)