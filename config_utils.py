import json
import os
from typing import Any, Dict, Optional, Tuple
import torch
from logger_utils import get_logger


logger = get_logger(__name__)


class Params:
    """
    The Params class is designed to take a configuration dictionary from a JSON file and turn it into an object with attribute-style access.
    """
    def __init__(self, config_dict: Dict[str, Any]):
        # Initialize common attributes to None to help with type checking
        self.sensitive_attr: Optional[str] = None
        self.lambda_adv: Optional[float] = None
        self.train_fader_network: Optional[str] = None
        self.dataset_name: Optional[str] = None
        self.output_column_name: Optional[str] = None
        self.train_file_path: Optional[str] = None
        self.test_file_path: Optional[str] = None
        self.batch_size: Optional[int] = None
        self.latent_dim: Any = None
        self.k_ED: Optional[int] = None
        self.encoder_depth: Optional[int] = None
        self.decoder_depth: Optional[int] = None
        self.hidden_dim: Optional[int] = None
        self.patience: Optional[int] = None
        self.warmup_epochs: Optional[int] = None
        self.ramp_epochs: Optional[int] = None
        self.eps: Optional[float] = None
        self.weight_decay_D: Optional[float] = None
        self.lambda_now: Optional[float] = None
        self.scaling_factor: Any = None
        self.encoder_dropout: Optional[float] = None
        self.decoder_dropout: Optional[float] = None
        self.encoder_decoder_learning_rate: Optional[float] = None
        self.discriminator_learning_rate: Optional[float] = None
        self.classifier_depth: Optional[int] = None
        self.classifier_dropout: Optional[float] = None
        self.regressor_depth: Optional[int] = None
        self.regressor_dropout: Optional[float] = None
        self.scheduler_step_size: Optional[int] = None
        self.scheduler_gamma: Optional[float] = None
        self.min_improvement: Optional[float] = None
        
        # Directories and paths
        self.t_way_files_root: Optional[str] = None
        self.ml_models_dir_root: Optional[str] = None
        self.t_way_run_dir: Optional[str] = None
        self.t_way_files_prediction_dir: Optional[str] = None
        self.ml_models_dir: Optional[str] = None
        self.counterfactual_dir: Optional[str] = None
        self.counterfactual_prediction_dir: Optional[str] = None
        self.results_dir: Optional[str] = None
        self.log_loss_dir: Optional[str] = None
        
        # Other computed attributes
        self.invalid_valid_testcases_stat_path: Optional[str] = None
        self.calculation_mode: Optional[str] = None
        self.aggregate_out: Optional[str] = None
        self.config_and_architecture_file: Any = None
        
        # Set attributes from config dictionary
        for key, value in config_dict.items():
            setattr(self, key, value)
        
    def __repr__(self):
        return f"{self.__class__.__name__}({self.__dict__})"
    
    def __setattr__(self, name: str, value: Any) -> None:
        """Allow setting any attribute dynamically."""
        super().__setattr__(name, value)
    
    def __getattr__(self, name: str) -> Any:
        """
        Return None for any missing attribute instead of raising AttributeError.
        """
        if name.startswith('__') and name.endswith('__'):
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
        return None
    
    
    
def load_config(config_path):
    """
    This function is loads a configuration file written in JSON format and converts it into a Params object.
    """
    with open(config_path, 'r') as f:
        config_dict = json.load(f)
    return Params(config_dict)
    
    
    
# Global flag to track seed initialization to prevent race conditions
_seed_initialized = False


def initialize_seed(seed_value: Optional[int] = None, gpu_id: Optional[int] = None) -> Tuple[int, torch.device]:
    """
    Seed initialization that calls comprehensive deterministic setup.
    """
    global _seed_initialized
    
    # Use deterministic helper functions
    if _seed_initialized:
        logger.info("Seed already initialized - skipping duplicate call")
        seed_value = int(os.getenv("GLOBAL_SEED", 42))
        # Get device from deterministic helper to maintain consistency
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
        else:
            device = torch.device("cpu")
        return seed_value, device
    
    _seed_initialized = True
    
    if seed_value is None:
        seed_value = int(os.getenv("GLOBAL_SEED", 42))

    # Importing here
    from deterministic_run_helper import initialize_deterministic_environment
    return initialize_deterministic_environment(seed_value, gpu_id)