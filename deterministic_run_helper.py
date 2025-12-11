import os
import random
import numpy as np
import pandas as pd
import torch
import psutil
import gc
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Optional, Tuple
import shutil

# Ensure the current directory is in the path for imports
import sys
CODE_DIR = Path(__file__).parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

# Import from modules in the same directory (code/)
from config_utils import Params
from logger_utils import get_logger
from path_utils import shorten_path_for_logging


logger = get_logger(__name__)

# Global flag to track deterministic environment initialization
_deterministic_environment_initialized = False


def initialize_deterministic_environment(seed_value: Optional[int] = None, gpu_id: Optional[int] = None):
    """
        Initialize a completely deterministic environment for reproducible ML results.
        This combines seed initialization and performance optimizations for determinism.
    """
    global _deterministic_environment_initialized
    
    if seed_value is None:
        seed_value = int(os.getenv("GLOBAL_SEED", 42))
    
    if _deterministic_environment_initialized:
        logger.info(f"Deterministic environment already initialized with seed={seed_value}")
        device = _setup_deterministic_gpu()
        return seed_value, device
    
    _deterministic_environment_initialized = True
    
    logger.info(f"Initializing deterministic environment with seed={seed_value}, gpu_id={gpu_id}")
    _set_deterministic_seeds(seed_value)
    _set_deterministic_environment()
    _configure_deterministic_frameworks(seed_value)
    _apply_deterministic_performance_optimizations()
    device = _setup_deterministic_gpu()
    _log_deterministic_configuration(seed_value, device)
    
    return seed_value, device


def _set_deterministic_seeds(seed_value: int):
    """Set seeds for all relevant libraries."""
    # Environment variable seeds (set once, no duplicates)
    os.environ['GLOBAL_SEED'] = str(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    os.environ['SKLEARN_SEED'] = str(seed_value)
    os.environ['PANDAS_SEED'] = str(seed_value)
    os.environ['SDV_SEED'] = str(seed_value)
    os.environ['RDT_SEED'] = str(seed_value)

    # Core Python seeds
    random.seed(seed_value)
    np.random.seed(seed_value)



def _set_deterministic_environment():
    """Configure environment variables for determinism."""
    # Core determinism settings
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['NUMEXPR_NUM_THREADS'] = '1'
    os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    os.environ['NUMBA_NUM_THREADS'] = '1'
    os.environ['NUMBA_DISABLE_JIT'] = '1'
    os.environ['KMP_AFFINITY'] = 'compact,1,0'



def _configure_deterministic_frameworks(seed_value: int):
    """Configure PyTorch for deterministic execution."""
    # PyTorch absolute determinism
    try:
        torch.manual_seed(seed_value)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed_value)
            torch.cuda.manual_seed_all(seed_value)
            # Disable optimizations that can introduce non-determinism
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True, warn_only=False)
        # Set worker timeout to avoid timing-based non-determinism
        torch.multiprocessing.set_sharing_strategy('file_system')
        logger.info("PyTorch configured for absolute determinism")
    except ImportError:
        logger.warning("PyTorch not available for deterministic configuration")


def _apply_deterministic_performance_optimizations():
    """Apply CPU, memory, and GPU optimizations for deterministic execution."""
    # CPU affinity for determinism
    try:
        process = psutil.Process()
        # Use highest numbered core (least likely to have system processes)
        available_cores = list(range(psutil.cpu_count()))
        isolated_core = available_cores[-1] if len(available_cores) > 1 else 0
        process.cpu_affinity([isolated_core])
        logger.info(f"deterministic mode: cpu affinity pinned to isolated core {isolated_core} (total cores: {len(available_cores)})")
        
        # Set process priority to normal for consistency
        process.nice(0)  # Normal priority
        logger.info("deterministic mode: Process priority set to normal")
    except Exception as e:
        logger.warning(f"Could not set cpu affinity/priority for determinism: {e}")
    
    # Memory settings for determinism
    try:
        # Force garbage collection for consistent memory state
        gc.collect()
        
        if torch.cuda.is_available():
            # Disable memory caching for consistent allocation patterns
            torch.cuda.empty_cache()
            # Set memory fraction to be deterministic
            torch.cuda.set_per_process_memory_fraction(0.4)  # Use 40% consistently
            logger.info("deterministic mode: gpu memory fraction set to 0.4")
    except Exception as e:
        logger.warning(f"Could not optimize memory settings for determinism: {e}")
    
    # gpu optimizations for determinism
    if torch.cuda.is_available():
        try:
            # Disable async memory operations for determinism
            os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
            
            # Force synchronous execution
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            
            # Disable optimizations that can introduce non-determinism
            if hasattr(torch.backends.cuda.matmul, 'allow_tf32'):
                torch.backends.cuda.matmul.allow_tf32 = False
            if hasattr(torch.backends.cudnn, 'allow_tf32'):
                torch.backends.cudnn.allow_tf32 = False
            
            logger.info("deterministic mode: gpu configured for determinism")
        except Exception as e:
            logger.warning(f"Could not configure gpu for determinism: {e}")


def validate_and_set_gpu_early(gpu_arg="--gpu_number", sys_argv=None):
    """
        Early gpu validation and setup before PyTorch import to ensure deterministic gpu usage.
    """
    if sys_argv is None:
        sys_argv = sys.argv
        
    if gpu_arg not in sys_argv:
        return  # No gpu specified, use default behavior
    
    try:
        gpu_index = sys_argv.index(gpu_arg) + 1
        
        # Validation 1: Check if next argument exists
        if gpu_index >= len(sys_argv):
            print(f"error: {gpu_arg} requires a gpu number argument")
            sys.exit(1)
        
        gpu_value = sys_argv[gpu_index]
        
        # Validation 2: Check if it's a valid integer
        try:
            gpu_number = int(gpu_value)
        except ValueError:
            print(f"error: gpu number must be an integer, got '{gpu_value}'")
            sys.exit(1)
        
        # Validation 3: Check if gpu number is non-negative
        if gpu_number < 0:
            print(f"error: gpu number must be non-negative, got {gpu_number}")
            sys.exit(1)
        
        # Set CUDA_VISIBLE_DEVICES to the validated gpu
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_number)
        print(f"CUDA_VISIBLE_DEVICES set to '{gpu_number}' (validated)")
        
    except ValueError as e:
        print(f"error: Failed to parse gpu argument: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"error: Unexpected error during gpu validation: {e}")
        sys.exit(1)


def _setup_deterministic_gpu() -> torch.device:
    """
    Configure gpu for deterministic execution.
    """
    if not torch.cuda.is_available():
        logger.info("No gpu available, using cpu for deterministic execution")
        return torch.device("cpu")
    
    # Use the gpu that's already been set via CUDA_VISIBLE_DEVICES
    device = torch.device("cuda:0")
    actual_gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "not_set")
    logger.info(f"Using gpu from CUDA_VISIBLE_DEVICES: {actual_gpu} (mapped to cuda:0)")
    
    try:
        # Set the device (gpu optimizations already applied by _apply_deterministic_performance_optimizations)
        torch.cuda.set_device(device)
        logger.info(f"gpu {device} configured for deterministic execution")
        return device
        
    except Exception as e:
        logger.warning(f"Could not configure gpu for determinism: {e}. Falling back to cpu.")
        return torch.device("cpu")


def _log_deterministic_configuration(seed_value: int, device: torch.device):
    """Log the final deterministic configuration for verification."""
    env_vars = [ f"GLOBAL_SEED={os.environ.get('GLOBAL_SEED')}",
        f"PYTHONHASHSEED={os.environ.get('PYTHONHASHSEED')}",
        f"CUBLAS_WORKSPACE_CONFIG={os.environ.get('CUBLAS_WORKSPACE_CONFIG')}",
        f"MKL_NUM_THREADS={os.environ.get('MKL_NUM_THREADS')}",
        f"OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS')}",
        f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', 'not_set')}",
        f"CUDA_LAUNCH_BLOCKING={os.environ.get('CUDA_LAUNCH_BLOCKING')}",
        f"DEVICE={device}",
        "MODE=DETERMINISTIC"
    ]
    logger.info("Deterministic environment configured: " + "; ".join(env_vars))
    
    # Log system info
    try:
        cpu_count = psutil.cpu_count()
        memory = psutil.virtual_memory()
        gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
        
        logger.info(f"System: cpu cores={cpu_count}, Memory={memory.available//1024**3}/{memory.total//1024**3} GB, gpus={gpu_count}")
        
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                logger.info(f"gpu {i}: {props.name} ({props.total_memory//1024**3} GB)")
                
    except Exception as e:
        logger.warning(f"Could not log system information: {e}")


def prepare_deterministic_run_directories(base_dir: Path, run_id: str, dataset_name: str, sensitive_attr: str) -> Path:
    """
        Prepare directories for a deterministic run.
    """
    timestamp = datetime.now().strftime("%Y_%m_%d_%H-%M-%S")
    run_dir = base_dir / f"fader_counterfactual_runs/{dataset_name}/{timestamp}_{sensitive_attr}" / run_id
    
    # Create all necessary subdirectories
    subdirs = [
        "logs",
        "fader_network_models", 
        "test_files",
        "counterfactuals",
        "constraints_passed_instances",
        "test_files_constraints_passed_test_instances",
        "results",
        "log_loss"
    ]
    
    for subdir in subdirs:
        (run_dir / subdir).mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Prepared deterministic run directories at: {shorten_path_for_logging(str(run_dir))}")
    return run_dir


def prepare_deterministic_run_directories_multi_attr(base_dir: Path, run_id: str, dataset_name: str, attrs: list) -> Path:
    """
    Prepare directories for a deterministic multi-attribute run.
    """
    timestamp = datetime.now().strftime("%Y_%m_%d_%H-%M-%S")
    attrs_str = "_".join(attrs)
    run_dir = base_dir / f"fader_counterfactual_runs/{dataset_name}/{timestamp}_{attrs_str}" / run_id
    
    # Create all necessary subdirectories
    subdirs = [
        "logs",
        "fader_network_models", 
        "test_files",
        "counterfactuals",
        "constraints_passed_instances",
        "test_files_constraints_passed_test_instances",
        "results",
        "log_loss",
        "test_files/predictions",
        "counterfactuals/predictions",
        "constraints_passed_instances/predictions",
        "test_files_constraints_passed_test_instances/predictions"
    ]
    
    for subdir in subdirs:
        (run_dir / subdir).mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Prepared deterministic multi-attribute run directories at: {shorten_path_for_logging(str(run_dir))}")
    return run_dir


def sanitize_age_in_csv_files(directory: Path, dataset_name: str):
    """
    Scans a directory for csv files and sanitizes the 'age' column in-place for 'credit' and 'bank_marketing' datasets. # Reference: AI360 

    If 'age' column has more than 2 unique values, it's binarized:
    - age > 25 and less than 65 is mapped to 1
    - age <= 25 or age >= 65 is mapped to 0
    """
    if dataset_name not in ['credit', 'bank_marketing']:
        logger.info(f"Dataset '{dataset_name}' does not require age sanitization. Skipping.")
        return
        
    logger.info(f"Sanitizing 'age' column for dataset '{dataset_name}' in directory: {shorten_path_for_logging(directory)}")
    
    if not directory.exists():
        logger.warning(f"Sanitization directory not found: {directory}. Skipping.")
        return

    for csv_file in directory.glob("*.csv"):
        try:
            df = pd.read_csv(csv_file)
            
            if 'age' in df.columns:
                if df['age'].nunique() > 3:
                    logger.info(f"Found raw 'age' values in '{csv_file.name}'. Binarizing in-place.")
                    original_ages = sorted(df['age'].unique())
                    df['age'] = df['age'].astype(int).apply(lambda x: 1 if x > 25 and x < 65 else 0)
                    df.to_csv(csv_file, index=False)
                    logger.info(f"  - Original unique 'age' values ({len(original_ages)}) [ min: {min(original_ages)}, max: {max(original_ages)} ] were mapped to {sorted(df['age'].unique())} with threshold 25.")
                else:
                    logger.debug(f"'age' column in '{csv_file.name}' is already binarized. Skipping.")
            else:
                logger.debug(f"No 'age' column in '{csv_file.name}'. Skipping.")

        except Exception as e:
            logger.error(f"Failed to process or save {csv_file.name}: {e}", exc_info=True) 


def copy_and_sanitize_test_files(config_dict: dict, test_files_run_dir: Path, dataset_name: str):
    """
    Copy only the original test file to the run directory and drop output column.
    """
    logger = get_logger(__name__)
    
    logger.info("phase: Copying original test file for experiment...")
    
    # Copy only the original test file from dataset directory
    dataset_dir = config_dict.get("dataset_dir")
    test_file_path = config_dict.get("test_file_path")
    output_column_name = config_dict.get("output_column_name")
    
    if dataset_dir and test_file_path:
        source_test_file = Path(dataset_dir) / test_file_path
        if source_test_file.exists():
            # Read the test file and drop output column if it exists
            test_df = pd.read_csv(source_test_file)
            if output_column_name and output_column_name in test_df.columns:
                test_df = test_df.drop(columns=[output_column_name])
                logger.info(f"Dropped output column '{output_column_name}' from test file")
            
            # Save the modified test file
            dest_file = test_files_run_dir / source_test_file.name
            test_df.to_csv(dest_file, index=False)
            logger.info(f"  - Copied original test file (output column dropped): '{source_test_file.name}'")
        else:
            logger.warning(f"  - Test file not found: {source_test_file}")
    else:
        logger.warning("  - Missing dataset_dir or test_file_path in config")

    # Sanitize the 'age' column in the copied file if needed
    logger.info("  - Sanitizing 'age' column in copied file...")
    sanitize_age_in_csv_files(test_files_run_dir, dataset_name)
    logger.info("phase: Test file preparation completed")


def create_shared_experiment_params(config_dict: dict, args, run_dir: Path, dataset_name: str):
    """
        Create shared parameters object that can be used for both ML training and experiment
    """
    
    # Create params object
    params = Params(config_dict)
    
    
    # Set command-line arguments
    params.sensitive_attr = args.sensitive_attr
    params.train_fader_network = "yes"
    
    # Set hyperparameters (now always single values)
    params.latent_dim = args.latent_dim  # Single value
    params.encoder_depth = args.encoder_depth
    params.k_ED = args.k_ED
    
    # Set run-specific paths
    params.fader_models_dir = str(run_dir / "fader_network_models")
    params.test_files_dir = str(run_dir / "test_files")
    params.counterfactual_dir = str(run_dir / "counterfactuals")
    params.results_dir = str(run_dir / "results")
    params.log_loss_dir = str(run_dir / "log_loss")
    
    # Set ml_models_dir to the shared location (not run-specific)
    params.ml_models_dir = str(Path("ml_models") / dataset_name)
    
    # Construct absolute paths for dataset files
    if hasattr(params, 'dataset_dir') and params.dataset_dir:
        if params.train_file_path and not os.path.isabs(params.train_file_path):
            params.train_file_path = os.path.join(params.dataset_dir, params.train_file_path)
        if params.test_file_path and not os.path.isabs(params.test_file_path):
            params.test_file_path = os.path.join(params.dataset_dir, params.test_file_path)
        if hasattr(params, 'val_file_path') and params.val_file_path and not os.path.isabs(params.val_file_path):
            params.val_file_path = os.path.join(params.dataset_dir, params.val_file_path)
    
    # Set additional deterministic parameters
    params.patience = 30
    params.warmup_epochs = 5
    params.ramp_epochs = 20
    params.eps = 1e-4
    params.weight_decay_D = 0.0001
    
    params.lambda_adv = args.lambda_adv

    params.lambda_now = 0.01
    params.scaling_factor = None
    
    # Neural network architecture parameters
    params.encoder_dropout = 0.05
    params.decoder_depth = params.encoder_depth
    params.decoder_dropout = 0.05
    params.encoder_decoder_learning_rate = 0.0005
    params.discriminator_learning_rate = 1e-3
    params.classifier_depth = max(1, (params.encoder_depth - 1)) if params.encoder_depth is not None else 2
    params.classifier_dropout = 0.2
    params.regressor_depth = max(1, (params.encoder_depth - 2)) if params.encoder_depth is not None else 1
    params.regressor_dropout = 0.2
    params.scheduler_step_size = 30
    params.scheduler_gamma = 0.1
    params.min_improvement = 0.0002    # Reduced from 0.0005 for longer training
    
    # Overwrite from CLI if provided
    if getattr(args, "batch_size", None) is not None:
        params.batch_size = args.batch_size
    
    logger.info(f"Created shared experiment params object for dataset: {dataset_name}")
    return params


def get_system_info():
    """
    Get system information for debugging and optimization.
    """
    info = {
        'cpu_count': psutil.cpu_count(),
        'cpu_count_logical': psutil.cpu_count(logical=True),
        'memory_total_gb': round(psutil.virtual_memory().total / (1024**3), 2),
        'memory_available_gb': round(psutil.virtual_memory().available / (1024**3), 2),
        'gpu_available': torch.cuda.is_available(),
        'gpu_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
    }
    
    if torch.cuda.is_available():
        gpu_info = []
        for i in range(torch.cuda.device_count()):
            gpu_props = torch.cuda.get_device_properties(i)
            gpu_info.append({
                'name': gpu_props.name,
                'memory_gb': round(gpu_props.total_memory / (1024**3), 2),
                'major': gpu_props.major,
                'minor': gpu_props.minor
            })
        info['gpu_details'] = gpu_info
    
    return info


def validate_experiment_arguments(args, config_dict=None):
    """
    Comprehensive validation for experiment arguments.
    """
    # Check config file exists
    if hasattr(args, 'config') and not Path(args.config).exists():
        raise FileNotFoundError(f"Config file not found: {args.config}")

    # Validate hyperparameters
    if hasattr(args, 'latent_dim') and args.latent_dim is not None and args.latent_dim <= 0:
        raise ValueError("latent_dim must be positive")
    if hasattr(args, 'encoder_depth') and args.encoder_depth is not None and args.encoder_depth <= 0:
        raise ValueError("encoder_depth must be positive")
    if hasattr(args, 'k_ED') and args.k_ED is not None and args.k_ED <= 0:
        raise ValueError("k_ED must be positive")
    
    # Validate gpu number if specified
    if hasattr(args, 'gpu_number') and args.gpu_number is not None and args.gpu_number < 0:
        raise ValueError("gpu_number must be non-negative")
    
    # Validate sensitive attribute
    if hasattr(args, 'sensitive_attr') and not args.sensitive_attr:
        raise ValueError("sensitive_attr cannot be empty")
    
    # Additional config-based validation
    if config_dict:
        required_config_keys = ['dataset_name', 'output_column_name', 'train_file_path', 'test_file_path']
        missing_keys = [key for key in required_config_keys if not config_dict.get(key)]
        if missing_keys:
            raise ValueError(f"Missing required config parameters: {missing_keys}")

    logger.info("All experiment arguments validated successfully")


def resolve_lambda_adv(args, params=None, default_value=2.0):
    """
    Centralized lambda_adv resolution with priority and logging.
    Priority: command line > config file > default
    """
    if hasattr(args, 'lambda_adv') and args.lambda_adv is not None:
        # Command line explicitly provided
        return args.lambda_adv, "command_line"
    elif params and hasattr(params, 'lambda_adv') and params.lambda_adv is not None:
        # Configuration file value
        return params.lambda_adv, "config_file"
    else:
        # Default fallback
        return default_value, "default"


def format_duration_seconds(total_seconds):
    """
    Convert total seconds to human-readable format.
    """
    if total_seconds < 60:
        return f"{total_seconds} seconds"
    elif total_seconds < 3600:
        minutes, seconds = divmod(total_seconds, 60)
        return f"{minutes} minutes {seconds} seconds"
    else:
        hours, remainder = divmod(total_seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        return f"{hours} hours {minutes} minutes {seconds} seconds"


def ensure_required_directories(base_path, subdirs):
    """
    Create required subdirectories under a base path.
    """
    paths = {}
    for subdir in subdirs:
        full_path = base_path / subdir
        full_path.mkdir(parents=True, exist_ok=True)
        paths[subdir] = full_path
    return paths