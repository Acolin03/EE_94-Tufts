# src/__init__.py

# Import models
from .models import RRAM_PINN, MLP_Current

# Import data handling
from .data import (
    Constants, 
    RRAMDataset, 
    collate_sequences
)

from .loss import (
    simulate_rram_wrapper,
    simulate_rram
)

# Import training functions
from .training import (
    setup_logger,
    train_sequence,
    validate_sequence,  
    AdaptivePDEScheduler
)

# Import utilities
from .utils import (
    configure_optimizers,
    calculate_accuracy,
    calculate_epoch_metrics,
    load_checkpoint,
    save_checkpoint,
    calculate_metric_range,
    print_training_progress
)

# Define what should be available when using "from src import *"
__all__ = [
    # Models
    'RRAM_PINN',
    'MLP_Current',
    
    # Data handling
    'Constants',
    'RRAMDataset',
    'load_data',
    'collate_sequences',
    
    # Loss
    'simulate_rram_wrapper',
    'simulate_rram',
    
    # Training
    'setup_logger',
    'train_step',
    'train_sequence',
    'validate_step',
    'validate_sequence',
    'AdaptivePDEScheduler',
    
    # Utils
    'configure_optimizers',
    'calculate_accuracy',
    'load_best_model',
    'save_checkpoint',
    'calculate_metric_range',
    'print_training_progress'
]