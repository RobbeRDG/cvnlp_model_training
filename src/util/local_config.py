from os.path import join
import pandas as pd
import numpy as np
import torch
import random

# Add the global modules to the python path
import sys
sys.path.append('/workspaces/cvnlp_model_training/code_mount/modules')

# Set the random seeds
seed = 0
np.random.seed(seed)
torch.manual_seed(seed)
random.seed(seed)

# Set the device
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Set the base path
BASE_PATH = '/workspaces/cvnlp_model_training'

# Set the checkpoint base paths
CHECKPOINT_BASE_PATH = join(BASE_PATH, 'model_checkpoints')

# Set the validation results base paths
VALIDATION_RESULTS_BASE_PATH = join(BASE_PATH, 'validation_results')
