from os.path import join
import pandas as pd
import numpy as np
import torch
import random

# Add the global modules to the python path
import sys
sys.path.append('/workspaces/cvnlp_speech_sentiment_recognition/code_mount/modules')

# Set the random seeds
seed = 0
np.random.seed(seed)
torch.manual_seed(seed)
random.seed(seed)

# Set the device
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Set the base path
BASE_PATH = '/workspaces/cvnlp_speech_sentiment_recognition'
