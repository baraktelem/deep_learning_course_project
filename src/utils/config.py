"""
Kaggle-Invariant Configuration

Usage: Copy this setup cell to the top of every notebook:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

import sys, os
from pathlib import Path

IS_KAGGLE = os.environ.get('KAGGLE_KERNEL_RUN_TYPE', '') != ''

if IS_KAGGLE:
    # Install packages not available on Kaggle
    !pip install -q kymatio kornia
    
    # Add repo to path (UPDATE 'deep-learning-course-project' to your dataset slug)
    repo_path = Path('/kaggle/input/deep-learning-course-project')
    if repo_path.exists():
        sys.path.insert(0, str(repo_path))

from src.utils.config import *
set_seed(42)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

After this cell, you have access to:
- All paths: MODELS_PATH, CHECKPOINTS_PATH, FIGURES_PATH, etc.
- All libraries: torch, nn, np, plt, pd, tqdm, etc.
- Helper functions: set_seed(), get_checkpoint_path(), etc.
"""

import os
from pathlib import Path

# ============================================================
# Common Imports (available after `from config import *`)
# ============================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

# PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# torchvision
import torchvision
import torchvision.transforms as transforms
from torchvision import models

# PIL
from PIL import Image

# ============================================================
# Environment Detection
# ============================================================
# Kaggle sets specific environment variables we can check
IS_KAGGLE = os.environ.get('KAGGLE_KERNEL_RUN_TYPE', '') != ''

# ============================================================
# Base Paths
# ============================================================
if IS_KAGGLE:
    # Kaggle Paths
    INPUT_PATH = Path('/kaggle/input')
    OUTPUT_PATH = Path('/kaggle/working')
    REPO_PATH = INPUT_PATH / 'deep-learning-course-project'  # UPDATE: your dataset slug
    
    # Dataset paths
    DATA_PATH = INPUT_PATH
    
    # Notebooks path (on Kaggle, notebooks run from working directory)
    NOTEBOOKS_PATH = OUTPUT_PATH
    
    # Environment (reference only - not used on Kaggle)
    ENVIRONMENT_PATH = REPO_PATH / 'environment'
    ENVIRONMENT_FILE = ENVIRONMENT_PATH / 'environment.yaml'
    
    # Artifacts - all outputs go to working directory on Kaggle
    ARTIFACTS_PATH = OUTPUT_PATH / 'artifacts'
    CHECKPOINTS_PATH = ARTIFACTS_PATH / 'checkpoints'  # Model checkpoints
    FIGURES_PATH = ARTIFACTS_PATH / 'figures'          # Saved figures/plots
    OUTPUTS_PATH = ARTIFACTS_PATH / 'outputs'          # Outputs for acceleration (cached tensors, etc.)
    STATS_PATH = ARTIFACTS_PATH / 'stats'              # Training statistics
    
    # SRC structure (from uploaded repo dataset)
    SRC_PATH = REPO_PATH / 'src'
    
    # src/models/ - Model definitions
    MODELS_PATH = SRC_PATH / 'models'
    ARCHITECTURES_PATH = MODELS_PATH / 'architectures'  # Full model architectures
    BACKBONES_PATH = MODELS_PATH / 'backbones'          # Backbone networks (ResNet, etc.)
    LAYERS_PATH = MODELS_PATH / 'layers'                # Custom layers & blocks
    
    # src/losses/ - Custom loss functions
    LOSSES_PATH = SRC_PATH / 'losses'
    
    # src/utils/ - Utilities
    UTILS_PATH = SRC_PATH / 'utils'
    
else:
    # Local Paths (relative to project root)
    PROJECT_ROOT = Path(__file__).parent.parent.parent  # Goes up from src/utils/ to project root
    
    INPUT_PATH = PROJECT_ROOT / 'data'
    OUTPUT_PATH = PROJECT_ROOT / 'output'
    DATA_PATH = INPUT_PATH
    
    # Notebooks path
    NOTEBOOKS_PATH = PROJECT_ROOT / 'notebooks'    # Experiment notebooks
    
    # Environment (for reproducibility)
    ENVIRONMENT_PATH = PROJECT_ROOT / 'environment'
    ENVIRONMENT_FILE = ENVIRONMENT_PATH / 'environment.yaml'
    
    # Artifacts folder structure (following guidelines)
    ARTIFACTS_PATH = PROJECT_ROOT / 'artifacts'
    CHECKPOINTS_PATH = ARTIFACTS_PATH / 'checkpoints'  # Model checkpoints
    FIGURES_PATH = ARTIFACTS_PATH / 'figures'          # Saved figures/plots
    OUTPUTS_PATH = ARTIFACTS_PATH / 'outputs'          # Outputs for acceleration (cached tensors, etc.)
    STATS_PATH = ARTIFACTS_PATH / 'stats'              # Training statistics
    
    # SRC structure (following guidelines)
    SRC_PATH = PROJECT_ROOT / 'src'
    
    # src/models/ - Model definitions
    MODELS_PATH = SRC_PATH / 'models'
    ARCHITECTURES_PATH = MODELS_PATH / 'architectures'  # Full model architectures
    BACKBONES_PATH = MODELS_PATH / 'backbones'          # Backbone networks (ResNet, etc.)
    LAYERS_PATH = MODELS_PATH / 'layers'                # Custom layers & blocks
    
    # src/losses/ - Custom loss functions
    LOSSES_PATH = SRC_PATH / 'losses'
    
    # src/utils/ - Utilities
    UTILS_PATH = SRC_PATH / 'utils'

# ============================================================
# Helper Functions
# ============================================================
def set_seed(seed=42):
    """Set seed for reproducibility across all libraries."""
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_data_path(filename):
    """Get path to a data file."""
    return DATA_PATH / filename

def get_checkpoint_path(filename):
    """Get path to a model checkpoint file (in artifacts)."""
    return CHECKPOINTS_PATH / filename

def get_figure_path(filename):
    """Get path for saving a figure."""
    return FIGURES_PATH / filename

def get_output_path(filename):
    """Get path for saving outputs (cached computations, etc.)."""
    return OUTPUTS_PATH / filename

def get_stats_path(filename):
    """Get path for saving training statistics."""
    return STATS_PATH / filename

def ensure_dirs():
    """Create all necessary directories if they don't exist."""
    # Artifact directories (outputs)
    artifact_dirs = [CHECKPOINTS_PATH, FIGURES_PATH, OUTPUTS_PATH, STATS_PATH]
    
    # SRC directories (code structure) - only create locally, not on Kaggle
    if not IS_KAGGLE:
        src_dirs = [ARCHITECTURES_PATH, BACKBONES_PATH, LAYERS_PATH, LOSSES_PATH, NOTEBOOKS_PATH]
        artifact_dirs.extend(src_dirs)
    
    for path in artifact_dirs:
        path.mkdir(parents=True, exist_ok=True)

# ============================================================
# Auto-create directories on import (optional, comment out if not desired)
# ============================================================
# ensure_dirs()


