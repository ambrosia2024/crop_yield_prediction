import os
import random

import numpy as np

import torch
from lightning.pytorch import seed_everything

def verify_parameters(crop, model, country):
    """
    Verifies if the pipeline for selected crop, model, and country is implemented.
    """
    assert crop in ["maize", "wheat"]
    assert model in ["ridge", "svr", "rf", "gb", "mlp"]

    assert crop in ["maize", "wheat"]
    assert model in ["ridge", "svr", "rf", "gb", "mlp"]

    if crop == "maize":
        assert country in ['AT', 'BE', 'BG', 'CZ', 'DE', 'DK', 'EL', 'ES', 'FR', 'HR', 'HU', 'IT', 'LT', 'NL', 'PL', 'PT', 'RO', 'SE', 'all']
    else:
        assert country in ['AT', 'BE', 'BG', 'CZ', 'DE', 'DK', 'EE', 'EL', 'ES', 'FI', 'FR', 'HR', 'HU', 'IE', 'IT', 'LT', 'LV', 'NL', 'PL', 'PT', 'RO', 'SE', 'all']

def select_country(crop, country):
    """
    A function to collect all the countries if country is set to "all". Otherwise,
    it just returns [country]
    """
    if country == "all":
        if crop == "maize":
            country = ['AT', 'BE', 'BG', 'CZ', 'DE', 'DK', 'EL', 'ES', 'FR', 'HR', 
                                'HU', 'IT', 'LT', 'NL', 'PL', 'PT', 'RO', 'SE']
        else:
            country = ['AT', 'BE', 'BG', 'CZ', 'DE', 'DK', 'EE', 'EL', 'ES', 'FI', 
                                'FR', 'HR', 'HU', 'IE', 'IT', 'LT', 'LV', 'NL', 'PL', 'PT', 
                                'RO', 'SE']
    else:
        country = [country]
    return country

def seed_uniformly(seed):
    # Setting seed value for reproducibility    
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ['CUDA_VISIBLE_DEVICES'] = "0"
    seed_everything(seed)

