import torch
import numpy as np

def set_seeds():
    """Set seeds to random generators
    """
    torch.manual_seed(42)
    np.random.seed(42)
    torch.cuda.manual_seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True