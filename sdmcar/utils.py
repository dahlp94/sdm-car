# sdmcar/utils.py

import torch
import torch.nn.functional as F

def set_default_dtype(dtype=torch.double):
    """
    Optionally call this once at the beginning of your script
    if you want a global default dtype.
    """
    torch.set_default_dtype(dtype)

def set_seed(seed: int = 42):
    """Set the PyTorch random seed."""
    torch.manual_seed(seed)

def softplus(x):
    """Alias to keep things explicit."""
    return F.softplus(x)

def kl_normal_std(mu, log_std):
    """
    KL( N(mu, std^2) || N(0,1) ) = 0.5 * (mu^2 + std^2 - 1 - log std^2),
    where std = exp(log_std).
    """
    std = torch.exp(log_std)
    return 0.5 * (mu**2 + std**2 - 1.0 - 2.0 * log_std)
