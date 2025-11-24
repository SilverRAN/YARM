from .fixed_config import CORRUPTIONS as cfg
from random import random
from .blur import *
from .noise import *
from .weather import *
from .image import *

def random_corruption(inp, severity='random', selected_corruption=None):
    """
    Args:
        inp: torch.Tensor in shape (B, C, H, W);
        severity: 'random' or int [0, 4], severity level of corruption;
        selected_corruption: str or None, name of corruption to apply;
    Returns:
        corrupted image tensor in shape (B, C, H, W).
    """
    corruptions = list(cfg.keys())
    if selected_corruption is None:
        selected_corruption = corruptions[int(random() * len(corruptions))]

    if severity == 'random':
        severity = int(random() * 5)
    elif isinstance(severity, int):
        pass
    return eval(selected_corruption)(inp, *cfg[selected_corruption][severity])