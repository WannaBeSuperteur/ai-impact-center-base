
import random

import numpy as np
import torch


SEED = 2026


def seed_everything(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)

    # PyTorch and cuDNN
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
