
import os
import random

import numpy as np
import torch
import torch.backends.cudnn as cudnn


SEED = 2026


def seed_everything(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)

    # PyTorch and cuDNN
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    cudnn.deterministic = True
    cudnn.benchmark = False

    print(f'seed set as {seed}')
