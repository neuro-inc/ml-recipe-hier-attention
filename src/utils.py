import random

import numpy as np
import torch


class OnlineAvg:
    _avg: float
    _n: int

    def __init__(self):
        self._avg = 0
        self._n = 0

    def update(self, new_x: float) -> None:
        self._n += 1
        self._avg = (self._avg * (self._n - 1) + new_x) / self._n

    @property
    def avg(self) -> float:
        return self._avg

    @property
    def n(self) -> int:
        return n

    def __str__(self) -> str:
        return self._avg.__str__()


def fix_seed(seed: int = 0) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
