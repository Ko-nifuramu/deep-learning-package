from typing import List, Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor


class EarlyStopping:
    def __init__(self, patience, verbose=True):
        self.patience = patience
        self.counter = 0
        self.early_stop = False
        self.loss_min = np.Inf
        self.verbose = verbose
        print("early stopping: patience 500")

    def __call__(self, loss, model):
        if loss > self.loss_min:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
            if self.verbose:
                print(f"(Counter: {self.counter} / {self.patience})", end="")
        else:
            self._save_model(loss, model)
            self.counter = 0

    def _save_model(self, loss, model):
        if self.verbose:
            print(f"(Decreased {self.loss_min-loss})")

        torch.save(model.state_dict(), f"models/{model.name}/best-model.pth")
        self.loss_min = loss


class LossHolder:
    def __init__(self, attrs: List[str]):
        self._attrs = attrs
        for attr in attrs:
            setattr(self, attr, [])
            setattr(self, "_iter" + attr, [])

    def __iadd__(self, packed_loss: Sequence[Tensor]):
        assert len(packed_loss) == len(self._attrs)
        for attr, unpacked_loss in zip(self._attrs, packed_loss):
            getattr(self, "_iter" + attr).append(unpacked_loss.item())
        return self

    def _calc(self) -> None:
        for attr in self._attrs:
            getattr(self, attr).append(np.mean(self.__dict__["_iter" + attr]))
            getattr(self, "_iter" + attr).clear()


class LossLogger:
    def __init__(self, attrs: List[str]):
        self.attrs = attrs
        self.train = LossHolder(attrs=attrs)
        self.val = LossHolder(attrs=attrs)
