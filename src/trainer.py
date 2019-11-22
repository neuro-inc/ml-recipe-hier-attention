from enum import Enum
from pathlib import Path

import torch
from torch import autograd, nn
from torch.optim import SGD, Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.utils import OnlineAvg


def rround(x: float) -> float:
    return round(x, 3)


class Mode(Enum):
    TRAIN = 'TRAIN'
    TEST = 'TEST'


class ImdbTrainer:
    _model: nn.Module
    _train_loader: DataLoader
    _test_loader: DataLoader
    _device: torch.device
    _ckpt_dir: Path

    _optim: Optimizer
    _criterion: nn.Module

    def __init__(self,
                 model: nn.Module,
                 train_loader: DataLoader,
                 test_loader: DataLoader,
                 device: torch.device,
                 ckpt_dir: Path,
                 ):
        self._model = model
        self._train_loader = train_loader
        self._test_loader = test_loader
        self._device = device
        self._ckpt_dir = ckpt_dir

        self._optim = SGD(self._model.parameters(), lr=.5 * 1e-2, momentum=.9)
        self._criterion = nn.BCELoss()

        self._model.to(self._device)

    def _loop(self, mode: Mode) -> float:
        if mode == mode.TRAIN:
            grad_context = autograd.enable_grad
            loader = self._train_loader
            self._model.train()

        elif mode == mode.TEST:
            grad_context = autograd.no_grad
            loader = self._test_loader
            self._model.eval()

        else:
            raise ValueError(f'Unexpected mode: {mode}.')

        loader_tqdm = tqdm(loader, total=len(loader.sampler))

        avg_accuracy = OnlineAvg()
        avg_loss = OnlineAvg()

        with grad_context():
            for batch in loader_tqdm:
                docs, labels = batch['features'], batch['targets']
                pred = self._model(x=docs.to(self._device))['logits']
                loss = self._criterion(target=labels.to(self._device),
                                       input=pred)

                if mode == Mode.TRAIN:
                    loss.backward()
                    self._optim.step()
                    self._optim.zero_grad()

                pred_th = (pred.detach().cpu() > .5).long()
                batch_acc = float((labels.long() == pred_th).float().mean())
                batch_loss = float(loss.detach().cpu())
                avg_accuracy.update(batch_acc)
                avg_loss.update(batch_loss)

                loader_tqdm.set_postfix([
                    (f'{mode} Accuracy', rround(avg_accuracy.avg)),
                    ('Loss', rround(avg_loss.avg))
                ])

        return avg_accuracy.avg

    def train(self, n_epoch: int) -> None:
        best_metric = - 1.0

        for i_epoch in range(n_epoch):

            print(f'Epoch: {i_epoch} / {n_epoch - 1}.')
            self._loop(mode=Mode.TRAIN)

            metric = self._loop(mode=Mode.TEST)

            if metric > best_metric:
                best_metric = metric
                self._model.save(self._ckpt_dir / 'best.ckpt')

        print(f'Train finished, best metric is {rround(best_metric)}.')
