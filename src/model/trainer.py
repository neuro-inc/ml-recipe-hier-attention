from enum import Enum

import torch
from torch import nn
from torch.autograd import enable_grad, no_grad
from torch.optim import SGD, Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.model.dataset import ImdbReviewsDataset, collate_docs
from src.model.model import HAN
from src.utils import OnlineAvg


class Mode(Enum):
    TRAIN = 'TRAIN'
    TEST = 'TEST'


class ImdbTrainer:
    _train_set: ImdbReviewsDataset
    _test_set: ImdbReviewsDataset
    _batch_size: int
    _device: torch.device

    _model: nn.Module
    _optim: Optimizer
    _criterion: nn.Module

    def __init__(self,
                 train_set: ImdbReviewsDataset,
                 test_set: ImdbReviewsDataset,
                 batch_size: int,
                 device: torch.device
                 ):
        self._train_set = train_set
        self._test_set = test_set
        self._batch_size = batch_size
        self._device = device

        self._model = HAN(vocab=self._train_set.vocab)
        self._optim = SGD(params=self._model.parameters(), lr=1e-3)
        self._criterion = nn.BCELoss()

        self._model.to(self._device)

    def _loop(self, mode: Mode) -> None:
        if mode == mode.TRAIN:
            grad_context = enable_grad
            dataset = self._train_set
            self._model.train()

        elif mode == mode.TEST:
            grad_context = no_grad
            dataset = self._test_set
            self._model.eval()

        else:
            raise ValueError(f'Unexpected mode: {mode}.')

        loader_tqdm = tqdm(DataLoader(
            dataset=dataset, batch_size=self._batch_size,
            collate_fn=collate_docs, num_workers=4,
            shuffle=True)
        )
        avg_accuracy = OnlineAvg()
        avg_loss = OnlineAvg()

        with grad_context():
            for docs, labels in loader_tqdm:
                pred, _, _ = self._model(x=docs.to(self._device))
                loss = self._criterion(input=pred, target=labels)

                if mode == Mode.TRAIN:
                    loss.backward()
                    self._optim.step()
                    self._optim.zero_grad()

                batch_acc = float((labels == (pred.detach().cpu() > .5)).float().mean())
                batch_loss = float(loss.detach().cpu())
                avg_accuracy.update(batch_acc)
                avg_loss.update(batch_loss)

                loader_tqdm.set_postfix([
                    ('Accuracy', round(avg_accuracy.avg, 3)),
                    ('Loss', round(avg_loss.avg, 3))
                ])

        print(f'{mode}: Accuracy: {avg_accuracy}, Loss: {avg_loss} \n')

    def train(self, n_epoch: int) -> None:
        for i in range(n_epoch):
            self._loop(mode=Mode.TRAIN)
            self._loop(mode=Mode.TEST)
