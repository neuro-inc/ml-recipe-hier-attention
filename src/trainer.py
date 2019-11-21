from enum import Enum
from pathlib import Path

import torch
from torch import autograd, nn
from torch.optim import SGD, Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataset import \
    (ImdbReviewsDataset,
     collate_docs,
     SimilarRandSampler
     )
from src.utils import OnlineAvg


def rround(x: float) -> float:
    return round(x, 3)


class Mode(Enum):
    TRAIN = 'TRAIN'
    TEST = 'TEST'


class ImdbTrainer:
    _model: nn.Module
    _train_set: ImdbReviewsDataset
    _test_set: ImdbReviewsDataset
    _batch_size: int
    _device: torch.device
    _ckpt_dir: Path

    _optim: Optimizer
    _criterion: nn.Module

    def __init__(self,
                 model: nn.Module,
                 train_set: ImdbReviewsDataset,
                 test_set: ImdbReviewsDataset,
                 batch_size: int,
                 device: torch.device,
                 ckpt_dir: Path,
                 ):
        self._model = model
        self._train_set = train_set
        self._test_set = test_set
        self._batch_size = batch_size
        self._device = device
        self._ckpt_dir = ckpt_dir

        self._optim = SGD(self._model.parameters(), lr=1e-3, momentum=.9)
        self._criterion = nn.BCELoss()

        self._model.to(self._device)

    def _loop(self, mode: Mode) -> float:
        if mode == mode.TRAIN:
            grad_context = autograd.enable_grad
            dataset = self._train_set
            self._model.train()

        elif mode == mode.TEST:
            grad_context = autograd.no_grad
            dataset = self._test_set
            self._model.eval()

        else:
            raise ValueError(f'Unexpected mode: {mode}.')

        sampler = SimilarRandSampler(keys=dataset.txt_lens,
                                     bs=self._batch_size)
        loader_tqdm = tqdm(DataLoader(dataset=dataset, num_workers=4,
                                      batch_size=self._batch_size,
                                      collate_fn=collate_docs,
                                      sampler=sampler),
                           total=len(sampler))

        avg_accuracy = OnlineAvg()
        avg_loss = OnlineAvg()

        with grad_context():
            for docs, labels in loader_tqdm:
                pred, _, _ = self._model(x=docs.to(self._device))
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
