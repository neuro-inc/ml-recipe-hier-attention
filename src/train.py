from argparse import ArgumentParser, Namespace
from enum import Enum
from pathlib import Path

import torch
from torch import nn, autograd
from torch.optim import Optimizer, SGD
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.const import LOG_DIR
from src.dataset import get_loaders
from src.model import HAN
from src.utils import fix_seed, OnlineAvg, rround


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

        self._optim = SGD(self._model.parameters(), lr=1e-3, momentum=.9)
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


def main(args: Namespace) -> None:
    fix_seed(args.seed)

    train_loader, test_loader, vocab = get_loaders(batch_size=args.batch_size)

    model = HAN(vocab=vocab, freeze_emb=args.freeze_emb)

    trainer = ImdbTrainer(train_loader=train_loader, test_loader=test_loader,
                          device=args.device, ckpt_dir=LOG_DIR, model=model)

    trainer.train(n_epoch=args.n_epoch)


def get_parser() -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument('--n_epoch', type=int, default=500)
    parser.add_argument('--freeze_emb', type=bool, default=False)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=torch.device, default='cuda:0')
    return parser


if __name__ == '__main__':
    main(args=get_parser().parse_args())
