import warnings
from argparse import ArgumentParser, Namespace
from collections import OrderedDict
from pathlib import Path

import catalyst.dl.callbacks as clb
import torch
from catalyst.dl.runner import SupervisedWandbRunner, SupervisedRunner
from catalyst.utils import set_global_seed
from torch import device as tdevice
from torch.cuda import is_available

from src.const import LOG_DIR
from src.dataset import get_loaders
from src.model import HAN
from src.utils import setup_wandb

warnings.simplefilter('ignore')


def main(args: Namespace) -> None:
    set_global_seed(args.seed)
    is_wandb = setup_wandb()

    train_loader, test_loader, vocab = get_loaders(batch_size=args.batch_size)
    loaders = OrderedDict([('train', train_loader), ('valid', test_loader)])

    model = HAN(vocab=vocab, freeze_emb=args.freeze_emb)

    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.SGD(lr=1e-2, momentum=.9,
                                params=model.parameters())
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)

    if is_wandb:
        Runner = SupervisedWandbRunner
        extra_args = {'monitoring_params': {'project': 'neuro_imdb'}}
    else:
        Runner = SupervisedRunner
        extra_args = {}

    runner = Runner(input_key='features', output_key=None,
                    input_target_key='targets',
                    device=args.device if is_available() else tdevice('cpu')
                    )

    callbacks = [
        clb.AccuracyCallback(prefix='accuracy', input_key='targets',
                             output_key='logits', accuracy_args=[1],
                             threshold=.5, num_classes=1, activation=None),
        clb.EarlyStoppingCallback(patience=5, minimize=False,
                                  min_delta=0.02, metric='accuracy01')
    ]
    runner.train(
        model=model, criterion=criterion, optimizer=optimizer,
        scheduler=scheduler, loaders=loaders, logdir=str(args.logdir),
        num_epochs=args.n_epoch, verbose=True, main_metric='accuracy01',
        valid_loader='valid', callbacks=callbacks, minimize_metric=False,
        checkpoint_data={'params': model.init_params},
        **extra_args
    )


def get_parser() -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument('--n_epoch', type=int, default=500)
    parser.add_argument('--freeze_emb', type=bool, default=False)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=tdevice, default='cuda:0')
    parser.add_argument('--logdir', type=Path, default=LOG_DIR)
    return parser


if __name__ == '__main__':
    main(args=get_parser().parse_args())
