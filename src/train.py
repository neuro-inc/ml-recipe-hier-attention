from argparse import ArgumentParser, Namespace

from torch import device

from src.model.dataset import get_datasets
from src.model.trainer import ImdbTrainer
from src.utils import fix_seed


def main(args: Namespace) -> None:
    fix_seed(args.seed)

    train_set, test_set = get_datasets()

    trainer = ImdbTrainer(train_set=train_set, test_set=test_set,
                          batch_size=args.batch_size, device=args.device)

    trainer.train(n_epoch=args.n_epoch)


def get_parser() -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument('--n_epoch', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=device, default='cpu')
    return parser


if __name__ == '__main__':
    main(args=get_parser().parse_args())
