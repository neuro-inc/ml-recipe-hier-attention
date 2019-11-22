from argparse import ArgumentParser, Namespace

from torch import device

from src.const import LOG_DIR
from src.dataset import get_loaders
from src.model import HAN
from src.trainer import ImdbTrainer
from src.utils import fix_seed


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
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=device, default='cuda:0')
    return parser


if __name__ == '__main__':
    main(args=get_parser().parse_args())
