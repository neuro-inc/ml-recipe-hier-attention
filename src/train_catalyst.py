import warnings

import torch
from catalyst.dl import SupervisedRunner
from catalyst.dl.callbacks import AccuracyCallback

from src.const import LOG_DIR
from src.dataset import get_loaders
from src.model import HAN

warnings.simplefilter('ignore')


def main() -> None:
    # experiment setup
    logdir = LOG_DIR.parent / 'catalyst'
    num_epochs = 10

    # data
    train_loader, test_loader, vocab = get_loaders(batch_size=16)

    loaders = {"train": train_loader, "valid": test_loader}

    # model, criterion, optimizer
    model = HAN(vocab=vocab, freeze_emb=False)

    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.SGD(lr=1e-3, momentum=.9,
                                params=model.parameters())
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)

    # model runner
    runner = SupervisedRunner(device=torch.device('cuda:0'),
                              input_key='features',
                              output_key=None,
                              input_target_key='targets'
                              )

    # model training
    runner.train(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        loaders=loaders,
        logdir=logdir,
        num_epochs=num_epochs,
        verbose=True,
        minimize_metric=False,
        main_metric='accuracy01',
        callbacks=[
            AccuracyCallback(prefix='accuracy',
                             accuracy_args=[1],
                             input_key='targets',
                             output_key='logits',
                             threshold=.5,
                             num_classes=1
                             )
        ]
    )


main()
