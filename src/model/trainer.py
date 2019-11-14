from enum import Enum

from torch.utils.data import DataLoader
from tqdm import tqdm

from src.model.dataset import ImdbReviewsDataset, collate_docs


class Mode(Enum):
    TRAIN = 'TRAIN'
    TEST = 'TEST'


class ImdbTrainer:
    _train_set: ImdbReviewsDataset
    _test_set: ImdbReviewsDataset
    _batch_size: int

    def __init__(self,
                 train_set: ImdbReviewsDataset,
                 test_set: ImdbReviewsDataset,
                 batch_size: int
                 ):
        self._train_set = train_set
        self._test_set = test_set
        self._batch_size = batch_size

    def loop(self, mode: Mode) -> None:
        if mode == mode.TRAIN:
            dataset = self._train_set
        else:
            raise ValueError(f'Unexpected mode: {mode}.')

        loader = DataLoader(dataset=dataset, batch_size=self._batch_size,
                            collate_fn=collate_docs, num_workers=1)

        for docs, labels in tqdm(loader):
            print(docs.shape, labels)
            break

    def train(self, n_epoch: int) -> None:
        for i in range(n_epoch):
            self.loop(mode=Mode.TRAIN)
