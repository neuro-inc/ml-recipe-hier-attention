import re
from pathlib import Path
from typing import Tuple, List

from tqdm import tqdm

__all__ = ['ImdbReviewsDataset', 'get_datasets']

TItem = Tuple[List[str], float]


class ImdbReviewsDataset:
    _path_to_data: Path
    _n_files: int
    _data: Tuple[TItem]
    _token_regex: re.Pattern

    def __init__(self, path_to_data: Path):
        self._path_to_data = path_to_data
        self._token_regex = re.compile(r'[A-Za-z]+|[!?.:,()]')

        files_neg = list((self._path_to_data / 'neg').glob('*_*.txt'))
        files_pos = list((self._path_to_data / 'pos').glob('*_*.txt'))
        self._n_files = len(files_neg) + len(files_pos)

        texts, labels = [], []
        for file_path in tqdm(files_pos + files_neg):
            label = 1 if 'pos' in str(file_path) else 0
            text = self.read_review(file_path)

            texts.append(text)
            labels.append(label)

        self._data = tuple(zip(texts, labels))

    def __len__(self) -> int:
        return self._n_files

    def __getitem__(self, i: int):
        pass

    def get_i_review(self, i: int) -> TItem:
        return self._data[i]

    def read_review(self, path_to_review: Path) -> List[str]:
        with open(path_to_review, 'r') as f:
            data = self._token_regex.findall(f.read())
            data = ' '.join([x.lower() for x in data])
        return data


def get_datasets() -> Tuple[ImdbReviewsDataset, ImdbReviewsDataset]:
    imdb_root = Path(__file__).parent.parent.parent / 'data' / 'aclImdb'
    train_set = ImdbReviewsDataset(imdb_root / 'train')
    test_set = ImdbReviewsDataset(imdb_root / 'test')
    return train_set, test_set


def fast_check() -> None:
    train_set, test_set = get_datasets()
    print(train_set.get_i_review(0))
    print(test_set.get_i_review(0))


if __name__ == '__main__':
    fast_check()
