import re
from functools import lru_cache
from pathlib import Path
from typing import Dict
from typing import Tuple, List

from nltk.tokenize import PunktSentenceTokenizer, WordPunctTokenizer
from tqdm import tqdm

__all__ = ['ImdbReviewsDataset', 'get_datasets']

TText = List[List[int]]  # text[i_sentece][j_word]
TItem = Tuple[TText, int, int]  # text, label, max_sent

TTextStr = List[List[str]]
TItemStr = Tuple[TTextStr, str, int]


class ImdbReviewsDataset:
    _path_to_data: Path
    _s_tokenizer: PunktSentenceTokenizer
    _w_tokenizer: WordPunctTokenizer
    _data: Tuple[TItemStr, ...]
    _html_re: re.Pattern  # type: ignore
    _vocab: Dict[str, int]

    def __init__(self, path_to_data: Path, vocab: Dict[str, int]):
        self._path_to_data = path_to_data
        self._vocab = vocab

        self._s_tokenizer = PunktSentenceTokenizer()
        self._w_tokenizer = WordPunctTokenizer()
        self._html_re = re.compile('<.*?>')

        self._load_data()

    def __len__(self) -> int:
        return len(self._data)

    @lru_cache(maxsize=50_000)  # equal to amount if reviews
    def __getitem__(self, i: int) -> TItem:
        text_str, label, max_sent = self.get_i_review(i)
        text = [[self._vocab[w] for w in s] for s in text_str]

        return text, label, max_sent

    def get_i_review(self, i: int) -> TItemStr:
        return self._data[i]

    def read_review(self, path_to_review: Path) -> TTextStr:
        with open(path_to_review, 'r') as f:
            text = f.read().lower()
            text = re.sub(self._html_re, ' ', text)
            text = [
                [w for w in self._w_tokenizer.tokenize(s) if w in self._vocab.keys()]
                for s in self._s_tokenizer.tokenize(text)
            ]

        return text

    def _load_data(self):
        files_neg = list((self._path_to_data / 'neg').glob('*_*.txt'))
        files_pos = list((self._path_to_data / 'pos').glob('*_*.txt'))

        texts, labels, max_sents = [], [], []
        for file_path in tqdm(files_pos + files_neg):
            label = file_path.parent.name
            text = self.read_review(file_path)
            max_sent = max([len(sent) for sent in text])

            texts.append(text)
            labels.append(label)
            max_sents.append(max_sent)

            break

        self._data = tuple(zip(texts, labels, max_sents))


def get_imdb_vocab(imdb_root: Path) -> Dict[str, int]:
    with open(imdb_root / 'imdb.vocab') as f:
        words = f.read().splitlines()

    ids = list(range(1, len(words) + 1))
    vocab = dict(zip(words, ids))
    # note, that we keep 0 for padding token
    return vocab


def get_datasets() -> Tuple[ImdbReviewsDataset, ImdbReviewsDataset]:
    imdb_root = Path(__file__).parent.parent.parent / 'data' / 'aclImdb'

    vocab = get_imdb_vocab(imdb_root)
    train_set = ImdbReviewsDataset(imdb_root / 'train', vocab)
    test_set = ImdbReviewsDataset(imdb_root / 'test', vocab)
    return train_set, test_set


def fast_check() -> None:
    train_set, test_set = get_datasets()

    print(train_set.get_i_review(0))
    print(train_set[0])


if __name__ == '__main__':
    fast_check()
