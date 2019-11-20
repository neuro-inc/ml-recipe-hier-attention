import re
from functools import lru_cache
from pathlib import Path
from typing import Tuple, List, Dict, Union

import torch
from nltk.tokenize import PunktSentenceTokenizer, WordPunctTokenizer
from torch import LongTensor, FloatTensor
from tqdm import tqdm

from src.const import IMBD_ROOT

TText = List[List[int]]  # text[i_sentece][j_word]
TItem = Dict[str, Union[TText, int]]


class ImdbReviewsDataset:
    _path_to_data: Path
    _s_tokenizer: PunktSentenceTokenizer
    _w_tokenizer: WordPunctTokenizer
    _html_re: re.Pattern  # type: ignore

    # data fields
    _paths: List[Path]
    _texts: List[TText]
    _labels: List[int]
    _txt_lens: List[int]
    _snt_lens: List[int]
    _vocab: Dict[str, int]

    def __init__(self, path_to_data: Path, vocab: Dict[str, int]):
        self._path_to_data = path_to_data
        self._vocab = vocab

        self._s_tokenizer = PunktSentenceTokenizer()
        self._w_tokenizer = WordPunctTokenizer()
        self._html_re = re.compile('<.*?>')

        self._paths = []
        self._texts = []
        self._labels = []
        self._txt_lens = []
        self._snt_lens = []

        self._load_data()

    def __len__(self) -> int:
        return len(self._texts)

    @lru_cache(maxsize=50_000)  # equal to number of reviews in imdb
    def __getitem__(self, i: int) -> TItem:
        return {
            'txt': self._texts[i],
            'label': self._labels[i],
            'txt_len': self._txt_lens[i],
            'snt_len': self._snt_lens[i]
        }

    def _load_data(self) -> None:
        files = list((self._path_to_data / 'neg').glob('*_*.txt')) + \
                list((self._path_to_data / 'pos').glob('*_*.txt'))

        print(f'Dataset loading from {self._path_to_data}.')
        for file_path in tqdm(files):
            with open(file_path, 'r') as f:
                text, snt_len_max, txt_len = self.tokenize_plane_text(f.read())
                label = 1 if file_path.parent.name == 'pos' else 0

                self._paths.append(file_path)
                self._texts.append(text)
                self._labels.append(label)
                self._snt_lens.append(snt_len_max)
                self._txt_lens.append(txt_len)

    def tokenize_plane_text(self, text_plane: str
                            ) -> Tuple[TText, int, int]:
        tokenize_w = self._w_tokenizer.tokenize
        tokenize_s = self._s_tokenizer.tokenize

        text_plane = text_plane.lower()
        text_plane = re.sub(self._html_re, ' ', text_plane)
        text = [
            [self.vocab[w] for w in tokenize_w(s) if w in self._vocab.keys()]
            for s in tokenize_s(text_plane)
        ]

        snt_len_max = max([len(snt) for snt in text])
        txt_len = len(text)

        return text, snt_len_max, txt_len

    @staticmethod
    def get_imdb_vocab(imdb_root: Path) -> Dict[str, int]:
        with open(imdb_root / 'imdb.vocab') as f:
            words = f.read().splitlines()

        # note, that we keep 0 for padding token
        ids = list(range(1, len(words) + 1))
        vocab = dict(zip(words, ids))

        return vocab

    @property
    def vocab(self) -> Dict[str, int]:
        return self._vocab


def collate_docs(batch: List[TItem]) -> Tuple[LongTensor, FloatTensor]:
    n_docs = len(batch)  # number of documents in batch
    max_snt = max([item['snt_len'] for item in batch])
    max_txt = max([item['txt_len'] for item in batch])

    labels_tensor = torch.zeros((n_docs, 1), dtype=torch.float32)
    docs_tensor = torch.zeros((n_docs, max_txt, max_snt),
                              dtype=torch.int64)

    for i_doc, item in enumerate(batch):
        labels_tensor[i_doc] = item['label']

        for i_snt, snt in enumerate(item['txt']):  # type: ignore
            snt_len = len(snt)
            docs_tensor[i_doc, i_snt, 0:snt_len] = torch.tensor(snt)

    return docs_tensor, labels_tensor


def get_datasets(imbd_root: Path = IMBD_ROOT
                 ) -> Tuple[ImdbReviewsDataset, ImdbReviewsDataset]:
    vocab = ImdbReviewsDataset.get_imdb_vocab(imbd_root)
    train_set = ImdbReviewsDataset(imbd_root / 'train', vocab)
    test_set = ImdbReviewsDataset(imbd_root / 'test', vocab)

    return train_set, test_set


def get_test_dataset(imbd_root: Path = IMBD_ROOT) -> ImdbReviewsDataset:
    vocab = ImdbReviewsDataset.get_imdb_vocab(imbd_root)
    return ImdbReviewsDataset(imbd_root / 'test', vocab)
