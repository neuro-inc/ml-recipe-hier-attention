from typing import Dict

import torch
import torchtext
from torch import nn, Tensor, LongTensor
from torch.nn.functional import cosine_similarity

from src.model.dataset import IMBD_ROOT, ImdbReviewsDataset


class HAN(nn.Module):
    _embedding: nn.Embedding

    def __init__(self, vocab: Dict[str, int]):
        super().__init__()

        self._embedding = get_pretrained_embedding(vocab)

    def forward(self, x: LongTensor) -> Tensor:
        # x is a tensor with shape of [n_txt, n_snt, n_words]
        x = self._embedding(x)
        return x


def get_pretrained_embedding(vocab: Dict[str, int]) -> nn.Embedding:
    emb_size = 100
    glove = torchtext.vocab.GloVe(name='6B', dim=emb_size)
    glove.unk_init = lambda x: torch.ones(emb_size, dtype=torch.float32)

    vocab_size = len(vocab) + 1  # add 1 because of padding token

    weights = torch.zeros((vocab_size, emb_size), dtype=torch.float32)
    for word, idx in vocab.items():
        emb = glove.get_vecs_by_tokens([word])
        weights[idx, :] = emb

    embedding = nn.Embedding.from_pretrained(embeddings=weights, freeze=True)

    # thus, we reserved 0-vector for padding, 1-vector for unknown tokens
    return embedding


def check_model() -> None:
    vocab = {'cat': 1, 'dog': 2, 'bird': 3}  # 0 reserved for padding
    model = HAN(vocab)
    batch = torch.randint(low=1, high=len(vocab),
                          size=(16, 12, 10), dtype=torch.int64
                          )
    output = model(batch)
    print(output.shape)


def check_ptratrained_embedding() -> None:
    vocab = ImdbReviewsDataset.get_imdb_vocab(IMBD_ROOT)
    embedding = get_pretrained_embedding(vocab=vocab)

    def word2emb(w: str) -> torch.LongTensor:
        ind_in_vocab = torch.tensor(vocab[w], dtype=torch.int64)
        emb = embedding(ind_in_vocab)
        return emb

    def similarity(w1: str, w2: str) -> torch.FloatTensor:
        emb1, emb2 = word2emb(w1), word2emb(w2)
        d = cosine_similarity(emb1, emb2, dim=0)
        return d

    s1 = similarity('green', 'blue')
    s2 = similarity('cloud', 'table')

    assert s1 > s2, (s1, s2)


if __name__ == '__main__':
    check_model()
    check_ptratrained_embedding()
