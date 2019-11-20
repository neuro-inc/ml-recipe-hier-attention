from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torchtext
from torch import nn, Tensor, LongTensor, FloatTensor
from torch import sigmoid, tanh
from torch.nn.functional import cosine_similarity, softmax, relu

from src.const import VECTORS_CACHE, IMBD_ROOT
from src.dataset import ImdbReviewsDataset


class HAN(nn.Module):

    def __init__(self,
                 vocab: Dict[str, int],
                 freeze_emb: bool,
                 hid: int = 50,
                 hid_fc: int = 50
                 ):
        super().__init__()

        self._embedding = get_pretrained_embedding(vocab, freeze_emb)
        emb_size = self._embedding.weight.shape[1]

        # 1. Words representation
        self._gru_word = nn.GRU(input_size=emb_size, hidden_size=hid,
                                batch_first=True, bidirectional=True)
        self._attn_word = Attention(in_features=2 * hid, out_features=2 * hid)

        # 2. Sentences representation
        self._gru_snt = nn.GRU(input_size=2 * hid, hidden_size=hid,
                               bidirectional=True, batch_first=True)
        self._attn_snt = Attention(in_features=2 * hid, out_features=2 * hid)

        # 3. Classification
        self._fc1 = nn.Linear(in_features=2 * hid, out_features=hid_fc)
        self._fc2 = nn.Linear(in_features=hid_fc, out_features=1)

        # Saving args for convinient restoring from ckpt
        self._params = {
            'vocab': vocab, 'freeze_emb': freeze_emb,
            'hid': hid, 'hid_fc': hid_fc
        }

    def forward(self,
                x: LongTensor,
                ) -> Tuple[Tensor, Optional[Tensor], Optional[Tensor]]:
        txt, snt, word = x.shape

        # 1. Words level
        x = self._embedding(x)  # [txt, snt, word, emb]

        x = x.view(-1, word, x.shape[3])  # [txt x snt, word, emb]

        x, _ = self._gru_word(x)  # [txt x snt, word, 2 x hid]

        x, w_scores = self._attn_word(x)  # [txt x snt, 2 x hid]

        # 2. Sentences level
        x = x.view(txt, snt, -1)  # [txt, snt, 2 x hid]

        w_scores = w_scores.view(txt, snt, -1)  # [txt, snt, word]

        x, _ = self._gru_snt(x)  # [txt, sent, 2 x hid]

        x, s_scores = self._attn_snt(x)  # [txt, 2 x hid]

        s_scores = s_scores.squeeze(-1)  # [txt, snt]

        # 3. Classification

        x = relu(self._fc1(x))  # [txt, hid_fc]

        x = self._fc2(x)  # [text, 1]

        x = sigmoid(x)  # [text, 1]

        return x, w_scores, s_scores

    def save(self, path_to_save: Path) -> None:
        checkpoint = {
            'state_dict': self.state_dict(),
            'params': self._params
        }
        torch.save(checkpoint, path_to_save)
        print(f'Model saved to {path_to_save}.')

    @classmethod
    def from_imbd_ckpt(cls, path_to_ckpt: Path) -> 'HAN':
        ckpt = torch.load(path_to_ckpt, map_location='cpu')
        model = cls(**ckpt['params'])
        model.load_state_dict(ckpt['state_dict'])
        print(f'Model was loaded from {path_to_ckpt}.')
        return model


class Attention(nn.Module):

    def __init__(self,
                 in_features: int,
                 out_features: int,
                 context_size: int = 100
                 ):
        super().__init__()

        self._fc = nn.Linear(in_features=in_features,
                             out_features=out_features,
                             bias=True)

        self._context = nn.Parameter(torch.randn((context_size, 1)).float())

    def forward(self, x: FloatTensor) -> FloatTensor:
        # [bs, seq, emb]

        x = tanh(self._fc(x))  # [bs, seq, hid]

        scores = softmax(x.matmul(self._context), dim=1)  # [bs, seq, 1]

        x = x.mul(scores).sum(dim=1)  # [bs, hid]

        return x, scores


def get_pretrained_embedding(vocab: Dict[str, int],
                             freeze_emb: bool
                             ) -> nn.Embedding:
    emb_size = 100
    glove = torchtext.vocab.GloVe(name='6B', dim=emb_size, cache=VECTORS_CACHE)
    glove.unk_init = lambda x: torch.ones(emb_size, dtype=torch.float32)

    vocab_size = len(vocab) + 1  # add 1 because of padding token
    weights = torch.zeros((vocab_size, emb_size), dtype=torch.float32)
    for word, idx in vocab.items():
        emb = glove.get_vecs_by_tokens([word])
        weights[idx, :] = emb

    weights[0, :] = glove.unk_init(None)
    embedding = nn.Embedding.from_pretrained(embeddings=weights,
                                             freeze=freeze_emb)
    return embedding


def check_model() -> None:
    vocab = {'cat': 1, 'dog': 2, 'bird': 3}  # 0 reserved for padding
    model = HAN(vocab=vocab, freeze_emb=True)
    batch = torch.randint(low=1, high=len(vocab),
                          size=(16, 12, 10), dtype=torch.int64
                          )
    output, _, _ = model(batch)

    torch.all(0 < output)
    torch.all(1 > output)


def check_pretrained_embedding() -> None:
    vocab = ImdbReviewsDataset.get_imdb_vocab(IMBD_ROOT)
    embedding = get_pretrained_embedding(vocab=vocab, freeze_emb=True)

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
    check_pretrained_embedding()
