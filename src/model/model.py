from typing import Dict, Optional, Tuple

import torch
import torchtext
from torch import nn, Tensor, LongTensor
from torch import sigmoid
from torch.nn.functional import cosine_similarity

from src.const import VECTORS_CACHE
from src.model.dataset import IMBD_ROOT, ImdbReviewsDataset


class HAN(nn.Module):
    _embedding: nn.Embedding

    def __init__(self, vocab: Dict[str, int]):
        super().__init__()

        hid, hid_fc = 100, 30
        self._embedding = get_pretrained_embedding(vocab)

        self._gru_word = nn.GRU(input_size=self._embedding.weight.shape[1],
                                hidden_size=hid, batch_first=True,
                                bidirectional=True)
        self._gru_sent = nn.GRU(input_size=2 * hid, hidden_size=2 * hid,
                                bidirectional=True, batch_first=True)

        self._attn_word = nn.MultiheadAttention(num_heads=1, embed_dim=2 * hid)
        self._attn_sent = nn.MultiheadAttention(num_heads=1, embed_dim=4 * hid)

        self._fc1 = nn.Linear(in_features=4 * hid, out_features=hid_fc)
        self._fc2 = nn.Linear(in_features=hid_fc, out_features=1)

    def forward(self,
                x: LongTensor,
                need_weights: bool = True
                ) -> Tuple[Tensor, Optional[Tensor], Optional[Tensor]]:
        txt, snt, word = x.shape

        # Words level
        x = self._embedding(x)  # [txt, snt, word, emb]

        x = x.view(-1, word, x.shape[3])  # [txt x snt, word, emb]

        x, _ = self._gru_word(x)  # [txt x snt, word, 2 x hid]

        x = x.permute(1, 0, 2)  # [word, txt x snt, 2 x hid]

        x, words_weights = self._attn_word(key=x, value=x, query=x,
                                           need_weights=need_weights
                                           )  # [word, txt x snt, 2 x hid]

        x = x.sum(dim=0)  # [txt x snt, 2 x hid]  TODO

        # Sentences level
        x = x.view(txt, snt, -1)  # [txt, snt, 2 x hid]

        x, _ = self._gru_sent(x)  # [txt, sent, 4 x hid]

        x = x.permute(1, 0, 2)  # [sent, txt, 4 x hid]

        x, sent_weights = self._attn_sent(key=x, value=x, query=x,
                                          need_weights=need_weights
                                          )  # [sent, txt, 4 x hid]

        x = x.sum(dim=0)  # [txt, 4 x hid]  TODO

        x = self._fc1(x)  # [txt, hid_fc]

        x = self._fc2(x)  # [text, 1]

        x = sigmoid(x)  # [text, 1]

        return x, words_weights, sent_weights


def get_pretrained_embedding(vocab: Dict[str, int]) -> nn.Embedding:
    emb_size = 100
    glove = torchtext.vocab.GloVe(name='6B', dim=emb_size, cache=VECTORS_CACHE)
    glove.unk_init = lambda x: torch.ones(emb_size, dtype=torch.float32)

    vocab_size = len(vocab) + 1  # add 1 because of padding token
    weights = torch.zeros((vocab_size, emb_size), dtype=torch.float32)
    for word, idx in vocab.items():
        emb = glove.get_vecs_by_tokens([word])
        weights[idx, :] = emb

    weights[0, :] = glove.unk_init(None)
    embedding = nn.Embedding.from_pretrained(embeddings=weights, freeze=False)
    return embedding


def check_model() -> None:
    vocab = {'cat': 1, 'dog': 2, 'bird': 3}  # 0 reserved for padding
    model = HAN(vocab)
    batch = torch.randint(low=1, high=len(vocab),
                          size=(16, 12, 10), dtype=torch.int64
                          )
    output, _, _ = model(batch, need_weights=False)

    torch.all(0 < output)
    torch.all(1 > output)


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
