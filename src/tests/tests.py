import torch
from torch.nn.functional import cosine_similarity

from src.const import IMBD_ROOT
from src.dataset import ImdbReviewsDataset, get_test_dataset, collate_docs
from src.model import HAN, get_pretrained_embedding


def test_model() -> None:
    vocab = {'cat': 1, 'dog': 2, 'bird': 3}  # 0 reserved for padding
    model = HAN(vocab=vocab, freeze_emb=True)
    batch = torch.randint(low=1, high=len(vocab),
                          size=(16, 12, 10), dtype=torch.int64
                          )
    output, _, _ = model(batch)

    assert torch.all(0 <= output)
    assert torch.all(1 >= output)


def test_pretrained_emb() -> None:
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


def test_forward_for_dataset() -> None:
    # data
    train_set = get_test_dataset()
    ids = [2, 5]
    docs, labels = collate_docs([train_set[i] for i in ids])
    n_doc, n_snt, n_wrd = docs.shape

    # model
    model = HAN(vocab=train_set.vocab, freeze_emb=True)

    # forwad
    pred, w_scores, s_scores = model(docs)

    assert pred.numel() == n_doc
    assert w_scores.shape == docs.shape
    assert s_scores.shape == (n_doc, n_snt)
