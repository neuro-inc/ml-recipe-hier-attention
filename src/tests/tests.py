import random
from collections import Counter

import torch
from torch.nn.functional import cosine_similarity

from src.const import IMBD_ROOT
from src.dataset import (
    ImdbReviewsDataset, get_test_dataset,
    collate_docs, SimilarRandSampler,
    TXT_CLIP, SNT_CLIP
)
from src.model import HAN, get_pretrained_embedding


def test_model() -> None:
    vocab = {'cat': 1, 'dog': 2, 'bird': 3}  # 0 reserved for padding
    model = HAN(vocab=vocab, freeze_emb=True)
    batch = torch.randint(low=1, high=len(vocab),
                          size=(16, 12, 10), dtype=torch.int64
                          )
    logits = model(batch)['logits']

    assert torch.all(0 <= logits)
    assert torch.all(1 >= logits)


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
    dataset = get_test_dataset()
    ids = [2, 5]
    docs = collate_docs([dataset[i] for i in ids])['features']
    n_doc, n_snt, n_wrd = docs.shape

    # model
    model = HAN(vocab=dataset.vocab, freeze_emb=True)

    # forwad
    output = model(docs)
    pred = output['logits']
    w_scores = output['w_scores']
    s_scores = output['s_scores']

    assert pred.numel() == n_doc
    assert w_scores.shape == docs.shape
    assert s_scores.shape == (n_doc, n_snt)


def test_sampler() -> None:
    for _ in range(20):
        max_len = random.randint(1, 20)
        n_txt = random.randint(1, 100)
        bs = random.randint(4, 8)
        diversity = random.randint(1, 2)

        lens = [random.randint(1, max_len) for _ in range(n_txt)]

        sampler = SimilarRandSampler(keys=lens, bs=bs, diversity=diversity)

        sampled_ids_flat = list(sampler)
        sampled_lens = [lens[i] for i in sampled_ids_flat]

        assert set(sampled_ids_flat) == set(list(range(n_txt)))
        assert Counter(sampled_lens) == Counter(lens)


def test_batch_size() -> None:
    bs = 384

    vocab = ImdbReviewsDataset.get_imdb_vocab(IMBD_ROOT)
    model = HAN(vocab=vocab, freeze_emb=True)
    model.cuda()

    batch = torch.ones((bs, TXT_CLIP, SNT_CLIP),
                       dtype=torch.int64).cuda()
    model(batch)
