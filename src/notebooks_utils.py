from typing import List, Tuple, Dict

import matplotlib.pyplot as plt
import numpy as np
import torch
from IPython.display import HTML, display_html
from torch import LongTensor, FloatTensor
from torch import nn


def display_predict(model: nn.Module,
                    batch: Tuple[LongTensor, FloatTensor],
                    itow: Dict[int, str],
                    pad_idx: int = 0,
                    ) -> None:
    document, gt = batch['features'], batch['targets']

    with torch.no_grad():
        output = model(document)
        pred = output['logits']
        w_score = output['w_scores']
        s_score = output['s_scores']

    conf = 2 * abs(.5 - float(pred))

    sign = 1 if bool(pred > .5) else -1
    document = document.cpu()

    pred_str = 'positive' if sign == 1 else 'negative'
    print(f'Predict: {pred_str} (confedence: {round(conf, 3)})')

    if gt is not None:
        gt_str = 'positive' if bool(gt) else 'negative'
        print(f'Ground truth: {gt_str}.')

    for i_sent in range(document.shape[1]):
        sent = document[0, i_sent, :]
        mask = sent != pad_idx

        # word level scores
        sent_str = [(itow[int(word)], float(score)) for word, score in
                    zip(sent[mask], sign * w_score[0, i_sent, mask])]

        # sentence level score
        sent_score = float(sign * s_score[0, i_sent])
        sent_str.insert(0, (f'Sent {i_sent + 1} | ', sent_score))

        display_weighted_sent(sent_str)


def display_weighted_sent(weighted_words: List[Tuple[str, float]]) -> None:
    cmap = plt.get_cmap('bwr')
    token_str = '<span style="background-color: {color_hex}">{token}</span>'
    font_style = 'font-size:14px;'
    hex_str = '#%02X%02X%02X'
    html_str = '<p style="{}">{}</p>'

    def color_hex(weight: float) -> str:
        rgba = cmap(1. / (1 + np.exp(weight)), bytes=True)
        return hex_str % rgba[:3]

    tokens_html = [
        token_str.format(token=t, color_hex=color_hex(w))
        for t, w in weighted_words
    ]

    raw_html = html_str.format(font_style, ' '.join(tokens_html))
    display_html(HTML(raw_html))
