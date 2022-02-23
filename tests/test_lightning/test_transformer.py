import torch
import warnings

from lightning import Transformer


def test_transformer():
    t = Transformer(10, 10, d_model=4, h=2, ff_dim=4)
    src = torch.LongTensor([[1,2,3,4], [5,6,7,8]])
    trg = torch.LongTensor([[2,3,4,5], [6,7,8,9]])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        t.training_step((src, trg))
