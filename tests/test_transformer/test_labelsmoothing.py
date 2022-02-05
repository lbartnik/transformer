import torch
import pytest

from transformer import LabelSmoothing

def test_labelsmoothing():
    ls = LabelSmoothing(4, 0, .1)
    
    target = torch.LongTensor([[1, 2, 0], [3, 3, 0]])
    nn_output = torch.Tensor([
        [
            [0, .1, .3, .6],
            [0, 0, .7, .3],
            [0, .1, 0, .9]
        ],
        [
            [0, .2, .0, .8],
            [0, 0, 0, 1],
            [0, .9, 0, .1]
        ]
    ])

    # SimpleLossCompute transforms the output of NN and the target into 2D matrices
    target = target.contiguous().view(-1)
    nn_output = nn_output.contiguous().view(-1, nn_output.size(-1))

    output = ls(nn_output, target)

    expected_true_dist = torch.Tensor([
        [0.0000, 0.9000, 0.0500, 0.0500],
        [0.0000, 0.0500, 0.9000, 0.0500],
        [0.0000, 0.0000, 0.0000, 0.0000],
        [0.0000, 0.0500, 0.0500, 0.9000],
        [0.0000, 0.0500, 0.0500, 0.9000],
        [0.0000, 0.0000, 0.0000, 0.0000]
    ])
    assert (expected_true_dist == ls.true_dist).all()

    assert output.item() == pytest.approx(-3.9875907)