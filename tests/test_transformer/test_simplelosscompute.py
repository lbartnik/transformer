import torch
from transformer import SimpleLossCompute
from unittest.mock import MagicMock

from transformer import SimpleLossCompute

def test_simplelosscompute():
    crit = MagicMock(return_value=MagicMock(data=[1], backward=lambda: None))
    slc = SimpleLossCompute(crit)

    nn_output = torch.Tensor([
        [
            [0, .1, .2, .7],
            [.1, 0, 0, .9],
            [0, .7, 0, .3]
        ],
        [
            [.1, 0, 0, .9],
            [0, 0, 0, 1],
            [.3, .7, 0, 0]
        ]
    ])
    target = torch.LongTensor([
        [3, 1, 2],
        [1, 3, 1]
    ])

    slc(nn_output, target, 6)

    # criterion is called with:
    #   nn_output cast as a 2D array no_sentences * no_words x target_vocab_size
    #   target cast as a 1D array of length no_sentences * no_words
    assert crit.called
    assert crit.call_args[0][0].shape == torch.Size([6, 4])
    assert crit.call_args[0][1].shape == torch.Size([6])