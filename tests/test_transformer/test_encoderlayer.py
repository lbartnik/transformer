import torch
import torch.nn as nn
from transformer import EncoderLayer, MultiHeadedAttention

# This test simply demonstrates how to use this layer.
def test_encoderlayer():
    torch.manual_seed(0)

    # two heads in multi-headed attention + an feed-forward layer + zero dropout
    # as presented in figure 1 in "Attention Is All You Need" (https://arxiv.org/pdf/1706.03762.pdf)
    el = EncoderLayer(2, MultiHeadedAttention(2, 2), nn.Linear(2,2), 0)

    # one phrase consisting of two words
    # unsqueeze turns input into a mini-batch of size 1
    input = torch.Tensor([[1,2], [0,2]]).unsqueeze(0)

    # simply the output which matches this random seed
    expected = torch.Tensor([[0.7359, 2.5578], [-0.2713, 2.3813]]).unsqueeze(0)

    output = el(input, None)
    expected.allclose(output, atol=0.01, rtol=0)
