import torch
from transformer import PositionwiseFeedForward

def test_ffn():
    torch.manual_seed(0)

    n = PositionwiseFeedForward(2, 4)
    input = torch.Tensor([[1,2], [-1,2]])
    expectation = torch.Tensor([[-0.1650, 0.3326], [-0.2703, 0.1632]])

    output = n(input)
    output.allclose(expectation)
