import torch
import torch.nn as nn
import pytest
from transformer import SublayerConnection, LayerNorm

def test_without_dropout():
    sc = SublayerConnection(2, 0)

    # sublayer passes the input to the output
    sublayer = nn.Linear(2, 2)
    sublayer.weight = nn.Parameter(torch.diag(torch.ones(2)))
    sublayer.bias = nn.Parameter(torch.zeros(2))

    input = torch.Tensor([[0, 1], [2, 3]])
    output = sc(input, sublayer)
    
    # input normalized: [[-0.7071, 0.7071], [-0.7071, 0.7071]]
    # input + input normalized: [[-0.7071, 1.7071], [1.2929, 3.7071]]
    # no dropout
    expected = torch.Tensor([[-0.7071,  1.7071], [1.2929,  3.7071]])
    assert output.allclose(expected, rtol=0, atol=0.001)
