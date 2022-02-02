import torch
from transformer import PositionalEncoding

def test_encoding():
    torch.manual_seed(0)

    pe = PositionalEncoding(2, 0, 3)
    input = torch.zeros((1, 3, 2))
    expected = torch.Tensor([[0.0000, 1.0000], [0.8415, 0.5403], [0.9093, -0.4161]]).unsqueeze(0)
    
    output = pe(input)
    assert output.allclose(expected, rtol=0, atol=0.001)
    