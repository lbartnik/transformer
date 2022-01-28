import torch
import torch.nn as nn
from transformer import Encoder, EncoderLayer, MultiHeadedAttention

# This test simply demonstrates how to use this layer.
def test_encoder():
    torch.manual_seed(0)

    layer = EncoderLayer(4, MultiHeadedAttention(2, 4), nn.Linear(4,4), 0)
    e = Encoder(layer, 6)

    input = torch.Tensor([[.1,-1,2,2], [0,1,-1,0]]).unsqueeze(0)
    expected = torch.Tensor([[-0.9176, 1.3789, 0.0435, -0.5047], [0.9313, 0.6805, -1.2363, -0.3756]]).unsqueeze(0)

    output = e(input, mask=None)
    assert expected.allclose(output, atol=0.0001, rtol=0.0001)
