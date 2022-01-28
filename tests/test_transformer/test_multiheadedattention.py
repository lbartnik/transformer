import torch
import torch.nn as nn
from transformer import MultiHeadedAttention

# This test is not great at explaining the inner workings of the
# MultiHeadedAttention class. At the very least, however, it presents
# the shapes of valid inputs.
def test_multiheaded_attention():
    torch.manual_seed(0)

    a = MultiHeadedAttention(2, 4)
    # unsqueeze() so that each Tensor is a minibatch of 1 input.
    # MultiHeadedAttention assumes batched processing in how it juggles
    # matrix dimensions
    query = torch.Tensor([[1,0,0,1], [0,1,1,0], [1,1,1,0]]).unsqueeze(0)
    key   = torch.Tensor([[2,1,0,1], [0,2,0,1], [0,2,0,1]]).unsqueeze(0)
    value = torch.Tensor([[0,0,1,1], [1,1,1,1], [1,3,0,1]]).unsqueeze(0)

    output = a(query, key, value)

    exp = torch.Tensor([[1.0213, -0.0138, -0.0467, -0.3815],
                        [0.8357,  0.0605, -0.1077, -0.4362],
                        [1.0044,  0.0144, -0.0500, -0.3714]])
    assert output.allclose(exp, rtol=0, atol=0.001)

