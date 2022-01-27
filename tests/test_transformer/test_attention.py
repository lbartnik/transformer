import torch

from transformer import attention

def test_attention():
    query = torch.diag(torch.Tensor(range(10)))
    key   = torch.diag(torch.Tensor(range(10)))
    value = torch.ones(10, 10)

    context, weights = attention(query, key, value)

    expected_weights = torch.Tensor([
        [.1, .1, .1, .1, .1, .1, .1, .1, .1, .1],
        [.096, .13, .096, .096, .096, .096, .096, .096, .096, .096],
        [.079, .079, .28, .079, .079, .079, .079, .079, .079, .079],
        [.038, .038, .038, .65, .038, .038, .038, .038, .038, .038],
        [.006, .006, .006, .006, .94, .006, .006, .006, .006, .006],
        [0, 0, 0, 0, 0, .99, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, .99, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
    ])
    context.allclose(torch.ones(10,10), rtol=0, atol=0.0001)
    weights.allclose(expected_weights, rtol=0, atol=0.0001)
