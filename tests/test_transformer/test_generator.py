import torch

from transformer import Generator

def test_generator():
    g = Generator(4, 5)

    input = torch.Tensor([
        [.1, -.1, 2, .5],
        [2, -3, 1, .1]
    ])
    log_probabilities = g(input)
    probabilities = log_probabilities.exp()

    rows = probabilities.sum(1)
    assert rows.shape == torch.Size([2])
    assert rows[0] == 1
    assert rows[1] == 1
