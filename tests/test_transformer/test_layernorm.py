import torch
import pytest
from transformer import LayerNorm

def test_normalizes_last_dimension():
    l = LayerNorm(3)
    input = torch.Tensor([[1,2,3], [4,5,6], [5,7,9]])
    normalized = l(input)

    assert normalized.shape == (3,3)

    # first row: ([1,2,3] - mean([1,2,3])) / stddev([1,2,3])
    #   mean([1,2,3]) = 2
    #   stddev([1,2,3]) = sqrt(var([1,2,3])) = 
    #   var([1,2,3]) = 1/(3-1) * {(1 - mean([1,2,3]))^2 + (2 - mean([1,2,3]))^2 + (3 - mean([1,2,3]))^2}
    #                = 1/2 * ((1-2)^2 + (2-2)^2 + (3-2)^2)
    #                = 1/2 * (1^2 + 0^2 + 1^2)
    #                = 1/2 * (1 + 0 + 1)
    #                = 1
    expected = (torch.Tensor([1,2,3]) - 2) / 1
    assert normalized[0,0].item() == pytest.approx(expected[0].item())
    assert normalized[0,1].item() == pytest.approx(expected[1].item())
    assert normalized[0,2].item() == pytest.approx(expected[2].item())

    # second row: [4,5,6] -> [-1, 0, 1]
    normalized[:,1].data == pytest.approx(expected.data)

    # third row
    expected = (torch.Tensor([5,7,9]) - 7) / torch.Tensor([5,7,9]).std()
    normalized[:,1].data == pytest.approx(expected.data)


def test_normalizes_with_minibatch():
    l = LayerNorm(3)
    # input has 3 dimensions which means it is a batch of phrases;
    # words are embedded in 2-dimension space; each phrase has 3 words;
    # there are 2 minibatches
    input = torch.Tensor([[[1,2,3], [1,2,4]], [[1,3,8], [1,8,9]]])
    normalized = l(input)

    normalized.data == pytest.approx(torch.Tensor([
        [[-1, 0, 1], [-0.8729, -0.2182,  1.0911]],
        [[-0.8321, -0.2774,  1.1094], [-1.1471,  0.4588,  0.6882]]
    ]))
