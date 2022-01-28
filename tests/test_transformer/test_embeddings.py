import torch
from transformer import Embeddings

def test_embeddings():
    torch.manual_seed(0)

    e = Embeddings(3, 20)
    output = e(torch.LongTensor([[1,3]]))

    expected = torch.Tensor([[-0.7515,  1.4700,  1.1986], [-2.1882,  0.6062,  0.5337]])
    assert output.allclose(expected, rtol=0, atol=0.001)
