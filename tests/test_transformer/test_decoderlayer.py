import torch
from transformer import DecoderLayer, MultiHeadedAttention, PositionwiseFeedForward, subsequent_mask, decoderlayer

def test_decoderlayer():
    torch.manual_seed(0)

    d = decoderlayer(4, 2, 12, .1)

    # each Tensor is a batch of one phrase, of three words, of four dimensions each
    # target phrase
    input = torch.Tensor([[1, 2, 0, 1], [-1, 1, -.1, .2], [.1, 3, .1, -.4]]).unsqueeze(0)
    # output from the encoder
    memory = torch.Tensor([[-1, .5, -.1, 1], [2, .1, -3, .4], [.3, -1, 2, .5]]).unsqueeze(0)
    # mask for the memory
    src_mask = None
    # mask for the decoder itself; size is the number of words in a phrase
    tgt_mask = subsequent_mask(3)

    expected = torch.Tensor([[ 0.6869,  2.5372, -0.1560,  0.0416],
        [-0.9674,  0.9857, -0.4791, -0.6867],
        [-0.3790,  3.9383, -0.4565, -0.3852]]).unsqueeze(0)

    output = d(input, memory, src_mask, tgt_mask)
    assert output.allclose(expected, rtol=0, atol=0.001)
