import torch
from transformer import Decoder, decoderlayer, subsequent_mask

def test_decoderlayer():
    torch.manual_seed(0)

    d = Decoder(decoderlayer(4, 2, 12, .1), 6)

    # each Tensor is a batch of one phrase, of three words, of four dimensions each
    # target phrase
    input = torch.Tensor([[1, 2, 0, 1], [-1, 1, -.1, .2], [.1, 3, .1, -.4]]).unsqueeze(0)
    # output from the encoder
    memory = torch.Tensor([[-1, .5, -.1, 1], [2, .1, -3, .4], [.3, -1, 2, .5]]).unsqueeze(0)
    # mask for the memory
    src_mask = None
    # mask for the decoder itself; size is the number of words in a phrase
    tgt_mask = subsequent_mask(3)

    expected = torch.Tensor([[[ 0.0570,  1.3645, -0.4611, -0.9603],
         [-0.2065,  1.4086, -0.2471, -0.9550],
         [ 0.0178,  1.3885, -0.4995, -0.9068]]])

    output = d(input, memory, src_mask, tgt_mask)
    assert output.allclose(expected, rtol=0, atol=.001)
