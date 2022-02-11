import torch
from .subsequentmask import subsequent_mask

def greedy_decode(model, src, src_mask, max_len, start_symbol):
    """
    Generates the output/decoded/translated sequence given the input/source sequence.

    :param model: EncoderDecoder object
    :param src: tokens (LongTensor)
    :param src_mask: True where token is not padding, False for padding tokens
    :param max_len: number of decoding steps
    :param start_symbol: initial symbol in the output/translated sequence
    """
    memory = model.encode(src, src_mask)
    ys = torch.ones(1, 1).fill_(start_symbol).type_as(src.data)
    for i in range(max_len-1):
        out = model.decode(memory, src_mask, ys, subsequent_mask(ys.size(1)).type_as(src.data))
        prob = model.generator(out[:, -1])
        prob[0, ys[0][-1]] = -100 # do not return the same word as the last word in the input
        _, next_word = torch.max(prob, dim = 1)
        next_word = next_word.data[0]
        ys = torch.cat([ys, 
                        torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=1)
    return ys
