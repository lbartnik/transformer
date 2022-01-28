import torch

def subsequent_mask(size):
    """
    Mask out subsequent positions.

    :param size: the number of rows and columns of the output matrix
    """
    attn_shape = (1, size, size)
    # this produces a matrix of booleans
    return torch.triu(torch.ones(attn_shape)).type(torch.uint8) == 0
