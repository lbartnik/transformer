import torch
import math
import torch.nn.functional as F

def attention(query, key, value, mask=None, dropout=None):
    """
    Compute 'Scaled Dot Product Attention'

    In classic attention mechanisms:
      - query: comes from the decoder
      - key: is in fact multiple keys whose affinity to query determines the weights
      - value: the same number of values as keys; summed element-wise with the weights
               coming from affinities between keys and the query

    In this implementation, in the case of self-attention, query, key and value are
    the same input matrix multiplied by respective weight matrices

    :param query: Tensor, batch_size x no_heads x phrase_length x d_k
    :param key:   Tensor, batch_size x no_heads x phrase_length x d_k
    :param value: Tensor, batch_size x no_heads x phrase_length x d_v
    :param mask:  Tensor, phrase_length x phrase_length; where mask == 0, weights will
                  be zeroed-out before applying softmax
    :param dropout: dropout probability, applied after applying the mask
    :return: Tensor batch_size x no_heads x phrase_length x d_v; context vectors for
             encoder/decoder which will be concatenated into a Tensor of shape
             batch_size x no_heads x n_features and pushed through a n_features x n_features
             linear layer before becoming the final output of the Multi-Headed Attention layer.
    """
    # d_k is the size of the word embedding space
    d_k = query.size(-1)
    
    # matrix-multiplies query against transposed keys; results in affinity score
    # for each word in phrase to all words in that phrase (in case of self-attention);
    # if mini-batched, each item in the batch is a separate matrix multiplication
    # (see https://pytorch.org/docs/stable/generated/torch.matmul.html)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

    # in case of encode-decoder attention, removes (zeroes-out) weights for words
    # ahead of the current position in the output and input phrases;
    # will be None in case of self-attention
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    
    # compute softmax in each row; these are normalized affinity/attention scores
    # between the given word (row index) and all words in the phrase (column
    # index)
    p_attn = F.softmax(scores, dim = -1)

    # only while training
    if dropout is not None:
        p_attn = dropout(p_attn)
    
    # summarize value vectors element-wise, using weights resulting from key-query
    # affinity/attention; also return p_attn to visualize attention weights
    return torch.matmul(p_attn, value), p_attn
