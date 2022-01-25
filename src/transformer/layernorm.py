import torch
import torch.nn as nn

# From https://arxiv.org/pdf/1607.06450.pdf:
#
# Training state-of-the-art, deep neural networks is computationally expensive. One
# way to reduce the training time is to normalize the activities of the neurons. A
# recently introduced technique called batch normalization uses the distribution of
# the summed input to a neuron over a mini-batch of training cases to compute a
# mean and variance which are then used to normalize the summed input to that
# neuron on each training case. This significantly reduces the training time in
# feedforward neural networks. However, the effect of batch normalization is dependent
# on the mini-batch size and it is not obvious how to apply it to recurrent neural
# networks. In this paper, we transpose batch normalization into layer normalization by
# computing the mean and variance used for normalization from all of the summed
# inputs to the neurons in a layer on a single training case. Like batch normalization,
# we also give each neuron its own adaptive bias and gain which are applied after
# the normalization but before the non-linearity. Unlike batch normalization, layer
# normalization performs exactly the same computation at training and test times.
# It is also straightforward to apply to recurrent neural networks by computing the
# normalization statistics separately at each time step. Layer normalization is very
# effective at stabilizing the hidden state dynamics in recurrent networks. Empirically,
# we show that layer normalization can substantially reduce the training time
# compared with previously published techniques.

class LayerNorm(nn.Module):
    """
    Construct a layernorm module (See https://arxiv.org/pdf/1607.06450.pdf).

    :param n_features: the number of inputs of the layer being normalized
    :param eps: used to avoid dividing by zero when normalizing with std dev
    """
    def __init__(self, n_features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(n_features))
        self.b_2 = nn.Parameter(torch.zeros(n_features))
        self.eps = eps

    def forward(self, x):
        # each column (Tensor dimension -1) is an input data point
        # each row is (Tensor dimension -2) a dimension in the feature space
        # those two dimensions constitute a single phrase
        # Tensor dimension -3 is the mini batch index
        #
        # mean and std are computed in each row, removing the "column" dimension.
        # In R, where we deal primarily with matrices, we would specify that operation
        # as apply(X, 1, fun) which means "row by row". In numpy/torch, where we deal
        # with multi-dimensional arrays, it is more convenient to specify the dimension
        # that "disappears" rather than the dimension that remains. Hence, we tell torch
        # to reduce dimension "-1" which is columns - which means computing mean and std
        # row after row.
        # 
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
