
class SimpleLossCompute:
    "A simple loss compute and train function."
    def __init__(self, criterion, opt=None):
        #self.generator = generator
        self.criterion = criterion
        self.opt = opt
        
    def __call__(self, x, y, norm):
        """
        Use criterion to compute the loss, invoke back-propagation.

        :param x: output of the neural network
        :param y: expected tokens
        :param norm: the number of tokens in the whole mini-batch
        """
        # model itself calls the generator
        #x = self.generator(x)

        # cast output of the NN (x) as 2D Tensor of shape: [batch_size * phrase_length, target_vocab_size]
        # cast target (y) as 1D Tensor of shape: [batch_size * phrase_length]
        #
        # loss per single token: normalization is necessary if we want to maintain a stable
        # rate of learning - since the value of loss translates directly to the step taken by
        # SGD, we want to maintain similar sizes of the optimizer step across all mini-batches
        # in an epoch
        loss = self.criterion(x.contiguous().view(-1, x.size(-1)), 
                              y.contiguous().view(-1)) / norm
        
        # back-propagate the error
        loss.backward()

        if self.opt is not None:
            self.opt.step()
            self.opt.optimizer.zero_grad()
        
        # reverse the earlier normalization and return the total loss
        return loss.item() * norm
