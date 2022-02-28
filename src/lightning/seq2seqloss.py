import torch.nn as nn

class Seq2SeqLoss(nn.Module):
    def __init__(self, padding_idx=0):
        super(Seq2SeqLoss, self).__init__()
        self.loss = nn.NLLLoss(reduction='sum')
        self.padding_idx = padding_idx
        self.ntokens = None
    
    def forward(self, input, target):
        r"""
        Args:
            input (Tensor): 3D tensor (batch, seq, vocab) which contains log-probabilities
                of tokens in the output sequences
            target (Tensor): 2D tensor with expected output token indices
        """
        self.ntokens = (target != self.padding_idx).sum().float().item()
        return self.loss(input.view(-1, input.size(-1)), target.contiguous().view(-1)) / self.ntokens
