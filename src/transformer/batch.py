from .subsequentmask import subsequent_mask

class Batch:
    "Object for holding a batch of data with mask during training."
    def __init__(self, src, trg=None, pad=0):
        self.src = src

        # WHY???
        # unsqueeze(1) adds a new dimension between dimensions associated
        # with mini-batch (0) and phrase (1)
        self.src_mask = (src != pad).unsqueeze(-2)

        # trg is None during scoring and not None during training
        if trg is not None:
            # trg is inputs to the decoder...
            self.trg = trg[:, :-1]
            # ...trg_y is output of the decoder; combined with applying mask
            # in the decoder, this means that we train the decoder to predict
            # the next word in the sequence
            self.trg_y = trg[:, 1:]
            self.trg_mask = self.make_std_mask(self.trg, pad)
            # how many tokens in the whole batch
            self.ntokens = (self.trg_y != pad).data.sum()
    
    @staticmethod
    def make_std_mask(tgt, pad):
        "Create a mask to hide padding and future words."
        # WHY???
        # same as with src_mask, we add a dimension between mini-batch and phrase
        tgt_mask = (tgt != pad).unsqueeze(-2)
        # mask out positions filled with padding and everything ahead of the current
        # position; decoder is not allowed to know future words in the output sequence
        tgt_mask = tgt_mask & subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data)
        tgt_mask.requires_grad_(False)
        return tgt_mask