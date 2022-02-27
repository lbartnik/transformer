import torch
import torchtext
import os
import pickle
import numpy as np
import torch.utils.data as data

from torchtext.datasets import IWSLT2017
from torch.nn.utils.rnn import pad_sequence
from tqdm.auto import tqdm

from .vocab import BOS_IDX, EOS_IDX, PAD_IDX, spacy_tokenizers


def numericalized(vocabs, path, wrap_in=(BOS_IDX, EOS_IDX)):
    if os.path.exists(path):
        with open(path, "rb") as pickled:
            train, val, test = pickle.load(pickled)
    else:
        val   = _numericalize(IWSLT2017(split='valid',language_pair=('en', 'de')), vocabs, wrap_in)
        test  = _numericalize(IWSLT2017(split='test', language_pair=('en', 'de')), vocabs, wrap_in)

        lines = torchtext.datasets.iwslt2017.NUM_LINES['train']['train'][('de','en')]
        iter = tqdm(IWSLT2017(split='train', language_pair=('en', 'de')), total=lines)
        train = _numericalize(iter, vocabs, wrap_in)

        with open(path, "wb") as output:
            pickle.dump((train, val, test), output)
    
    return train, val, test

def _numericalize(iter, vocabs, wrap_in):
    tokenizers = spacy_tokenizers()

    def _process(entry):
        numbers = [vocab(tokenizer(text)) for vocab, tokenizer, text in zip(vocabs, tokenizers, entry)]
        # add begin/end of sequence indices
        if wrap_in is not None and len(wrap_in) == 2:
            numbers = [[wrap_in[0]] + seq + [wrap_in[1]] for seq in numbers]
        return [np.array(seq) for seq in numbers]
    
    return list(map(_process, iter))


class TextDataset(data.Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def datasets(vocabs, path='data.bin'):
    ds = numericalized(vocabs, path)
    return [TextDataset(d) for d in ds]


def generate_batch(data_batch, padding_idx=PAD_IDX):
    src, tgt = map(list,zip(*data_batch))
    src, tgt = tuple(map(torch.LongTensor, src)), tuple(map(torch.LongTensor, tgt))
    src = pad_sequence(src, padding_value=padding_idx, batch_first=True)
    tgt = pad_sequence(tgt, padding_value=padding_idx, batch_first=True)
    return src, tgt


def dataloaders(vocabs, batch_size):
    train, val, test = datasets(vocabs)
    train = data.DataLoader(train, batch_size=batch_size, shuffle=True,  collate_fn=generate_batch)
    val   = data.DataLoader(val,   batch_size=batch_size, shuffle=False, collate_fn=generate_batch)
    test  = data.DataLoader(test,  batch_size=batch_size, shuffle=False, collate_fn=generate_batch)
    return train, val, test
