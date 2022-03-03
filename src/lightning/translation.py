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


class BatchedTextDataset(TextDataset):
    def __init__(self, data, no_tokens):
        self.data = _batch_by_tokens(data, no_tokens)


def _batch_by_tokens(data, no_tokens, padding_idx=PAD_IDX):
    # sort sequences by source length and then by target length
    data = sorted(data, key = lambda x: (len(x[0]), len(x[1])))
    indices = _find_indices(data, no_tokens)
    return _collate(data, indices, padding_idx)


def _find_indices(data, no_tokens):
    # group sequences by their total padded length
    begin, end = 0, 0
    max_src_length = 0
    max_trg_length = 0
    indices = []
    for src, trg in data:
        max_src_length = max(max_src_length, len(src))
        max_trg_length = max(max_trg_length, len(trg))
        # if adding this sequence takes us above the limit, mark the batch boundary
        if (end-begin+1)*(max_src_length+max_trg_length) > no_tokens:
            indices.append((begin, end))
            max_src_length, max_trg_length = len(src), len(trg)
            begin = end
        end += 1
    # append the last batch
    indices.append((begin, end))
    return indices
    
def _collate(data, indices, padding_idx):
    ans = []
    for begin, end in indices:
        src, tgt = [], []

        for sr, tg in data[begin:end]:
            src.append(torch.LongTensor(sr))
            tgt.append(torch.LongTensor(tg))
        src = pad_sequence(src, padding_value=padding_idx, batch_first=True)
        tgt = pad_sequence(tgt, padding_value=padding_idx, batch_first=True)
        ans.append((src, tgt))
    return ans


def datasets(vocabs, path='data.bin'):
    ds = numericalized(vocabs, path)
    return [TextDataset(d) for d in ds]


def generate_batch(data_batch, padding_idx=PAD_IDX):
    src, tgt = map(list,zip(*data_batch))
    src, tgt = tuple(map(torch.LongTensor, src)), tuple(map(torch.LongTensor, tgt))
    src = pad_sequence(src, padding_value=padding_idx, batch_first=True)
    tgt = pad_sequence(tgt, padding_value=padding_idx, batch_first=True)
    return src, tgt


def data_loaders(vocabs, batch_size):
    train, val, test = datasets(vocabs)
    train = data.DataLoader(train, batch_size=batch_size, shuffle=True,  collate_fn=generate_batch)
    val   = data.DataLoader(val,   batch_size=batch_size, shuffle=False, collate_fn=generate_batch)
    test  = data.DataLoader(test,  batch_size=batch_size, shuffle=False, collate_fn=generate_batch)
    return train, val, test

# necessary to keep batch a 2D tensor; without this call DataLoader makes the already 2D tensor
# into a 3D one
def _unwrap_batch(data_batch):
    assert len(data_batch) == 1
    return data_batch[0]

def batched_data_loaders(vocabs, no_tokens):
    train, val, test = numericalized(vocabs, 'data.bin')
    return wrap_data(no_tokens, train, val, test)

def wrap_data(no_tokens, train, val, test):
    train = data.DataLoader(BatchedTextDataset(train, no_tokens), batch_size=1, shuffle=True,  collate_fn=_unwrap_batch)
    val   = data.DataLoader(BatchedTextDataset(val, no_tokens),   batch_size=1, shuffle=False, collate_fn=_unwrap_batch)
    test  = data.DataLoader(BatchedTextDataset(test, no_tokens),  batch_size=1, shuffle=False, collate_fn=_unwrap_batch)
    return train, val, test
