from spacy.lang.en import English
from spacy.lang.de import German
from torchtext.datasets import IWSLT2017
from torchtext.vocab import build_vocab_from_iterator


# Define special symbols and indices
PAD_IDX, UNK_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3
# Make sure the tokens are in order of their indices to properly insert them in vocab
special_symbols = ['<pad>', '<unk>', '<bos>', '<eos>']


# --- build vocabularies ---

class Tokenizer:
    def __init__(self, spacy):
        self.spacy = spacy
    def __call__(self, text):
        return [t.text for t in self.spacy(text)]

def vocab():
    tokenizer_en = Tokenizer(English().tokenizer)
    tokenizer_de = Tokenizer(German().tokenizer)

    def build_vocab(iter, tokenizer, max_len=100):
        filtered = filter(lambda t: len(t) > 0 and len(t) <= max_len, map(tokenizer, iter))
        vocab = build_vocab_from_iterator(filtered, min_freq=2, specials=special_symbols)
        vocab.set_default_index(vocab['<unk>'])
        return vocab

    train_iter = IWSLT2017(split='train', language_pair=('en', 'de'))
    vocab_en = build_vocab(map(lambda p: p[0], train_iter), tokenizer_en)

    train_iter = IWSLT2017(split='train', language_pair=('en', 'de'))
    vocab_de = build_vocab(map(lambda p: p[1], train_iter), tokenizer_de)

    return vocab_en, vocab_de


# --- tokenize and encode texts ---

def indices(vocab_en, vocab_de):
    tokenizer_en = Tokenizer(English().tokenizer)
    tokenizer_de = Tokenizer(German().tokenizer)

    def to_indices(pair):
        src, tgt = pair
        return vocab_en(tokenizer_en(src)), vocab_de(tokenizer_de(tgt))

    train_iter = IWSLT2017(split='train', language_pair=('en', 'de'))
    test_iter = IWSLT2017(split='test', language_pair=('en', 'de'))

    train_indices = sorted(map(to_indices, train_iter), key = lambda x: (len(x[0]), len(x[1])))
    test_indices = sorted(map(to_indices, test_iter), key = lambda x: (len(x[0]), len(x[1])))

    return train_indices, test_indices

# --- batchify ---

import torch
import random
from .batch import Batch


def pad(lists, phrase_len):
    return torch.LongTensor([l + [PAD_IDX] * (phrase_len - len(l)) for l in lists])

class Batchify:
    def __init__(self, data, batch_tokens=1000, pad_idx=PAD_IDX, cuda=False):
        assert type(data) == list
        self.data = data
        self.batch_tokens = batch_tokens
        self.pad_idx = pad_idx
        self.cuda = cuda
        self.batches = None

    def next(self):
        if not self.batches:
            self._prepare_batches()
            self._batches_to_tensors()
            self.batches = iter(self.batches)
        
        src, tgt = next(self.batches)
        if self.cuda:
            src, tgt = src.cuda(), tgt.cuda()
        
        return Batch(src, tgt, self.pad_idx)
    
    def _prepare_batches(self):
        begin, end = 0, 0
        max_phrase_length = 0
        indices = []
        for src, trg in self.data:
            max_phrase_length = max(max_phrase_length, len(src) + len(trg) + 4)
            if (end-begin+1)*max_phrase_length > self.batch_tokens:
                indices.append((begin, end))
                begin = end
            end += 1
        indices.append((begin, end))
        random.shuffle(indices)
        self.batches = indices
    
    def _batches_to_tensors(self):
        ans = []
        for begin, end in self.batches:            
            src = []
            tgt = []

            for sr, tg in self.data[begin:end]:
                src.append([BOS_IDX] + sr + [EOS_IDX])
                tgt.append([BOS_IDX] + tg + [EOS_IDX])
            src = pad(src, max(map(len, src)))
            tgt = pad(tgt, max(map(len, tgt)))
            ans.append((src, tgt))
        self.batches = ans

    def __next__(self):
        return self.next()

    def __iter__(self):
        return self


# --- train model ---

from .makemodel import make_model
from .labelsmoothing import LabelSmoothing
from .noamopt import NoamOpt
from .runepoch import run_epoch
from .simplelosscompute import SimpleLossCompute
from .greedydecode import greedy_decode

class Translation:
    def __init__(self, src_vocab, tgt_vocab, padding_idx=PAD_IDX, cuda=False, model=None):
        if model:
            self.model = model
        else:
            self.model = make_model(src_vocab, tgt_vocab)
        
        if cuda:
            self.model.cuda()
        
        self.tgt_vocab = tgt_vocab
        self.padding_idx = padding_idx
        self.cuda = cuda

    def train(self, train, test, nepoch=10, batch_tokens=1000, base_lr=1, warmup=2000):
        criterion = LabelSmoothing(size=self.tgt_vocab, padding_idx=self.padding_idx, smoothing=.1)
        if self.cuda:
            criterion.cuda()
        
        model_opt = NoamOpt(self.model.src_embed[0].n_features, base_lr, warmup,
                torch.optim.Adam(self.model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))

        for epoch in range(nepoch):
            self.model.train()
            b = Batchify(train, batch_tokens=batch_tokens, cuda=self.cuda)
            run_epoch(b, self.model, SimpleLossCompute(criterion, model_opt))

            self.model.eval()
            b = Batchify(test, batch_tokens=batch_tokens, cuda=self.cuda)
            loss = run_epoch(b, self.model, SimpleLossCompute(criterion, None))
            print(f"Epoch {epoch} completed with validation loss per token {loss}")

            self.save("checkpoint.bin")

    def translate(self, src, start_symbol=BOS_IDX, max_len=5000):
        if type(src) is list:
            src = torch.LongTensor(src)
            if self.cuda:
                src.cuda()
            while len(src.shape) != 2:
                src.unsqueeze_(0)

        ans = greedy_decode(self.model, src, src != self.padding_idx, max_len, start_symbol)
        return ans[:ans.index(EOS_IDX)]
    
    def save(self, path):
        with open(path, "wb") as out:
            torch.save(self.model, out)
    
    @staticmethod
    def load(src_vocab, tgt_vocab, path, cuda=True):
        map_location = "cuda" if cuda else "cpu"
        model = torch.load(path, map_location=map_location)
        return Translation(src_vocab, tgt_vocab, cuda=cuda, model=model)



if __name__ == "__main__":
    vocab_en, vocab_de = vocab()
    train_indices, test_indices = indices(vocab_en, vocab_de)

    trans = Translation(len(vocab_en), len(vocab_de))
    trans.train(train_indices, test_indices)
