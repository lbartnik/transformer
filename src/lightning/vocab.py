import spacy
import torchtext
from torchtext.datasets import IWSLT2017
from torchtext.vocab import build_vocab_from_iterator
from tqdm.auto import tqdm

import os
import pickle

# Define special symbols and indices
PAD_IDX, UNK_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3
# Make sure the tokens are in order of their indices to properly insert them in vocab
SPECIAL_SYMBOLS = ['<pad>', '<unk>', '<bos>', '<eos>']


class SpacyTokenizer:
    def __init__(self, spacy):
        self.spacy = spacy
    def __call__(self, text):
        return [t.text for t in self.spacy(text.strip())]

def spacy_tokenizers():
    return SpacyTokenizer(spacy.load("en_core_web_sm")), SpacyTokenizer(spacy.load("de_core_news_sm"))

def build_vocabs(tokenizers=None):
    if tokenizers is None:
        tokenizers = spacy_tokenizers()
    
    en, de = tokenizers
    lines = torchtext.datasets.iwslt2017.NUM_LINES['train']['train'][('de','en')]

    def build_vocab(iter, tokenizer, max_len=100):
        filtered = filter(lambda t: len(t) > 0 and len(t) <= max_len, map(tokenizer, iter))
        progress = tqdm(filtered, total=lines)
        vocab = build_vocab_from_iterator(progress, min_freq=2, specials=SPECIAL_SYMBOLS)
        vocab.set_default_index(vocab['<unk>'])
        return vocab

    train_iter = IWSLT2017(split='train', language_pair=('en', 'de'))
    vocab_en = build_vocab(map(lambda p: p[0], train_iter), en)

    train_iter = IWSLT2017(split='train', language_pair=('en', 'de'))
    vocab_de = build_vocab(map(lambda p: p[1], train_iter), de)

    return vocab_en, vocab_de

def vocabs(en=None, de=None, path='vocabs.bin'):
    if os.path.exists(path):
        with open(path, "rb") as pickled:
            en, de = pickle.load(pickled)
    else:
        en, de = build_vocabs((en, de))
        with open(path, "wb") as output:
            pickle.dump((en, de), output)
    return en, de


class TranslationVocab:
    pass


from tokenizers import normalizers
from tokenizers.normalizers import NFD, StripAccents, Lowercase, Strip
from tokenizers.pre_tokenizers import Whitespace
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
import numpy as np


class FlatteningIterator:
    def __init__(self, it):
        self.it = it
        self.el = None
    def __iter__(self):
        return self
    def __next__(self):
        if not self.el:
            self.el = iter(next(self.it))
        try:
            return next(self.el)
        except StopIteration:
            self.el = iter(next(self.it))
        return next(self.el)

class HuggingfaceTranslationVocab:
    def __init__(self, hf_tokenizer, csp, vocab_size):
        self.vocab_size = vocab_size
        self.csp = csp
        self.tokenizer = hf_tokenizer

    def train(self, data_iter):
        """
        Args:
            data_iter: function which returns an iterator to a data set
        """
        trainer = BpeTrainer(special_tokens=SPECIAL_SYMBOLS,
                        vocab_size=self.vocab_size,
                        continuing_subword_prefix=self.csp)
        self.tokenizer.train_from_iterator(FlatteningIterator(data_iter), trainer=trainer)

    def encode(self, pair):
        src, tgt = self.encode_raw(pair)
        return (src.ids, tgt.ids)
    
    def encode_raw(self, pair):
        src, tgt = pair
        return (self.tokenizer.encode(src), self.tokenizer.encode(tgt))
    
    def decode(self, tgt):
        tokens = self.tokenizer.decode(tgt)
        return ' '.join(tokens).replace(' ' + self.csp, '')

    def numericalize(self, data_iter):
        ans = []
        for src, tgt in data_iter:
            src = [BOS_IDX] + self.tokenizer.encode(src).ids + [EOS_IDX]
            tgt = [BOS_IDX] + self.tokenizer.encode(tgt).ids + [EOS_IDX]
            ans.append((np.array(src), np.array(tgt)))
        return ans

    def __len__(self):
        return self.tokenizer.get_vocab_size()

    def save(self, path):
        self.tokenizer.save(path)

    @staticmethod
    def default_tokenizer(csp='_', vocab_size=37000):
        tokenizer = Tokenizer(BPE(unk_token='<unk>', continuing_subword_prefix=csp))
        tokenizer.normalizer = normalizers.Sequence([Strip(), NFD(), StripAccents(), Lowercase()])
        tokenizer.pre_tokenizer = Whitespace()

        return HuggingfaceTranslationVocab(tokenizer, csp, vocab_size)

    @staticmethod
    def load(path, csp='_'):
        tokenizer = Tokenizer.from_file(path)
        return HuggingfaceTranslationVocab(tokenizer, csp, tokenizer.get_vocab_size())
