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
