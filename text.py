import src.transformer as t
import pickle

vocab_en, vocab_de = t.vocab()
train_indices, test_indices = t.indices(vocab_en, vocab_de)

with open("tt.data", "wb") as out:
    pickle.dump((vocab_en, vocab_de, train_indices, test_indices), out)


with open("tt.data", "rb") as input:
    vocab_en, vocab_de, train_indices, test_indices = pickle.load(input)

trans = t.Translation(len(vocab_en), len(vocab_de))
trans.train(train_indices, test_indices, batch_tokens=10000)
