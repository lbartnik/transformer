# Transformer

A sample implementation of the DNN transformer architecture along with
unit tests which help understand the semantics of each component.

Original implementation: [nlp.seas.harvard.edu/2018/04/03/attention.html](https://nlp.seas.harvard.edu/2018/04/03/attention.html)


# Thoughts, Comments, Observations

## Phrase Lenght

The length of the phrase is never explicitly coded in the neural network. Instead,
matrix multiplications etc. are done in a way that assumes this dimension and
passes/uses it when necessary to arrive at the correct output. This is a very neat
trick which simplifies operating this particular implementation of the Transformer
architecture.

It seems to me, that this goes even further: the phrase length must be consistent
within a single mini-batch, but not between mini-batches. So, you can train this
Transformer on phrases of certain length and then use it to process sentences of
different length.
