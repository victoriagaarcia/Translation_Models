# Translation Models
The goal of this project is to build a translation model that is capable of translating between
two languages. The model will be based on the Seq2Seq architecture, which consists of two
main components: an encoder and a decoder. The former takes an input sequence and encodes
it into a fixed-size representation, while the latter takes this representation and decodes it into
an output sequence.

Moreover, the model will be trained using embeddings to represent the words in the input and
output languages. This will allow the model to treat words as vectors in a continuous space,
which can help capture the semantic relationships between them. This fulfills the requirements
included in the given project description.

In order to evaluate the available options to build the model, we have conducted research to understand different approaches and techniques that can be valuable to build a translation model.

Some of these include, among others, the use of attention mechanisms and transformers.
The project will be implemented in Python using the PyTorch library. Once the model is built,
it will be trained on a dataset of parallel sentences in two languages. Subsequently, the model
will be evaluated on a separate test set to assess its performance. Based on the results, we will
analyze the strengths and weaknesses of the model and propose potential improvements.
