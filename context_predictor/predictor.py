import tensorflow as tf
import numpy as np
import collections


class Predictor:
    """
    NN to predict word context.

    Attributes
    ----------
    vocabulary_size : int
    learning_rate : float, optional

    Methods
    -------
    train(X, y)
        Train the NN model.
    predict(X)
        Make prediction from X.

    """
    def __init__(self, vocabulary_size, learning_rate=0.1):
        self.vocabulary_size = vocabulary_size
        self.learning_rate = learning_rate
        self.x = tf.placeholder(tf.float32, [None, vocabulary_size])
        self.y = tf.placeholder(tf.float32, [None, vocabulary_size])

        self.weights = {
            'h1': tf.Variable(tf.random_uniform([self.vocabulary_size,
                                                 self.vocabulary_size]))
        }

        def perceptron(x, weights):
            layer_1 = tf.matmul(x, weights['h1'])
            layer_1 = tf.nn.sigmoid(layer_1)
            tmp_sum = tf.reduce_sum(layer_1)
            out_layer = tf.divide(layer_1, tmp_sum)
            return out_layer

        self.pred = perceptron(self.x, self.weights)
        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                                    logits=self.pred, labels=self.y))
        self.optimizer = tf.train.AdamOptimizer(
                         learning_rate=self.learning_rate).minimize(self.cost)
        self.init = tf.global_variables_initializer()
        self.sess = tf.Session()

    def init_model(self):
        """
        Just to execute `self.init`.

        """
        self.sess.run(self.init)

    def train(self, X, y):
        """
        Train the NN model.

        Parameters
        ----------
        X, y : array_like
            One-hot-encoding vectors of words and their contexts.

        Returns
        -------
        c : float
            Cost value.

        """
        _, c = self.sess.run([self.optimizer, self.cost],
                             feed_dict={self.x: X, self.y: y})
        return c

    def predict(self, X):
        """
        Make prediction from `X`. Predict context for OHE of words in `X`.

        Parameters
        ----------
        X : array_like
            One-hot-encoding vectors of words.

        Returns
        -------
        preds : array_like
            Context - the probabilities of each word in vocabulary to be
            in the context of words in `X`.

        """
        preds = self.sess.run([self.pred], feed_dict={self.x: X})
        return preds

    def close(self):
        """
        Close the `self.sess`.

        """
        self.sess.close()


def read_data(filename):
    """
    Read text data from file.

    Parameters
    ----------
    filename : str

    Returns
    -------
    data : list
        List of pieces of texts.

    """
    with open(filename, 'r') as data_file:
        data = data_file.readlines()
    return data


def good_dots(text):
    """
    Replace dots with commas in order to split
    sentences correctly by dots.

    Parameters
    ----------
    text : str

    Returns
    -------
    transformed_text : str

    Examples
    --------
    >>> good_dots('score (+/-SD) was 1.8+/-1.2. Prothrombin times')
    'score (+/-SD) was 1,8+/-1,2. Prothrombin times'

    """
    #
    # TODO: replace with regular expression
    #
    # >>> import re
    # >>> test = 'score (+/-SD) was 1.8+/-1.2. Prothrombin times'
    # >>> re.sub('\.\S', ',', test)
    # 'score (+/-SD) was 1,+/-1,. Prothrombin times'
    #
    def is_dot(c):
        return c == '.'

    def is_right_space(c):
        return c == ' '

    transformed_text = ''
    for index in range(len(text) - 1):
        if is_dot(text[index]) and not is_right_space(text[index+1]):
            transformed_text += ','
        else:
            transformed_text += text[index]

    transformed_text += text[-1]
    # To avoid dot in the end of string.
    transformed_text = transformed_text.strip('.')
    return transformed_text


def text_preproc(input_text):
    """
    Make preprocessing of input text.

    Parameters
    ----------
    input_data : str

    Returns
    -------
    sentences_stripped : list
        List of sentences. Sentence is a list of words.

    """
    sentences = input_text.lower().split('.')
    sentences_splitted = list(map(lambda x: x.split(), sentences))

    def strip_words(list_input):
        return list(map(lambda x: x.strip('<>():,.;!?'), list_input))

    sentences_stripped = list(map(lambda x: strip_words(x),
                                  sentences_splitted))
    return sentences_stripped


def clean_data(data_input):
    """
    Prepare text data for dataset building.

    Parameters
    ----------
    data_input : list
        List of pieces of texts.

    Returns
    -------
    res_data : list
        List of sentences. Sentence is a list of words.

    """
    data_good_dots = list(map(lambda x: good_dots(x), data_input))
    res_data = list(map(lambda x: text_preproc(x), data_good_dots))
    return res_data


def build_vocabulary(texts, vocabulary_size, stopwords_input):
    """
    Build vocabulary with size of vocabulary_size.
    Use only top-n frequent words.

    Parameters
    ----------
    texts : list
        List of strings.
    vocabulary_size : int
    stopwords_input : list
        List of words to remove from dictionary.

    Returns
    -------
    res_vocab : dict
        Vocabulary of words: {word: id, ...}.
    reversed_vocab : dict
        Reversed vocabulary: {id: word, ...}.

    """
    concat_sent = []
    for sentence in texts:
        concat_sent.extend(sentence.lower().split())
    concat_sent = list(map(lambda x: x.strip('<>():,.;!?'), concat_sent))
    count = collections.Counter(concat_sent)

    # Remove stop-words from vocabulary.
    for word in stopwords_input:
        if word in count:
            count.pop(word)

    # Get top-n words (-1 because of <UNK>)
    list_vocab = list(map(lambda x: x[0],
                          count.most_common(vocabulary_size-1)))
    res_vocab = {'UNK': 0}
    for word in list_vocab:
        res_vocab[word] = len(res_vocab)

    reversed_vocab = dict(zip(res_vocab.values(), res_vocab.keys()))
    return res_vocab, reversed_vocab


def build_dataset(sentence, vocab, max_dist=3):
    """
    Build dataset from words of `sentence`.

    Parameters
    ----------
    sentence : list
        List of words.
    vocab : dict
    max_dist : int, optional
        Maximum distance between input word and word form context.

    Returns
    -------
    dataset : list
        List of tuples: [(word, [context_word_1, ..., context_word_m]), ...].

    Examples
    --------
    >>> build_dataset(['background', 'rivaroxaban', 'is', 'currently',
    'used', 'to'])
    [('background', ['rivaroxaban', 'is', 'currently']),
     ('rivaroxaban', ['background', 'is', 'currently', 'used']),
     ('is', ['background', 'rivaroxaban', 'currently', 'used', 'to']),
     ('currently', ['background', 'rivaroxaban', 'is', 'used', 'to']),
     ('used', ['rivaroxaban', 'is', 'currently', 'to']),
     ('to', ['is', 'currently', 'used'])]

    """
    dataset = []

    for i in range(len(sentence)):
        indexes = list(filter(lambda x: x >= 0 and x < len(sentence),
                              np.arange(i - max_dist, i + max_dist + 1)))
        indexes.remove(i)
        tmp = []
        for index in indexes:
            if sentence[index] in vocab:
                tmp.append(sentence[index])
        if sentence[i] in vocab and len(tmp) > 0:
            dataset.append((sentence[i], tmp))

    return dataset


def get_ohe(words, vocab_input):
    """
    Compute one-hot-encoding for every word in `words`
    and summurize them into `res`.

    Parameters
    ----------
    words : list
        List of words.
    vocab_input : dict
        Vocabulary of words: {word: id, ...}

    Returns
    -------
    res : array_like
        Summarized one-hot-encoding vectors of every word.

    """
    res = np.zeros(len(vocab_input))
    for word in words:
        # One-hot-encoding of `word`.
        tmp_ohe = np.zeros(len(vocab_input))
        if word in vocab_input:
            tmp_ohe[vocab_input[word]] = 1
        else:
            tmp_ohe[0] = 1  # In case of <UNK>.
        res += tmp_ohe

    return res


def normalize_ohe(vector_input):
    """
    Normalize one-hot-encoding vector.
    Divide input vector by the sum of components.

    Parameters
    ----------
    vector_input : array_like

    Returns
    -------
    Normalized vector.

    Examples
    --------
    >>> normalize_ohe([1, 3, 4])
    array([ 0.125,  0.375,  0.5  ])

    """
    return vector_input / np.sum(vector_input)


if __name__ == '__main__':
    assert(0)
