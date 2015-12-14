from math import log

class KNNClassifier:

  def __init__(self, training_set):
    self._vocabulary = {}
    word_id = 0

    for text, gender in training_set:
      for token in text:
        if token not in self._vocabulary.keys():
          self._vocabulary[token] = word_id
          word_id += 1

    self.vectors = self.train(training_set)

  def generate_tf_vector(self, text):
    tf_vector = [(self._vocabulary[word], 1 + log(document.count(word))) for word in self._vocabulary if \
                 document.count(word) is not 0]
    tf_vector.sort()

    return tf_vector

  def generate_tf_idf_vector(self, tf_vector):
    tf_idf_vector = [(x[0], x[1] * log(len(self.documents) / self.document_frequency[x[0]])) for x in tf_vector]
    tf_idf_vector = list(filter(lambda x: x[1] != 0.0, tf_idf_vector))

    return tf_idf_vector

  def normalize(self, vector):
    norm = sqrt(sum([x * x for x in vector]))
    return [(x / norm) for x in vector]

  def train(self, training_set):
    vectors = []
    for text, gender in training_set:
      vector = self.generate_tf_idf_vector(self.generate_tf_vector(text))
      vectors.append((self.normalize(vector), gender))

    return vectors

  def classify(self, text):
    ''' Returns 'M' if the text was written by a man
        and 'F' if it was written by a woman.

        text: [token]
    '''
    query_vector = self.normalize(self.generate_tf_vector(text))

