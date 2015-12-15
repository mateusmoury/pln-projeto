from math import log
from math import sqrt

class KNNClassifier:

  def __init__(self, training_set, k):
    self.k = k
    self._vocabulary = {}
    self.training_set = training_set
    word_id = 0

    for text, gender in training_set:
      for token in text:
        if token not in self._vocabulary:
          self._vocabulary[token] = word_id
          word_id += 1

    self.document_frequency = {}
    for word, word_id in self._vocabulary.items():
      self.document_frequency[word_id] = sum(1 for text in training_set if word in text[0])

    self.vectors = self.train(training_set)

  def generate_tf_vector(self, text):
    tf_vector = [(self._vocabulary[word], 1 + log(text.count(word))) for word in list(set(text)) if \
                  word in self._vocabulary]
    tf_vector.sort()

    return tf_vector

  def generate_tf_idf_vector(self, tf_vector):
    tf_idf_vector = [(x[0], x[1] * log(len(self.training_set) / self.document_frequency[x[0]])) for x in tf_vector]
    tf_idf_vector = list(filter(lambda x: x[1] != 0.0, tf_idf_vector))

    return tf_idf_vector

  def normalize(self, vector):
    norm = sqrt(sum([x[1] * x[1] for x in vector]))
    return [(x[0], (x[1] / norm)) for x in vector]

  def train(self, training_set):
    vectors = []
    for text, gender in training_set:
      vector = self.generate_tf_idf_vector(self.generate_tf_vector(text))
      vectors.append((self.normalize(vector), gender))

    return vectors

  def calculate_score(self, query, vector):
    return sum([query[k] * v for (k, v) in vector if k in query])

  def classify(self, text):
    ''' Returns 'M' if the text was written by a man
        and 'F' if it was written by a woman.

        text: [token]
    '''
    query_vector = self.normalize(self.generate_tf_vector(text))
    query = {a: b for (a, b) in query_vector}

    scores = [(self.calculate_score(query, vector[0]), vector[1]) for vector in self.vectors]
    scores.sort()
    scores.reverse()

    gender_count = {'M': 0, 'F': 0}
    for i in range(self.k):
      gender_count[scores[i][1]] += 1

    return max(gender_count.keys(), key=lambda x: gender_count[x])

