class NaiveBayesClassifier:

  def __init__(self, training_set):
    '''
        training_set: [([token], 'M' or 'F')]
    '''
    
    male_examples = 0
    female_examples = 0

    for (x,t) in training_set:
      if t == 'M':
        male_examples += 1
      else:
        female_examples += 1

    self._word_freq = {'M' : {}, 'F' : {}}
    self._vocabulary = set()
    # self._bigram_freq = {'M' : {}, 'F' : {}}
    # self._trigram_freq = {'M' : {}, 'F' : {}}
    self._prob_class = {'M': male_examples / (male_examples + female_examples),
                        'F': female_examples / (male_examples + female_examples)}

    self.train(training_set)
  

  def train(self, training_set):
    ''' Trains this classifier.

        training_set: [([token], 'M' or 'F')]
    '''

    for (text, gender) in training_set:
      for token in text:
        self._vocabulary.add(token)
        if token in self._word_freq[gender]:
          self._word_freq[gender][token] += 1
        else:
          self._word_freq[gender][token] = 1


  def classify(self, text):
    ''' Returns 'M' if the text was written by a man
        and 'F' if it was written by a woman.

        text: [token]
    '''

    smoothing_alpha = 0.77
    
    from math import log

    prob = {'M' : log(self._prob_class['M']),
            'F' : log(self._prob_class['F'])}

    for gender in prob:
      denominator = log(sum(self._word_freq[gender].values()) + len(self._vocabulary) * smoothing_alpha)
      for token in text:
        try:
          f = self._word_freq[gender][token]
        except:
          f = 0
        prob[gender] += log(f + smoothing_alpha) - denominator

    return 'M' if prob['M'] > prob['F'] else 'F'


  def metrics(self, test_set):
    ''' Returns the metrics: precision, recall accuracy and f1 according to macro averaging.

        test_set: [([token], 'M' or 'F')]
    '''

    correct = {'M': 0, 'F': 0}
    wrong = {'M': 0, 'F': 0}

    for (text, gender) in test_set:
      if self.classify(text) == gender:
        correct[gender] += 1
      else:
        wrong[gender] += 1

    precision_male = correct['M'] / (correct['M'] + wrong['F'])
    precision_female = correct['F'] / (correct['F'] + wrong['M'])

    recall_male = correct['M'] / (correct['M'] + wrong['M'])
    recall_female = correct['F'] / (correct['F'] + wrong['F'])

    accuracy = (correct['M'] + correct['F']) / (correct['M'] + correct['F'] + wrong['M'] + wrong['F'])
    precision = (precision_male + precision_female) / 2.0
    recall = (recall_male + recall_female) / 2.0
    f1 = 2.0 * precision * recall / (precision + recall)

    return {'accuracy' : accuracy, 'f1' : f1, 'precision' : precision, 'recall' : recall}
    