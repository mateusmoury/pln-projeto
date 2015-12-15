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
    self._prob_gender = {'M': male_examples / (male_examples + female_examples),
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

    prob = {'M' : log(self._prob_gender['M']),
            'F' : log(self._prob_gender['F'])}

    for gender in prob:
      denominator = log(sum(self._word_freq[gender].values()) + len(self._vocabulary) * smoothing_alpha)
      for token in text:
        try:
          f = self._word_freq[gender][token]
        except:
          f = 0
        prob[gender] += log(f + smoothing_alpha) - denominator

    return 'M' if prob['M'] > prob['F'] else 'F'

    