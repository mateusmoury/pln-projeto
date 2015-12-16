class NaiveBayesClassifier:

  def __init__(self, training_set, interpolation=False):
    '''
        training_set: [([token], 'M' or 'F')]
    '''

    self._interpolation = interpolation
    
    male_examples = 0
    female_examples = 0

    for (x,t) in training_set:
      if t == 'M':
        male_examples += 1
      else:
        female_examples += 1

    self._word_freq = {'M' : {}, 'F' : {}}
    self._bigram_freq = {'M' : {}, 'F' : {}}
    self._trigram_freq = {'M' : {}, 'F' : {}}
    
    self._vocabulary = set()
    self._bigram_vocabulary = set()
    self._trigram_vocabulary = set()

    self._prob_gender = {'M': male_examples / (male_examples + female_examples),
                        'F': female_examples / (male_examples + female_examples)}

    self.train(training_set)
  

  def train(self, training_set):
    ''' Trains this classifier.

        training_set: [([token], 'M' or 'F')]
    '''

    for (text, gender) in training_set:
      for i in range(len(text)):
        self._vocabulary.add(text[i])
        
        if text[i] in self._word_freq[gender]:
          self._word_freq[gender][text[i]] += 1
        else:
          self._word_freq[gender][text[i]] = 1

        if self._interpolation:
          if i > 0:
            bigram = (text[i - 1], text[i])
            self._bigram_vocabulary.add(bigram)
            if bigram in self._bigram_freq[gender]:
              self._bigram_freq[gender][bigram] += 1
            else:
              self._bigram_freq[gender][bigram] = 1

          if i > 1:
            trigram = (text[i - 2], text[i - 1], text[i])
            self._trigram_vocabulary.add(trigram)
            if trigram in self._trigram_freq[gender]:
              self._trigram_freq[gender][trigram] += 1
            else:
              self._trigram_freq[gender][trigram] = 1


  def classify(self, text):
    ''' Returns 'M' if the text was written by a man
        and 'F' if it was written by a woman.

        text: [token]
    '''

    smoothing_alpha = 0.77
    
    from math import log

    if self._interpolation:
      prob = {'M' : {'word' : log(self._prob_gender['M']), 'bigram' : log(self._prob_gender['M']), 'trigram' : log(self._prob_gender['M'])},
              'F' : {'word' : log(self._prob_gender['F']), 'bigram' : log(self._prob_gender['F']), 'trigram' : log(self._prob_gender['F'])}}
    else:
      prob = {'M' : log(self._prob_gender['M']),
              'F' : log(self._prob_gender['F'])}


    for gender in prob:
      denominator = log(sum(self._word_freq[gender].values()) + len(self._vocabulary) * smoothing_alpha)
      for token in text:
        try:
          f = self._word_freq[gender][token]
        except:
          f = 0
        if self._interpolation:
          prob[gender]['word'] += log(f + smoothing_alpha) - denominator
        else:
          prob[gender] += log(f + smoothing_alpha) - denominator

    if self._interpolation:
      for gender in prob:
        denominator = log(sum(self._bigram_freq[gender].values()) + len(self._bigram_vocabulary) * smoothing_alpha)
        for i in range(1, len(text)):
          bigram = (text[i - 1], text[i])
          f = 0
          if bigram in self._bigram_freq[gender]:
            f = self._bigram_freq[gender][bigram]

          prob[gender]['bigram'] += log(f + smoothing_alpha) - denominator

      for gender in prob:
        denominator = log(sum(self._trigram_freq[gender].values()) + len(self._trigram_vocabulary) * smoothing_alpha)
        for i in range(2, len(text)):
          trigram = (text[i - 2], text[i - 1], text[i])
          f = 0
          if trigram in self._trigram_freq[gender]:
            f = self._trigram_freq[gender][trigram]

          prob[gender]['trigram'] += log(f + smoothing_alpha) - denominator

    if self._interpolation:
      a = 0.65
      b = 0.22
      c = 1.0 - a - b

      prob_male = prob['M']['word'] * a + prob['M']['bigram'] * b + prob['M']['trigram'] * c
      prob_female = prob['F']['word'] * a + prob['F']['bigram'] * b + prob['F']['trigram'] * c
    else:
      prob_male = prob['M']
      prob_female = prob['F']

    return 'M' if prob_male > prob_female else 'F'

    
