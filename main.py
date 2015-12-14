from reader import read
from preprocessor import preprocess
from classifier import NaiveBayesClassifier

if __name__ == '__main__':
  annotated_texts = read('blog-gender-dataset.xlsx')

  training_set_len = 0.7 * len(annotated_texts)

  training_set = []
  test_set = []

  for (text,gender) in annotated_texts:
    if 'M' in gender:
      gender = 'M'
    else:
      gender = 'F'
    if len(training_set) < training_set_len:
      training_set.append((preprocess(text), gender))
    else:
      test_set.append((preprocess(text), gender))

  classifier = NaiveBayesClassifier(training_set)

  print(classifier.metrics(test_set))