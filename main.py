from reader import read
from preprocessor import preprocess
from naivebayes import NaiveBayesClassifier
from knn import KNNClassifier
from metrics import calculate_metrics
import sys

if __name__ == '__main__':

  if len(sys.argv) != 2:
    print("Wrong number of arguments. Please specify a classifier. Choose from [naivebayes, knn]")

  else:
    if sys.argv[1] == 'naivebayes' or sys.argv[1] == 'knn':
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

      if sys.argv[1] == 'naivebayes':
        classifier = NaiveBayesClassifier(training_set)

      else:
        classifier = KNNClassifier(training_set, 7)

      print(calculate_metrics(test_set, classifier))

    else:
      print('Invalid classifier name. Choose from [naivebayes, knn]')
