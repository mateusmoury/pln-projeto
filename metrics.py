def calculate_metrics(test_set, classifier):
    ''' Returns the metrics: precision, recall accuracy and f1 according to macro averaging.

        test_set: [([token], 'M' or 'F')]
    '''

    correct = {'M': 0, 'F': 0}
    wrong = {'M': 0, 'F': 0}

    for (text, gender) in test_set:
      if classifier.classify(text) == gender:
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
