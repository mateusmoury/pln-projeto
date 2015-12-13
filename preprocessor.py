from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from spellcheck import correct
import string

def preprocess(text):
  stemmer = PorterStemmer()
  stop = stopwords.words("english")
  result = word_tokenize(text)
  result = [stemmer.stem_word(word.lower()) for word in result if \
            word not in stop and \
            word not in string.punctuation and \
            word not in string.digits]
  return result

if __name__ == '__main__':
  print(preprocess("Hi, my name is Mateus and I hope that you delete some of the text I just wrote and here is a wrng wrd"))

