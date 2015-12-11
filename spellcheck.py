import re, collections

def words(text):
  return re.findall('[a-z]+', text.lower())

def train(features):
  model = collections.defaultdict(lambda: 1)
  for f in features:
    model[f] += 1
  return model

ALLWORDS = train(words(open('spellchecktrainning.txt').read()))

alphabet = 'abcdefghijklmnopqrstuvwxyz'

def edits1(word):
  splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
  deletes = [a + b[1:] for a, b in splits if b]
  transposes = [a + b[1] + b[0] + b[2:] for a, b in splits if len(b) > 1]
  replaces = [a + c + b[1:] for a, b in splits for c in alphabet if b]
  inserts = [a + c + b for a, b in splits for c in alphabet]
  return set(deletes + transposes + replaces + inserts)

# para cobrir casos com distancia de edicao maior que 1
def known_edits2(word):
  return set(e2 for e1 in edits1(word) for e2 in edits1(e1) if e2 in ALLWORDS)

def known(words):
  return set(w for w in words if w in ALLWORDS)

def correct(word):
  candidates = known([word]) or known(edits1(word)) or known_edits2(word) or [word]
  return max(candidates, key=ALLWORDS.get)

if __name__ == '__main__':
  print(correct('Hi'))
  print(correct('wrng'))
  print(correct('wrod'))
  print(correct('deccisions'))