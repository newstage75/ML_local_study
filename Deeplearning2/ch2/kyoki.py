import sys
sys.path.append('..')
import numpy as np
from common.util import preprocess

text = 'You say googlebye and I say hello.'
corpus, word_to_id, id_to_word = preprocess(text)

print(corpus)
# [0 1 2 3 4 1 5 6]

print(id_to_word)
# {0: 'you', 1: 'say', 2: 'googlebye', 3: 'and', 4: 'i', 5: 'hello', 6: '.'}
