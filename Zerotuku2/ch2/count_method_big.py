import sys
sys.path.append('..')
import numpy as np
from common.util import most_similar, create_co_matrix, ppmi
from dataset import ptb

window_size = 2
wordvec_size = 100

corpus, word_to_id, id_to_word = ptb.load_data('train')
vocab_size = len(word_to_id)
print('counting co-occurrence ...')
C = create_co_matrix(corpus, vocab_size, window_size)
print('calculating PPMI ...')
W = ppmi(C, verbose=True)

print('calculationg SVD...')
try:
    # truncated SVD(fast!)
    from sklearn.units.extmath import randomized_svd
    U, S, V = randomized_svd(W, n_components = wordvec_size, n_iter=5, random_state = None)

except ImportError:
    #SVD(slow)
    U, S, V = np.linalg.svd(W)

word_vecs = U[:, :wordvec_size]

querys = ['you', 'year', 'car', 'toyota']
for query in querys:
    most_similar(query, word_to_id, id_to_word, word_vecs, top=5)


# calculationg SVD...（高速のSVDでは、乱数を用いているので、毎回結果は変わる）
#
# [query]you
#  i: 0.7003179788589478
#  we: 0.6367184519767761
#  anybody: 0.5657642483711243
#  do: 0.563567042350769
#  'll: 0.5127798318862915
#
# [query]year
#  month: 0.6961644887924194
#  quarter: 0.6884942054748535
#  earlier: 0.6663320064544678
#  last: 0.628136396408081
#  next: 0.6175755858421326
#
# [query]car
#  luxury: 0.672883152961731
#  auto: 0.6452109813690186
#  vehicle: 0.6097723245620728
#  cars: 0.6032834053039551
#  corsica: 0.5698372721672058
#
# [query]toyota
#  motor: 0.7585657835006714
#  nissan: 0.7148030996322632
#  motors: 0.692615807056427
#  lexus: 0.6583304405212402
#  honda: 0.6350275278091431