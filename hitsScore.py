import numpy as np


def hits_score(pre, test, k):
    pre = np.argsort(-pre)
    NumberOfHits = 0
    GT = test.sum()
    for item in range(len(test)):
        if sum(test[item]) != 0:
            rel = test[item][pre[item]]
            NumberOfHits = NumberOfHits + rel[:k].sum()
    return NumberOfHits / GT
