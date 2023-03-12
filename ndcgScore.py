import numpy as np


def cum_gain(relevance):
    if relevance is None or len(relevance) < 1:
        return 0.0
    return np.asarray(relevance).sum()


def dcg(relevance, alternate=True):
    if relevance is None or len(relevance) < 1:
        return 0.0

    rel = np.asarray(relevance)
    p = len(rel)
    if alternate:
        log2i = np.log2(np.asarray(range(1, p + 1)) + 1)
        return ((np.power(2, rel) - 1) / log2i).sum()
    else:
        log2i = np.log2(range(2, p + 1))
        return rel[0] + (rel[1:] / log2i).sum()


def idcg(relevance, alternate=True):
    if relevance is None or len(relevance) < 1:
        return 0.0
    rel = np.asarray(relevance).copy()
    rel.sort()
    return dcg(rel[::-1], alternate)


def ndcg(relevance, nranks, alternate=True):
    if relevance is None or len(relevance) < 1:
        return 0.0

    if (nranks < 1):
        raise Exception('nranks < 1')

    rel = np.asarray(relevance)
    pad = max(0, nranks - len(rel))

    rel = np.pad(rel, (0, pad), 'constant')

    rel = rel[0:min(nranks, len(rel))]

    ideal_dcg = idcg(rel, alternate)
    if ideal_dcg == 0:
        return 0.0

    return dcg(rel, alternate) / ideal_dcg


def ndcg_score(pre, test, k):
    sorce=0
    pre = np.argsort(-pre)
    count = 0
    for item in range(len(test)):
        if sum(test[item]) != 0:
            count = count + 1
            rel = test[item][pre[item]]
            test_ndcg = ndcg(rel, k)
            sorce = sorce + test_ndcg
    return sorce/count
