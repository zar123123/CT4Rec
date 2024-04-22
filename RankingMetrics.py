#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import math

def precision_recall_ndcg_at_k(k, rankedlist, test_matrix):
    idcg_k = 0
    dcg_k = 0
    n_k = k if len(test_matrix) > k else len(test_matrix)#nk=1
    for i in range(n_k):
        idcg_k += 1 / math.log(i + 2, 2)
    
    b1 = rankedlist
    b2 = test_matrix
    s2 = set(b2)
    hits = [(idx, val) for idx, val in enumerate(b1) if val in s2]
    
    count = len(hits)
    
    for c in range(count):
        dcg_k += 1 / math.log(hits[c][0] + 2, 2) #1/log2(3)

    return float(count / k), float(count / len(test_matrix)), float(dcg_k / idcg_k)







