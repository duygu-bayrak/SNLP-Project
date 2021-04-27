import numpy as np
import pandas as pd
import random

from sklearn.model_selection import train_test_split

from baseline import *
from preprocessing import *
from synonym_enrich import *
from cosine_sim import *


def validate(**params):
    # Set random seed to make experiments reproducible
    random.seed(0)
    
    # Load data
    docs = params.get("data_parse_fnt")(*params.get("doc_args"))
    queries = params.get("data_parse_fnt")(*params.get("query_args"))
    true_results = params.get("retrieval_fnt")(**params.get("retrieval_args"))

    # Split train, val queries:
    qids_train, qids_test = train_test_split(list(range(1, len(queries))), test_size=params.get("test_size"))

    # Preprocess + make model for documents
    docs = preprocess(docs, params.get("preprocessings"))  # list of list of lemmatized tokens
    queries = preprocess(queries, params.get("preprocessings"))

    # Validate
    # Baseline case
    if params.get("embedding") is None:
        docs_data, sorted_vocab = create_term_doc_matrix(docs)
        query_params = {"sorted_vocab": sorted_vocab,}

        if params.get("use_tfidf"):
            docs_data, idf_vector = tf_idf(docs_data)
            query_params["idf_vector"] = idf_vector

        if params.get("use_lsi"):
            docs_data, dimred_transform = lsi(docs_data, params.get("d"))
            query_params["dimred_transform"] = dimred_transform
        
    # Embedding case
    else:
        docs_data = embed(None, docs, embedding=params.get("embedding"))
        query_params = {"embedding": params.get("embedding"),}

    # Calculate scores
    results = {}
    rs = []
    k = params.get("k")
    for qid in qids_test:
        q = embed([qid], queries, **query_params)[:,0]
        q_pred = [x[1] for x in get_k_relevant(k, q, docs_data)]
        q_true = true_results[qid]
        rs.append([int(x in q_true) for x in q_pred])

    results["mAP"] = 100.0 * mean_average_precision(rs)
    return results


# Retrieval Metrics,
# patched version from https://github.com/lgalke/vec4ir/blob/master/vec4ir/rank_metrics.py

def precision(r):
    r = np.asarray(r)
    if r.size == 0:
        return 0
    tp = np.count_nonzero(r)
    return tp / r.size

def recall(r, n_relevant):
    return np.count_nonzero(r) / n_relevant

def precision_at_k(r, k):
    """Score is precision @ k

    Relevance is binary (nonzero is relevant).

    >>> r = [0, 0, 1]
    >>> precision_at_k(r, 1)
    0.0
    >>> precision_at_k(r, 2)
    0.0
    >>> precision_at_k(r, 3)
    0.33333333333333331
    >>> precision_at_k(r, 4)
    Traceback (most recent call last):
        File "<stdin>", line 1, in ?
    ValueError: Relevance score length < k


    Args:
        r: Relevance scores (list or numpy) in rank order
            (first element is the first item)

    Returns:
        Precision @ k

    Raises:
        ValueError: len(r) must be >= k
    """
    r = np.asarray(r)[:k] != 0
    if r.size != k:
        raise ValueError('Relevance score length < k')
    return np.mean(r)

def average_precision(r):
    """Score is average precision (area under PR curve)

    Relevance is binary (nonzero is relevant).

    >>> r = [1, 1, 0, 1, 0, 1, 0, 0, 0, 1]
    >>> delta_r = 1. / sum(r)
    >>> sum([sum(r[:x + 1]) / (x + 1.) * delta_r for x, y in enumerate(r) if y])
    0.7833333333333333
    >>> average_precision(r)
    0.78333333333333333
    >>> r = [1, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0]
    >>> average_precision(r)
    0.78333333333333333
    >>> average_precision([1,1,0,0]) == average_precision([1,1])
    True
    >>> average_precision([0])
    0.0


    Args:
        r: Relevance scores (list or numpy) in rank order
            (first element is the first item)

    Returns:
        Average precision
    """
    r = np.asarray(r) != 0
    out = [precision_at_k(r, k + 1) for k in range(r.size) if r[k]]
    if not out:
        return 0.0
    return np.mean(out)

def mean_average_precision(rs):
    """Score is mean average precision

    Relevance is binary (nonzero is relevant).

    >>> rs = [[1, 1, 0, 1, 0, 1, 0, 0, 0, 1]]
    >>> mean_average_precision(rs)
    0.78333333333333333
    >>> rs = [[1, 1, 0, 1, 0, 1, 0, 0, 0, 1], [0]]
    >>> mean_average_precision(rs)
    0.39166666666666666

    Args:
        rs: Iterator of relevance scores (list or numpy) in rank order
            (first element is the first item)

    Returns:
        Mean average precision
    """
    return np.mean([average_precision(r) for r in rs])


def preprocess(docs, preprocessings):
    for fnt in preprocessings:
        docs = fnt(docs)
    return docs

def embed(qids, queries, **params):
    if qids is not None:
        queries = [queries[i] for i in qids]
    
    # Baseline case
    if "sorted_vocab" in params:
        # TODO: add baseline further processing options
        queries = create_term_doc_matrix_queries(queries, params.get("sorted_vocab"))
        if "idf_vector" in params:
            queries = queries * params.get("idf_vector")
        if "dimred_transform" in params:
            queries = queries.T.dot(params.get("dimred_transform")).T
        return queries

    # Embedding case
    embedding = params.get("embedding")
    # TODO: Decide, what to do with out-of-vocabary with word2vec
    # if isinstance(embedding, Word2Vec):
    #     docs_data = [np.mean(np.array([embedding.wv[w] for w in doc]), axis=0) for doc in queries]
    # else:
    docs_data = [embedding.infer_vector(doc) for doc in queries]
    return np.array(docs_data).T