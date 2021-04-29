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
    if "embedding" not in params or params.get("embedding") is None:
        docs_data, query_params = embed(None, docs, **params)
        
    # Embedding case
    else:
        docs_data, query_params = embed(None, docs, **params)
        query_params["embedding"] = params.get("embedding")

    # Calculate scores
    results = {}
    rs = []
    k = params.get("k")
    for qid in qids_test:
        q, _ = embed([qid], queries, **query_params)
        q = q[:,0]
        q_retrived = [x[1] for x in get_k_relevant(k, q, docs_data)]
        q_relevant = true_results[qid]
        rs.append([int(x in q_relevant) for x in q_retrived])

    p_score = np.mean([precision(r) for r in rs])
    r_score = np.mean([recall(r, len(q_relevant)) for r in rs])
    results["precision (mean)"] = 100.0 * p_score
    results["recall (mean)"] = 100.0 * r_score
    results["F1 (mean)"] = 100.0 * f1_score(p_score, r_score)
    results["mAP"] = 100.0 * mean_average_precision(rs)
    results["MRR"] = 100.0 * mean_reciprocal_rank(rs)
    
    return results


def preprocess(docs, preprocessings):
    for fnt in preprocessings:
        docs = fnt(docs)
    return docs


def embed(qids, X, **params):
    """ Overloaded function!
        Performs embedding for both reference documents and queries.
        ------
        Input:
            qids: list of ints
                Query ids correspoding to columns in X to process
            X:  list of strings or 
                Corresponds to list of lemmas.
        ------
        Output:
            Y: numpy array of shape (m,n)
                m is the vector embedding size
                n is number of documents to be retrieved
            additional_data: dict
                Dictionary with additional data from reference documents first case.
    """

    # (A) Documents first case
    if qids is None:
        additional_data = {}
        
        # Embedding case
        if "embedding" in params and params.get("embedding") is not None:
            X = w2v_embed(X, params.get("embedding"))
        # Baseline case
        else:
            X, sorted_vocab = create_term_doc_matrix(X)
            additional_data = {"sorted_vocab": sorted_vocab,}

        # Apply optional math processing
        if "use_tfidf" in params and params.get("use_tfidf"):
            X, idf_vector = tf_idf(X)
            additional_data["idf_vector"] = idf_vector
        if "use_lsi" in params and params.get("use_lsi"):
            X, dimred_transform = lsi(X, params.get("d"))
            additional_data["dimred_transform"] = dimred_transform
    
        return X, additional_data

    # (B) Queries other cases
    X = [X[i] for i in qids]

    # Baseline case (only queries)
    if "sorted_vocab" in params:
        X = create_term_doc_matrix_queries(X, params.get("sorted_vocab"))
    # Embedding case (any documents)    
    else:
        X = w2v_embed(X, params.get("embedding"))

    # Apply optional math processing
    if "idf_vector" in params:
        X = X * params.get("idf_vector")
    if "dimred_transform" in params:
        X = X.T.dot(params.get("dimred_transform")).T

    return X, {}


def w2v_embed(X, embedding, tf_matrix, sorted_vocab):
    if "infer_vector" in dir(embedding):
        docs_data = [embedding.infer_vector(doc) for doc in X]
    else: #Vanilla Position Vector
        docs_data = []
        for i, doc in enumerate(X):
            n_words = len(doc)
            weighted_sum = np.zeros((vec_size,))
            for w in doc:
                if w in embedding.wv:  # skip, if OOV
                    word_idx = sorted_vocab.index(w)
                    weight = tf_matrix[word_idx][i]
                    weighted_sum += weight * embedding.wv[w]
                   
            avg = weighted_sum/n_words
            docs_data.append(avg)
    return np.array(docs_data).T


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

def f1_score(precision, recall):
    if precision == 0 and recall == 0:
        return 0
    return 2 * precision * recall / (precision + recall)

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

def mean_reciprocal_rank(rs):
    """Score is reciprocal of the rank of the first relevant item

    First element is 'rank 1'.  Relevance is binary (nonzero is relevant).

    Example from http://en.wikipedia.org/wiki/Mean_reciprocal_rank
    >>> rs = [[0, 0, 1], [0, 1, 0], [1, 0, 0]]
    >>> mean_reciprocal_rank(rs)
    0.61111111111111105
    >>> rs = np.array([[0, 0, 0], [0, 1, 0], [1, 0, 0]])
    >>> mean_reciprocal_rank(rs)
    0.5
    >>> rs = [[0, 0, 0, 1], [1, 0, 0], [1, 0, 0]]
    >>> mean_reciprocal_rank(rs)
    0.75

    Args:
        rs: Iterator of relevance scores (list or numpy) in rank order
            (first element is the first item)

    Returns:
        Mean reciprocal rank
    """
    rs = (np.asarray(r).nonzero()[0] for r in rs)
    return np.mean([1. / (r[0] + 1) if r.size else 0. for r in rs])
