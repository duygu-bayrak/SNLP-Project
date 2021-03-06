import numpy as np
import pandas as pd
import random

from sklearn.model_selection import train_test_split

from baseline import *
from preprocessing import *
from synonym_enrich import *
from cosine_sim import *

from nltk.corpus import wordnet

def validate(**params):
    # Set random seed to make experiments reproducible
    random.seed(0)
    
    # Load data
    docs = params.get("data_parse_fnt")(*params.get("doc_args"))
    queries = params.get("data_parse_fnt")(*params.get("query_args"))
    true_results = params.get("retrieval_fnt")(**params.get("retrieval_args"))
    embedding = params.pop("embedding", None)

    # Remove gaps in indexing
    docs, queries, true_results = reindex_data(docs, queries, true_results)

    # Split train, val queries:
    qids_train, qids_test = train_test_split(list(true_results.keys()), test_size=params.get("test_size"))

    # Preprocess + make model for documents
    docs = preprocess(docs, params.get("preprocessings"))  # list of list of lemmatized tokens
    queries = preprocess(queries, params.get("preprocessings"))

    # Validate
    # Baseline case
    if embedding is None:
        docs_data, query_params = embed(None, docs, embedding, **params)
        
    # Embedding case
    else:
        docs_data, query_params = embed(None, docs, embedding, **params)

    # Calculate scores
    results = {}
    rs_ranked = []
    rs_unranked = []
    n_relevant = []
    similarity_threshold = params.get("similarity_threshold")
    k = params.get("k")
    for qid in qids_test:
        q, _ = embed([qid], queries, embedding, **query_params)
        q = q[:,0]
        q_retrived = get_k_relevant(0, q, docs_data)
        q_retrived = list(filter(lambda x: x[0] is not None and x[0] > similarity_threshold, q_retrived))
        q_retrived = [x[1] for x in q_retrived]
        q_relevant = true_results[qid]
        rs_unranked.append([int(x in q_relevant) for x in q_retrived])
        rs_ranked.append([int(x in q_relevant) for x in q_retrived[:k]])
        n_relevant.append(len(true_results[qid]))

    precision_unranked = np.mean([precision(r) for r in rs_unranked])
    recall_unranked = np.mean([recall(r, n) for r, n in zip(rs_unranked, n_relevant)])
    results["precision"] = 100.0 * precision_unranked
    results["recall"] = 100.0 * recall_unranked
    results["F1"] = 100.0 * f1_score(precision_unranked, recall_unranked)

    precision_ranked = np.mean([precision(r) for r in rs_ranked])
    recall_ranked = np.mean([recall(r, min(k, n)) for r, n in zip(rs_ranked, n_relevant)])
    results["precision@{}".format(k)] = 100.0 * precision_ranked
    results["recall@{}".format(k)] = 100.0 * recall_ranked
    results["mAP"] = 100.0 * mean_average_precision(rs_ranked)
    results["MRR"] = 100.0 * mean_reciprocal_rank(rs_ranked)
    
    return results


def preprocess(docs, preprocessings):
    for fnt in preprocessings:
        docs = fnt(docs)
    return docs


def embed(qids, X, embedding, **params):
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
                Dictionary with additional data from reference documents.
    """

    # (A) Documents first case
    if qids is None:
        additional_data = {}
        
        # Embedding case
        if embedding is not None:
            if params.get("use_position_vector"):
                tf_matrix, sorted_vocab = create_term_doc_matrix(X)
                additional_data["tf_matrix"] = tf_matrix
                additional_data["sorted_vocab"] = sorted_vocab
            X = w2v_embed(X, embedding, **additional_data)
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

    # Embedding case   
    if embedding is not None:
        X = w2v_embed(X, embedding, **params)
    # Baseline case
    else:
        X = create_term_doc_matrix_queries(X, params.get("sorted_vocab"))

    # Apply optional math processing
    if "idf_vector" in params:
        X = X * params.get("idf_vector")
    if "dimred_transform" in params:
        X = X.T.dot(params.get("dimred_transform")).T

    return X, {}


def w2v_embed(X, embedding, **params):
    # Doc2Vec:
    if "infer_vector" in dir(embedding):
        docs_data = [embedding.infer_vector(doc) for doc in X]
    # Vanilla Position Vector:
    elif "tf_matrix" in params and "sorted_vocab" in params:
        tf_matrix = params.get("tf_matrix")
        sorted_vocab = params.get("sorted_vocab")
        docs_data = []
        
        def getSynonyms(term): #uses wordNet to get the synonym of every term
            syns =[]
            for synset in wordnet.synsets(term):
                for lemma in synset.lemmas():
                    syns.append(lemma.name())
            return list(set(syns +[term]))
        
        for i, doc in enumerate(X):
            n_words = len(doc)
            weighted_sum = np.zeros((50,))
            for w in doc:
                if w in embedding.wv:  # skip, if OOV
                    if w in sorted_vocab:
                        word_idx = sorted_vocab.index(w)
                        weight = tf_matrix[word_idx][i]
                        weighted_sum += weight * embedding.wv[w]
                    else:
                        #replace word by its synonym and get the average
                        synonyms = getSynonyms(w)
                        weighted_sum_syn = np.zeros((50,))
                        cnt=0
                        for syn in synonyms:
                            if syn in embedding.wv:  # skip, if OOV
                                if syn in sorted_vocab:
                                    cnt+=1
                                    word_idx = sorted_vocab.index(syn)
                                    weight = tf_matrix[word_idx][i]
                                    weighted_sum_syn += weight * embedding.wv[syn]
                        if cnt!=0: 
                            avg_syn = weighted_sum_syn/cnt
                            weighted_sum += avg_syn
            avg = weighted_sum/n_words
            docs_data.append(avg)
    # Pure word2vec:        
    else:
        docs_data = []
        for doc in X:
            vecs = []
            for w in doc:
                if w in embedding.wv:  # skip, if OOV 
                    vecs.append(embedding.wv[w])
            docs_data.append(np.mean(np.array(vecs), axis=0))
    return np.vstack(docs_data).T


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