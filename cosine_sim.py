import numpy as np


def cosine_sim(x, y):
    """calculates cosine similarity between 2 vectors.
        
    Parameters
    ----------
    x : numpy.ndarray
        vector representation (of query)
    y : numpy.ndarray
        vector representation (of document)
    
    Returns
    -------
    cosine_sim: numpy.float64
        cosine similarity between vector x and y
    """
    
    cos_sim = np.dot(x, y)/(np.linalg.norm(x)*np.linalg.norm(y))
    
    return cos_sim


def get_k_relevant(k, query, D):
    """returns ranked list of top k documents in descending order of their
    cosine similarity with the query
        
    Parameters
    ----------
    k : int
        number of documents to retrieve (top k)
    query : numpy.ndarray
        vector representation of query whose cosine similarity is to be computed with the corpus
    D: list of numpy.ndarray
        vector representation of all documents in corpus
    
    Returns
    -------
    ranked_sims: list of tuples (cosine similarity, index of document)
        list of top k cosine similarities and the corresponding documents (their index) in descending order
    """
      
    cosine_sims = []
    
    for i, d in enumerate(D):
        cosine_sims.append((cosine_sim(query, d), i))
        
    ranked_sims = sorted(cosine_sims, key=lambda x: x[0], reverse=True)
    
    if k != 0:
        # if k=0 retrieve all documents in descending order
        ranked_sims = ranked_sims[:k]
    
    return ranked_sims



def get_over_thresh(thresh, query, D):
    """returns ranked list of top k documents in descending order of their
    cosine similarity with the query
        
    Parameters
    ----------
    thresh : numpy.float64
        minimum similarity that returned documents should have
    query : numpy.ndarray
        vector representation of query whose cosine similarity is to be computed with the corpus
    D: list of numpy.ndarray
        vector representation of all documents in corpus
    
    Returns
    -------
    ranked_sims: list of tuples (cosine similarity, index of document)
        list of cosine similarities (greater than thresh) and the corresponding documents (their index) in descending order
    """
      
    cosine_sims = []
    
    for i, d in enumerate(D):
        cosine_sims.append((cosine_sim(query, d), i))
        
    ranked_sims = sorted(cosine_sims, key=lambda x: x[0], reverse=True)
    
    if thresh != 1:
        # if thresh=1 retrieve all documents in descending order
        ranked_sims = [elem for elem in ranked_sims if elem[0]>=thresh ]
    
    return ranked_sims



