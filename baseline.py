import numpy as np

from collections import Counter


def create_term_doc_matrix(base_docs):
    """ Constructs a frequency term-document matrix
    
    this function takes in a list of documents and returns a term-document matrix
    the rows are lemma types, the columns are documents 
    the rows should be sorted alphabetically
    the order of the columns should be preserved as it's given in base_docs
    the cell values are a number of times a lemma was seen in a document
    the value should be zero, if a lemma is absent from a document
    
    Parameters
    ----------
    base_docs : a list of lists of strings [['a','a','b'], ['a','b','c']]
        a list of documents represented as a list of lemmas
    
    Returns
    -------
    matrix : numpy array
        a matrix where columns are documents and rows are lemma types,
        the cells of the matrix contain lemma counts in a document,
        the lemmas for rows are sorted alphabetically
        for the example above it will be:
            np.array([[2,1],
                      [1,1],
                      [0,1]])
        
    sorted_vocab : list of strings
        a list of all the lemma types used in all documents (the rows of our matrix)
        the words should be strings sorted alphabetically
        for the example above it should be ['a','b','c']
    """
    sorted_vocab = list({word for doc in base_docs for word in set(doc)})    
    sorted_vocab.sort()

    freqs = []
    for doc in base_docs:
        counter = {k: 0 for k in sorted_vocab}
        for k, v in Counter(doc).items():
            counter[k] = v
        freqs.append(list(counter.values()))
    
    return np.array(freqs, dtype=np.float64).T, sorted_vocab


def create_term_doc_matrix_queries(normalized_queries, sorted_vocabulary):
    """ Constructs a frequency term-document matrix for queries
    
    this function takes in a list of queries and a vocabulary list and returns a term-document matrix
    the rows are lemma types as given in vocabulary, the columns are documents
    the rows should be in the same order as in vocabulary given
    the order of the columns should be preserved as it's given in normalized_queries
    the cell values are a number of times a lemma was seen in a document
    the value should be zero, if a lemma is absent from a document
    
    Parameters
    ----------
    normalized_queries : a list of lists of strings [['a','a','b','d'], ['a','b','c']]
        a list of queries represented as a list of lemmas
    sorted_vocabulary : list of strings
        a list of all the lemma types used in all training documents (the rows of our matrix)
        the words are strings sorted alphabetically
        for our example it will be ['a','b','c']
    
    Returns
    -------
    query_matrix : numpy array
        a matrix where columns are documents in normalized_queries 
        and rows are lemma types from sorted_vocabulary.
        for the example above it will be:
            np.array([[2,1],
                      [1,1],
                      [0,1]])
        'd' is not included in the matrix, because it is absent from sorted_vocabulary
    """
    filtered_queries = [list(filter(lambda x: x in sorted_vocabulary, doc)) for doc in normalized_queries]

    freqs = []
    for song in filtered_queries:
        counter = {k: 0 for k in sorted_vocabulary}
        for k, v in Counter(song).items():
            counter[k] = v
        freqs.append(list(counter.values()))
    
    return np.array(freqs, dtype=np.float64).T


def tf_idf(td_matrix):
    """ Weighs a term-document matrix of raw counts with tf-idf scheme
    
    this function takes in a term-document matrix as a numpy array, 
    and weights the scores with the tf-idf algorithm described above.
    idf values are modified with log_10
    
    Parameters
    ----------
    td_matrix : numpy array 
        a matrix where columns are songs and 
        rows are word counts in a song
    
    Returns
    -------
    tf_idf_matrix : numpy array 
        a matrix where columns are songs and 
        rows are word tf-idf values in a song
        
    idf_vector : numpy array of shape (vocabulary-size, 1)
        a vector of idf values for words in the collection. the shape is (vocabulary-size, 1)
        this vector will be used to weight new query documents
    """
    df_vector = np.sum(np.sign(td_matrix), axis=1)[..., np.newaxis]
    n = td_matrix.shape[1]
    idf_vector = np.log10(n / df_vector)
    
    return td_matrix * idf_vector, idf_vector


def lsi(matrix, d):
    """ Returns truncted SVD components
    
    this function takes in a term-document matrix, where
    values can be both raw frequencies and weighted values (tf_idf, ppmi)
    and returns their trunctaded SVD matrices.


    Parameters
    ----------
    matrix : numpy array
        a numpy array where columns are songs and 
        rows are lemmas
    d : int
        a number of features we will be reducing our matrix to
    
    Returns
    -------
    DT_d : numpy array
        a [d x m], where m is the number of word dimensions in the original matrix, 
        and d is the number of features we want to keep
        this is a matrix that represents documents with values for d hidden topics
    transformation_matrix : numpy array 
        a matrix to transform queries into the same vector space as DT_d
        T_dS_d^-, where S_d^- is inverse of S_d
    """

    T, s, DT = np.linalg.svd(matrix)
    return DT[:d, :], T[:, :d] @ np.linalg.inv(np.diag(s[:d]))