# Importing libraries
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.tokenize import sent_tokenize, TreebankWordTokenizer
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
import re
from bs4 import BeautifulSoup #to remove HTML tags
from collections import defaultdict

lemmatizer = WordNetLemmatizer()
stop_list = stopwords.words('english')


def parseDocs(filename): 
    with open(filename) as f:
        content = f.read()

    indexes = re.findall(r"\.I [0-9][0-9]*", content)
    indexes = [re.sub('\D', '', x).lstrip("0") for x in indexes]
    indexes = [int(x) for x in indexes]
    n_indexes = max(indexes)

    docs = ["" for _ in range(n_indexes)] 
    doc = ""
    cont = False
    i = 0

    for line in content.split('\n'): 
        if ".I" in line[:2]:
            if cont:
                docs[i] = doc
            idx_in_line = re.sub('\D', '', line).lstrip("0")
            i = int(idx_in_line) - 1
            cont = False
            doc = ""
        elif ".W" in line[:2]: #skip text between .I and .W
            cont = True
        elif cont:
            if re.match(r"\.[A-Z]\s*", line):
                cont = False
                docs[i] = doc
            else:
                if doc != "":
                    doc += " "
                doc += line
            
    if len(doc)>0: #needed for the last document
        docs[i] = doc
    
    return docs


def tokenize_and_clean(docs):
    """This function tokenizes texts into lowercased tokens with TreebankWordTokenizer
    
    Preprocesses the list of strings given as input.
    Tokenize each string into sentences using sent_tokenize(),
    tokenize each sentence into tokens using TreebankWordTokenizer().tokenize(),
    Lowercasing the characters, removing non-ASCII values, special characters, HTML tags and stopwords.
    

    Parameters
    ----------
    docs : list of strings
        list of document contents
    
    Returns
    -------
    tokens : list of list of strings
        each text as a list of lowercased tokens
    """
    tokens = []
    
    for doc in docs:
        # converting to lower case
        txt = doc.lower()
        
        # remove HTML tags
        txt = BeautifulSoup(txt, 'html.parser').get_text()
        
        # tokenize
        sentence = sent_tokenize(txt)
        tok = [TreebankWordTokenizer().tokenize(sent) for sent in sentence]
        tok = [item for sublist in tok for item in sublist] #convert to one list
        
        # removing stop words and special characters from the tokens
        clean_tokens = [word for word in tok if (word not in stop_list and not re.match('[^A-Za-z0-9]', word))]
        
        tokens.append(clean_tokens)


    return tokens


def lemmatize(doc_tokens):
    """This function lemmatizes texts with NLTK WordNetLemmatizer

    Parameters
    ----------
    doc_tokens : list of list of tokens
    
    Returns
    -------
    doc_lemmas : list of list of lemmatized tokens
    """
    doc_lemmas = []
    
    for doc in doc_tokens:
        lemmas = [lemmatizer.lemmatize(token) for token in doc]
        doc_lemmas.append(lemmas)
        
    return doc_lemmas


def read_cran_relevancy(path, relevancy_threshold=3):
    """ Loads query relevancy information from the cran dataset.

    Parameters
    ----------
    path : string
        path to the cranqrel file containing space-separated columns:
            query_id relevant_document_id relevant_document_number

    relevancy_threshold : int
        relevant_document_number column values are in 5 classes,
        1 - most important, 5 - no relevance
        this argument defines up to which relevancy the queries are paired
    
    Returns
    -------
    query_results : {qid: list of results sorted by relevancy, then by document id} 
    """

    # TODO: Make some sanity check.
    # Here, indexes should correspond, since
    #  td_docs, sorted_vocab = create_term_doc_matrix(docs)
    #  td_queries = create_term_doc_matrix_queries(queries, sorted_vocab)
    # so queries are indexed the same way as docs.
    # In addition, indexes from create_term_doc_matrix should correpond to
    # indexes from cran.all.1400, since documents are indexed one-by-one.

    query_results = defaultdict(list)

    with open(path) as f:
        for line in f.readlines():
            qid, docid, relevancy = tuple(line.split())
            qid, docid, relevancy = int(qid) - 1, int(docid) - 1, int(relevancy)
            if relevancy <= relevancy_threshold and relevancy != -1:
                query_results[qid].append((docid, relevancy))
    
    for k, v in query_results.items():
        query_results[k] = [docid for (docid, relevancy) in sorted(v, key=lambda x: x[1])]
    
    return dict(query_results)


def read_cisi_relevancy(path):
    """ Loads query relevancy information from the cisi dataset.

    Parameters
    ----------
    path : string
        path to the cranqrel file containing space-separated columns:
            query_id relevant_document_id relevant_document_number
    
    Returns
    -------
    query_results : {qid: list of results sorted by relevancy, then by document id} 
    """

    query_results = defaultdict(list)

    with open(path) as f:
        for line in f.readlines():
            qid, docid, _, _ = tuple(line.split())
            qid, docid = int(qid) - 1, int(docid) - 1
            query_results[qid].append(docid)
    
    return dict(query_results)


def reindex_data(docs, queries, results):
    """ Removes gaps and reindexes documents and queries.
    
        >>> reindex_data(["ab", "ac", "", "bb", "hh"],
        >>>              ["t", "", "", "", "n", "l"],
        >>>              {0: [1, 3], 4: [0,4], 5: [2,4]})
        (['ab', 'ac', 'bb', 'hh'],
         ['t', 'n', 'l'],
         {0: [1, 2], 1: [0, 3], 2: [1, 3]})
    """
    # Reindex documents
    i = 0
    n_docs = len(docs)
    while i < n_docs:  # Python is missing C hacks like for(;;i++);
        if docs[i] == "":
            docs = remove_at(i, docs)
            results = {k: [x-1 if x >= i else x for x in v] for k, v in results.items()}
            i -= 1
            n_docs = len(docs)
        i += 1
            
    # Reindex queries
    i = 0
    n_queries = len(queries)
    while i < n_queries:
        if queries[i] == "":
            queries = remove_at(i, queries)
            results = {(k-1 if k >= i else k): v for k, v in results.items()}
            i -= 1
            n_queries = len(queries)
        i += 1

    return docs, queries, results


def remove_at(i, arr):
    for j in range(i, len(arr) - 1):
        arr[j] = arr[j+1]
    arr = arr[:-1]
    return arr
