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

lemmatizer = WordNetLemmatizer()
stop_list = stopwords.words('english')


def parseDocs(filename): 
    with open(filename,"r") as f:
        docs = []
        doc = ""
        cont = False
    
        for line in f: #skip text between .I and .W
            if ".I" in line:
                cont = False
                if len(doc)>0:
                    docs.append(doc)
                    doc = ""
            elif ".W" in line:
                cont = True
            elif cont == True:
                doc = doc + line

        if len(doc)>0: #needed for the last document
            docs.append(doc)

        f.close()
    
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


