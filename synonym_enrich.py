import copy
from collections import Counter
import pandas as pd

from nltk.corpus import wordnet


def synonym_enrich(model):
    [vocabulary, mapping] = synonym_mapping(model)
    return synonym_enrichment_v1(model, mapping)


def synonym_mapping(doc_lemmas):
    """This function computes the synonym mappings of the terms in the vocabulary.

    Parameters
    ----------
    doc_lemmas : list of list of strings (lemmas)
    
    Returns
    -------
    mapping : dictionary mapping a term to a list of synonyms in the vocabulary
    """
    def getSynonyms(term): #uses wordNet to get the synonym of every term
        syns =[]
        for synset in wordnet.synsets(term):
            for lemma in synset.lemmas():
                syns.append(lemma.name())
        return list(set(syns +[term]))

    #get the vocabulary
    vocabulary =  []
    for doc in doc_lemmas:
        vocabulary += doc
    vocabulary = list(set(vocabulary))

    # get the mapping between each term of the vocabulary and a list of synonyms from wordNet
    mapping ={vocabulary[i]: getSynonyms(vocabulary[i]) for i in range(len(vocabulary)) }
     
    #filter synonyms to only keep the ones in the vocabulary 
    mapping ={k: [ term for term in mapping[k] if term in vocabulary] for k in mapping.keys() }
    
    return [vocabulary, mapping]

def synonym_enrichment_v1(doc_lemmas,mapping):
    """This function enriches the documents using a semantic knowledge base such as wordNet
    
    It adds the synonyms of each term to the document to enrich it

    Parameters
    ----------
    doc_lemmas : list of list of strings (lemmas)
    
    Returns
    -------
    doc_enrich : list of list of strings
    """
    doc_enrich = []
    
    for doc in doc_lemmas:
        enriched_doc =copy.deepcopy(doc)
        for lemma in doc:
            enriched_doc += mapping[lemma]
        doc_enrich.append(enriched_doc)
            
              
    return doc_enrich

def synonym_enrichment_v2(doc_lemmas,mapping):
    """This function enriches the documents using a semantic knowledge base such as wordNet
    
    It adds the synonyms of each term to the document to enrich it

    Parameters
    ----------
    doc_lemmas : list of list of strings (lemmas)
    
    Returns
    -------
    doc_enrich : list of list of strings
    """
    
#     dog = wn.synset('dog.n.01')
#     cat = wn.synset('cat.n.01')

#     print(dog.path_similarity(cat))
#     print(dog.lch_similarity(cat))
#     print(dog.wup_similarity(cat))

    doc_enrich = []
    for doc in doc_lemmas:
        enriched_doc = Counter(doc)
        
        for lemma in doc:
            #lemma has a freq of enriched_doc[lemma]
            
            for v in mapping[lemma]:
                if v not in enriched_doc.keys():
                    enriched_doc[v] =0
                enriched_doc[v] +=  enriched_doc[lemma]*1/len(mapping[lemma])  #similarities(lemma,v) is too expensive... => attenuate
        doc_enrich.append(enriched_doc)
            
            
    return doc_enrich


def synonym_enrichment_v3(doc_lemmas,mapping):
    NewTokens = {k: [] for k in mapping.keys()}
    for k in mapping.keys():
        NewTokens[k].append("_".join(mapping[k]))

        for token in mapping[k]:
            NewTokens[token].append("_".join(mapping[k]))

    NewTokens= {k: list(set(NewTokens[k])) for k in mapping.keys()}
    
    doc_enrich=[]
    for doc in doc_lemmas:
        doc_en = []
        for term in doc:
            doc_en +=NewTokens[term] 
        doc_enrich.append(doc_en)
    return doc_enrich
