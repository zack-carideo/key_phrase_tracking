import sys, os,  requests, io, zipfile , re 
from collections import defaultdict
from typing import Dict, List
from nltk.stem import PorterStemmer
import numpy as np  
import pandas as pd 
from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer

#stemmer    
stemmer = PorterStemmer()


def nearest(items, pivot): 
    return min(items, key=lambda x: abs(x-pivot))

def download_and_unzip(url: str, destination_folder: str):
    
    """
    Downloads a file from the given URL and extracts its contents to the specified destination folder.

    Args:
        url (str): The URL of the file to download.
        destination_folder (str): The path to the folder where the contents of the zip file will be extracted.

    Returns:
        None

    Raises:
        None
    """

    # Send a GET request to download the file
    response = requests.get(url)

    # Check if the request was successful
    if response.status_code == 200:

        # Read the content of the response
        content = response.content

        # Create a file-like object from the response content
        file = io.BytesIO(content)

        # Extract the contents of the zip file
        with zipfile.ZipFile(file, 'r') as zip_ref:
            zip_ref.extractall(destination_folder)
    else:
        print("Failed to download the file.")




def map_stems_to_orig(original_corpus: List[List[str]], stemmer:PorterStemmer) -> Dict[str, str]:
    """
    Maps stemmed tokens to their corresponding original tokens in the given original_corpus.

    Args:
        original_corpus (List[List[str]]): A list of documents, where each document is a list of tokens.
        stemmer (Stemmer): An object that provides a stem() method to stem tokens.

    Returns:
        Dict[str, str]: A dictionary mapping stemmed tokens to their corresponding original tokens.
    """
    # Create a defaultdict to store the counts of stemmed tokens and their corresponding original tokens
    counts = defaultdict(lambda: defaultdict(int))
    # Create a dictionary to store the surface forms of the stemmed tokens
    surface_forms = {}

    # Iterate over each document in the original corpus
    for document in original_corpus: 
        # Iterate over each token in the document
        for token in document: 
            # Stem the token using the specified stemmer
            stemmed = stemmer.stem(token)
            # Increment the count of the stemmed token and its corresponding original token
            counts[stemmed][token] += 1 
    
    # Iterate over the stemmed tokens and their corresponding original tokens
    for stemmed, originals in counts.items():
        # Store the most frequent original token as the surface form of the stemmed token
        surface_forms[stemmed] = max(originals, key=lambda i: originals[i])
    
    # Return the dictionary mapping stemmed tokens to their surface forms
    return surface_forms



def get_top_idf_terms(df_tokens, stopwords = None , topn=20, n_df = None, sent_tokenized=False):

    def topn_tfidf_freq(tfidfvectorizer,tfidf_fit_transform,n=20):

        occ = np.asarray(tfidf_fit_transform.sum(axis=0)).ravel().tolist()
        counts_df = pd.DataFrame({'term':tfidfvectorizer.get_feature_names_out(), 'occurrences':occ})
        freqtable = counts_df.sort_values(by='occurrences', ascending=False).head(n)


        weights = np.asarray(tfidf_fit_transform.mean(axis=0)).ravel().tolist()
        weights_df = pd.DataFrame({'term':tfidfvectorizer.get_feature_names_out(), 'weights':weights})
        weights_dfout = weights_df.sort_values(by='weights', ascending=False).head(n)
        return freqtable, weights_dfout 


    if sent_tokenized:
        df = df_tokens.apply(lambda x: ' '.join([item for sublist in x for item in sublist]))
    else: 
        df = df_tokens.apply(lambda x: ' '.join([word for word in x]))

    Tfidf_vectorizer = TfidfVectorizer(ngram_range = (1,1)
                                        , min_df = .005 
                                        , max_df=.2 if n_df==None else .5
                                        , lowercase=True
                                        , analyzer = 'word'
                                        , max_features = None 
                                        , binary = False
                                        , use_idf = True 
                                        , smooth_idf = False
                                        , sublinear_tf=True
                                        ,  stop_words=stopwords)
    
    tfidf_vec = Tfidf_vectorizer.fit_transform([v for v in df])
    topterms = topn_tfidf_freq(Tfidf_vectorizer,tfidf_vec,n=topn)
    return topterms[0]


def incorp_static_phrases(doc,phrase_tups: dict):

    if type(phrase_tups).__name__ == 'dict': 
        
        for item in phrase_tups.items():
            redata = re.compile(re.escape(item[0]),re.IGNORECASE)
            doc = redata.sub(item[1], doc)
        
        return doc 
    
    elif type(phrase_tups).__name__ =='list':
        for item in phrase_tups: 
            redata = re.compile(re.escape(item[0]),re.IGNORECASE)
            doc = redata.sub(item[1], doc)
        return doc 
    else: 
        raise Exception('phrases to substitute must be in form of dictionary or tuple')
    



def build_wordfreq_stat_dict(word_tokenized_doc_list):
    """
    Builds a word frequency statistics dictionary from a list of word tokenized documents.

    Args:
        word_tokenized_doc_list (list): A list of word tokenized documents.

    Returns:
        dict: A dictionary containing the word frequency statistics, where the keys are the tokens and the values are the frequencies.

    Example:
        >>> word_tokenized_doc_list = [['I', 'love', 'programming'], ['Programming', 'is', 'fun']]
        >>> build_wordfreq_stat_dict(word_tokenized_doc_list)
        {'I': 1, 'love': 1, 'programming': 2, 'is': 1, 'fun': 1}
    """
    corpus_len = 0
    word_freq_dict = defaultdict(int)
    for _list in word_tokenized_doc_list:
        for _sentence in _list:
            for token in _sentence:
                word_freq_dict[token] += 1
                corpus_len += 1
    return word_freq_dict
