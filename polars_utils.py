import math
from scipy.stats import t, chi2
import re
import string
from bs4 import BeautifulSoup
from unidecode import unidecode
import contractions
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from spello.model import SpellCorrectionModel
import polars as pl 
from collections import defaultdict
#stemmer 
stemmer = PorterStemmer()
stopwords = nltk.corpus.stopwords.words('english')
stopwords= set([str(s_t_r).lower() for s_t_r in list(stopwords) + ['xxxxxxxx','xxxx','xxx','xx'] if str(s_t_r).lower() not in ['up','down']])


def prep_data(df,agg_by:list = ['Product'], count_of:str = 'keywords2',datecol='Date received'):
    
    # #format dates 
    df = df.with_columns(date_col = pl.col(datecol).str.to_datetime("%m/%d/%Y"))
    df = df.with_columns(year_month=pl.col('date_col').dt.strftime("%Y/%m"))

    #generate datafarme of ngrams by date and product to plot / derive monitoring rules to define EMERINGING
    plot_df = df.groupby(['date_col']+agg_by+[count_of]).count()
    plot_df = plot_df.sort(by=['date_col']+agg_by)

    #calculate %chagne in each group / ngram
    plot_df = plot_df.sort(['date_col']+agg_by).with_columns([pl.col('count').pct_change().over(agg_by).alias('pct_chg')])

    return plot_df 

def preprocess_text(text: str, stopwords=stopwords,  stemmer=stemmer):
    def remove_html(text):
        soup = BeautifulSoup(text)

        text = soup.get_text()
        return text

    def remove_urls(text):
        pattern = re.compile(r'https?://(www\.)?(\w+)(\.\w+)(/\w*)?')
        text = re.sub(pattern, "", text)
        return text

    def remove_emails(text):
        pattern = re.compile(r"[\w\.-]+@[\w\.-]+\.\w+")
        text = re.sub(pattern, "", text)
        return text

    def handle_accents(text):
        text = unidecode(text)
        return text

    def remove_unicode_chars(text):
        text = text.encode("ascii", "ignore").decode()
        return text

    def remove_punctuations(text):
        text = re.sub('[%s]' % re.escape(string.punctuation), " ",text)
        return text
    
    def remove_digits(text):
        pattern = re.compile("\w*\d+\w*")
        text = re.sub(pattern, "",text)
        return text

    def remove_stopwords(text,stopwords=stopwords):
        return " ".join([word for word in str(text).split(" ") if str(word).lower() not in stopwords])

    def remove_extra_spaces(text):
        text = re.sub(' +', ' ', text).strip()
        return text
    
    #return remove_html(contractions.fix(remove_urls(remove_emails(handle_accents(remove_unicode_chars(remove_punctuations(remove_digits(remove_stopwords(remove_extra_spaces(correct_spelling(sp_model,text)))))))))))
    clean_txt = remove_stopwords(remove_html(
                contractions.fix(
                    remove_urls(
                        remove_emails(
                            handle_accents(
                                remove_unicode_chars(
                                    remove_punctuations(
                                        remove_digits(
                                            remove_stopwords(
                                                remove_extra_spaces(text),stopwords=stopwords)
                                                )))))))))
    if stemmer is not None: 
        return ' '.join(stemmer.stem(word) for word in clean_txt.split()) 
    else:
        return clean_txt                              

    
def correct_spelling( model: SpellCorrectionModel,text: str) -> str:
    """
    Corrects the spelling of the given text using the specified spell correction model.

    Args:
        text (str): The text to correct.
        model (SpellCorrectionModel): The spell correction model to use.

    Returns:
        str: The corrected text.

    Raises:
        None
    """
    # Correct the spelling of the text
    corrected_text = model.spell_correct(text)['spell_corrected_text']

    return corrected_text

def remove_duplicate_words(string): 
    # Use a regular expression to match consecutive duplicate words, 
    # capturing the first instance of the word in a group 
    pattern = r'(\b\w+\b)(?:\s+\1)+' 
        
    # Use the sub() function to replace all instances of the pattern with 
    # the first captured group (the first instance of the word) 
    return re.sub(pattern, r'\1', string) 



def rightTypes(ngrams): 
    
    """
    Checks if the given ngram consists of the right types of words.

    Args:
        ngrams (list): A list of words representing an ngram.

    Returns:
        bool: True if the ngram consists of the right types of words, False otherwise.
    """

    if '-pron-' in ngrams or '' in ngrams or ' ' in ngrams or 't' in ngrams: 
        return False
    
    for word in ngrams: 
        if word in stopwords: 
            return False 
    
    acceptable_types = ('JJ', 'JJR', 'JJS', 'NN', 'NNS', 'NNP', 'NNPS')
    second_type = ('NN','NNS','NNP','NNPS')
    tags = nltk.pos_tag(ngrams)

    if tags[0][1] in acceptable_types and tags[1][1] in second_type: 
        return True 
    else: 
        return False
    

def rightTypesTri(ngram): 
    
    """
    Checks if the given ngram consists of the right types of words.

    Args:
        ngrams (list): A list of words representing an ngram.

    Returns:
        bool: True if the ngram consists of the right types of words, False otherwise.
    """

    if '-pron-' in ngram or '' in ngram or ' ' in ngram or 't' in ngram or '  ' in ngram: 
        return False
    
    for word in ngram: 
        if word in stopwords: 
            return False 
    
    first_type = ('JJ', 'JJR', 'JJS', 'NN', 'NNS', 'NNP', 'NNPS')
    third_type = ('JJ', 'JJR', 'JJS', 'NN', 'NNS', 'NNP', 'NNPS')
    tags = nltk.pos_tag(ngram)

    if tags[0][1] in first_type and tags[2][1] in third_type: 
        return True 
    else: 
        return False


def ngram_phraser(word_tokenized_doc_list, score_method = 'Student_test', ngram_len = 2, min_freq=20, tthresh=.01, pmi_thresh =.5):

    #generate list of all possible ngrams from docs 
    document_ngram_list = [['_'.join(_sentence[i:i + ngram_len]) for _sentence in _document for i in range(len(_sentence) - ngram_len +1)] for _document in word_tokenized_doc_list]
                        

    #get word frequency and identify number of unique tokens in corpus 
    corpus_len = 0 
    word_freq_dict = defaultdict(int)
    token_list = []

    #generate word counts , tokens, and doc length 
    for _list in word_tokenized_doc_list: 
        for _sentence in _list: 
            for token in _sentence: 
                word_freq_dict[token]+=1
                corpus_len += 1
                token_list.append(token)

    #get ngram frequency dictionary 
    ngram_freq_dict = defaultdict(int)
    ngram_doc_dict = defaultdict(list)
    for i, _list in enumerate(document_ngram_list):
        for ngram in _list: 
            ngram_freq_dict[ngram]+=1
            ngram_doc_dict[ngram].append(i)


    def invalid_ngram(*gram):
        ngram='_'.join(gram)
        if ngram_freq_dict[ngram]==0:
            result = True
            ngram_freq_dict.pop(ngram)
        else:
            result=False 
        return result 

    #use nltk collactions to try differnt scoring methods 
    #bigrams using PMI as  metric 
    if ngram_len == 2: 

        bigrams = nltk.collocations.BigramAssocMeasures()
        bigramFinder = nltk.collocations.BigramCollocationFinder.from_words(token_list)
        bigramFinder.apply_freq_filter(min_freq)
        bigramFinder.apply_ngram_filter(invalid_ngram)

        if score_method == 'PMI':

            ngramTable = (pl.DataFrame(list(bigramFinder.score_ngrams(bigrams.pmi))
                                    , schema = {'ngram':pl.List(pl.String), 'PMI':pl.Float32}
                                    ).sort(by='PMI', descending=True)
                                    ).lazy(
                                        ).select(
                                                    [   pl.col('*').exclude('ngram')
                                                        , pl.col('ngram').apply(lambda x: '_'.join(x)).alias('ngram')
                                                    ]
                                                ).select(
                                                    [ pl.col('*')
                                                    , (pl.col('PMI') /   pl.col('ngram').apply(lambda x: (-1*(math.log2(ngram_freq_dict[x]/corpus_len))))).alias('N-PMI')
                                                    , (pl.col('ngram').apply(lambda x: ngram_freq_dict[x])).alias('ngram_freq')
                                                    ]
                                                    ).collect()


        elif score_method == 'Student_t':

            ngramTable = pl.DataFrame(list(bigramFinder.score_ngrams(bigrams.student_t))
                                    , schema = {'ngram':pl.List(pl.String), 't':pl.Float32}
                                    ).sort(by='t', descending=True
                                            ).filter(pl.col('ngram').apply(lambda x: rightTypes(x))
                                                    ).select(
                                                            [   pl.col('*').exclude('ngram')
                                                                , pl.col('ngram').apply(lambda x: '_'.join(x)).alias('ngram')
                                                            ]   
                                                       )
            
            #calculate p vals 
            ngramTable = ngramTable.select(
                                        [ pl.col('*')
                                        , t.sf([item for sublist in ngramTable.select('t').to_numpy() for item in sublist]   , len(ngram_freq_dict)-1)
                                        , (pl.col('ngram').apply(lambda x: ngram_freq_dict[x])).alias('ngram_freq')

                                        ]
                                        ).rename({'literal':'p'})
        
        elif score_method == 'Chi_sq': 

            ngramTable = pl.DataFrame(list(bigramFinder.score_ngrams(bigrams.chi_sq))
                                    , schema = {'ngram':pl.List(pl.String), 'chi-sq':pl.Float32}
                                    ).sort(by='chi-sq', descending=True
                                            ).select(
                                                    [   
                                                        
                                                        pl.col('*').exclude('ngram')
                                                        , pl.col('ngram').apply(lambda x: '_'.join(x)).alias('ngram')
                                                    ]
                                                )
            #calculate pvals
            ngramTable = ngramTable.select(
                                                    [ pl.col('*')
                                                    , chi2.sf([item for sublist in ngramTable.select('chi-sq').to_numpy() for item in sublist]   , 1)
                                                    , (pl.col('ngram').apply(lambda x: ngram_freq_dict[x])).alias('ngram_freq')

                                                    ]
                                                    ).rename({'literal':'p'})
        else:
            print('NOT VALID SCORING METHOD!')
            return False
    
    elif ngram_len==3: 

        trigrams = nltk.collocations.TrigramAssocMeasures()
        trigramFinder = nltk.collocations.TrigramCollocationFinder.from_words(token_list)
        trigramFinder.apply_freq_filter(min_freq)
        trigramFinder.apply_ngram_filter(invalid_ngram)

        if score_method=='PMI': 

            ngramTable = (pl.DataFrame(list(trigramFinder.score_ngrams(trigrams.pmi))
                                    , schema = {'ngram':pl.List(pl.String), 'PMI':pl.Float32}
                                    ).sort(by='PMI', descending=True)
                                    ).lazy(
                                        ).select(
                                                    [   pl.col('*').exclude('ngram')
                                                        , pl.col('ngram').apply(lambda x: '_'.join(x)).alias('ngram')
                                                    ]
                                                ).select(
                                                    [ pl.col('*')
                                                    , (pl.col('PMI') /   pl.col('ngram').apply(lambda x: (-1*(math.log2(ngram_freq_dict[x]/corpus_len))))).alias('N-PMI')
                                                    , (pl.col('ngram').apply(lambda x: ngram_freq_dict[x])).alias('ngram_freq')
                                                    ]
                                                    ).collect()
            

        elif score_method == 'Student_t':

            ngramTable = pl.DataFrame(list(trigramFinder.score_ngrams(trigrams.student_t))
                                    , schema = {'ngram':pl.List(pl.String), 't':pl.Float32}
                                    ).sort(by='t', descending=True
                                            ).filter(pl.col('ngram').apply(lambda x: rightTypes(x))
                                                    ).select(
                                                            [   pl.col('*').exclude('ngram')
                                                                , pl.col('ngram').apply(lambda x: '_'.join(x)).alias('ngram')
                                                            ]   
                                                       )
            
            #calculate p vals 
            ngramTable = ngramTable.select(
                                        [ pl.col('*')
                                        , t.sf([item for sublist in ngramTable.select('t').to_numpy() for item in sublist]   , len(ngram_freq_dict)-1)
                                        , (pl.col('ngram').apply(lambda x: ngram_freq_dict[x])).alias('ngram_freq')

                                        ]
                                        ).rename({'literal':'p'})
        
        elif score_method == 'Chi_sq': 

            ngramTable = pl.DataFrame(list(trigramFinder.score_ngrams(trigrams.chi_sq))
                                    , schema = {'ngram':pl.List(pl.String), 'chi-sq':pl.Float32}
                                    ).sort(by='chi-sq', descending=True
                                            ).select(
                                                    [   
                                                        
                                                        pl.col('*').exclude('ngram')
                                                        , pl.col('ngram').apply(lambda x: '_'.join(x)).alias('ngram')
                                                    ]
                                                )
            #calculate pvals
            ngramTable = ngramTable.select(
                                                    [ pl.col('*')
                                                    , chi2.sf([item for sublist in ngramTable.select('chi-sq').to_numpy() for item in sublist]   , 1)
                                                    , (pl.col('ngram').apply(lambda x: ngram_freq_dict[x])).alias('ngram_freq')

                                                    ]
                                                    ).rename({'literal':'p'})

    #generate final output dic of ngrams mapped to docs 
    doc_ngram_dict = defaultdict(list)
    if score_method in ['PMI']: 
        for ngram in ngramTable.filter(pl.col('N-PMI')>pmi_thresh).select('ngram').iter_rows(): 
            for doc in ngram_doc_dict[ngram[0]]:
                doc_ngram_dict[doc].append(ngram[0])


    #extract all ngram counts for each document in original input 
    doc_to_ngram = [doc_ngram_dict[i] for i in range(len(word_tokenized_doc_list))]

    return ngramTable, doc_to_ngram