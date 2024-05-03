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
import logging, datetime
from itertools import chain
import numpy as np
import pandas as pd

#logger
logger = logging.getLogger(__name__)
logging.basicConfig()
logging.getLogger().setLevel(logging.DEBUG)

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






def gen_control_limits(df: pl.DataFrame
                       , date_col: str= None
                       , start_date: datetime.datetime = None
                       , end_date: datetime.datetime = None 
                       , cur_date: datetime.datetime = None

                       , stopwords = None
                       , stopgrams = None
                       , ngram_len: int = 2
                       , ngram_min_freq: int = 10

                       , ngram_cutoff_periods:int = None
                       , ngram_pval_thresh: float = None 
                       , ngram_pmi_thresh:float = None
                       , max_tfidf_ngrams:int  = None
                       , rule1_topn: int = None
                       , d2: float = None):
    
    if stopwords is None: 
        stopwords = []
    if stopgrams is None: 
        stopgrams = []

    #load data, format dates, and subset based on in scope date range 
    cl_data = (df.with_columns(date_col = pl.col(date_col).str.to_datetime("%m/%d/%Y"))
            ).with_columns(
                pl.col('date_col').map_elements(
                    lambda x: datetime.datetime(x.year,x.month,x.day)- datetime.timedelta(days=datetime.datetime(x.year,x.month,x.day).weekday()-4)
                    if datetime.datetime(x.year,x.month,x.day).weekday() in [5,6] 
                    else datetime.datetime(x.year,x.month,x.day))
                ).sort(by='date_col'
                    ).filter((pl.col('date_col')>=start_date) & (pl.col('date_col')<=cur_date))

    logger.debug(f"Dataframe shape after filtering: {cl_data.shape}")
    logger.debug(f"Dataframe contains ndates after filtering: {cl_data.select(pl.col('date_col')).n_unique()}")
    logger.debug(f"Dataframe contains nunique complaints after filtering: {cl_data.select(pl.col('input_col')).n_unique()}")

    if cl_data.shape[0]<1: 
        raise Exception(f"No data exists in date range specified. Data exists between {cl_data.select(pl.col('date_col').min())} and {cl_data.select(pl.col('date_col').max())} for reference")


    #ngramcutoff (we only identify ngrams that are found prior to the ngram cutoff date, this ensure we have enough 'future' periods to evaluate control limits)
    ngram_cutoff_date= sorted([v[0] for v in cl_data.select('date_col').unique().to_numpy()])[-ngram_cutoff_periods]

    #
    #get ngrams
    #

    #PMI
    ngramTable_pmi, doc_to_ngram_pmi = ngram_phraser(
                                                    [v for v in cl_data.filter(pl.col('date_col')<= ngram_cutoff_date).select(['word_tokens'])][0]
                                                    , score_method = 'PMI'
                                                    , ngram_len =ngram_len
                                                    , min_freq=ngram_min_freq
                                                    , pmi_thresh =ngram_pmi_thresh)


    ngramTable_pmi = ngramTable_pmi.filter(pl.col('N-PMI')>=pmi_thresh).sort(by=['N-PMI','ngram_freq'],descending=True)

    #Chi2
    ngramTable_chi2, doc_to_ngram_chi2 = ngram_phraser(
                                                    [v for v in cl_data.filter(pl.col('date_col')<= ngram_cutoff_date).select(['word_tokens'])][0]
                                                    , score_method = 'Chi_sq'
                                                    , ngram_len = ngram_len
                                                    , min_freq= ngram_min_freq
                                                    , tthresh =ngram_pval_thresh) #the pval for chi2 and t-test


    ngramTable_chi2 = ngramTable_chi2.filter(pl.col('p')<=tthresh).sort(by=['chi-sq','ngram_freq'],descending=True).head(200)

    #ttest
    ngramTable_ttest, doc_to_ngram_ttest = ngram_phraser(
                                                    [v for v in cl_data.filter(pl.col('date_col')<= ngram_cutoff_date).select(['word_tokens'])][0]
                                                    , score_method = 'Student_t'
                                                    , ngram_len =ngram_len
                                                    , min_freq=7
                                                    , tthresh =tthresh)


    ngramTable_ttest = ngramTable_chi2.filter(pl.col('p')<=ngram_pval_thresh).sort(by=['p','ngram_freq'],descending=True).head(300)

    #unigrams 
    ngramTable_tfidf = get_top_idf_terms(cl_data['word_tokens'].apply(lambda x: list(chain.from_iterable(x))).to_pandas(), topn=max_tfidf_ngrams
                                        , sent_tokenized=False, stopwords=[v for v in stopwords])


    #combine ngrams from all three approaches 
    #filter out ngrams with stopgrams 
    ngram_list = list(set(list(ngramTable_pmi['ngram']) + list(ngramTable_chi2['ngram']) + list(ngramTable_ttest['ngram']) + list(ngramTable_tfidf['term'])))
    ngram_list_final = [s for s in ngram_list if all(xs.lower() not in s.lower() for xs in stopgrams)]
    logger.info(f"{len(ngram_list)-len(ngram_list_final)} ngram removed due to stopgrams")
    logger.info(f"ngram generation complete. {ngramTable_ttest['ngram'].n_unique()} ngrams generated from ttest, {ngramTable_chi2['ngram'].n_unique()} ngrams generated from chi2, {ngramTable_pmi['ngram'].n_unique()} ngrams generated from pmi, and {len(ngramTable_tfidf['term'].unique())} ngrams generated from tfidf")
    logger.info(f"total of {len(ngram_list_final)} ngrams generated after filtering out stopgrams")
    
    #incorporate static ngrams into text 
    logger.info(f"ngram substitution begining for {len([len(ngram.split('_')) for ngram in ngram_list_final if len(ngram.split('_'))>1])}")
    ngram_map_tups = [((' '.join(ngram.split('_'))),ngram) for ngram in ngram_list_final if len(ngram.split('_'))>1]

    #incorp static phrases into document 
    cl_data = cl_data.select([pl.col('*')
             , (pl.col('word_tokens').apply(lambda x:  incorp_static_phrases(' '.join([v for v in list(chain.from_iterable(x)) if v not in stopgrams]),ngram_map_tups).split())).alias('word_tokens2')])


    #part3
    #generate count freqs for each data and % of total freqs for each unigram 
    
    count_stats_dict = defaultdict(dict)
    for i,dt in enumerate(sorted(set([v for v in cl_data.select('date_col').to_series()]))): 
        df_t = cl_data.filter(pl.col('date_col')==dt).select([pl.col('word_tokens2')])
        if len(df_t)>0: 
            count_stats_dict[dt] = build_wordfreq_stat_dict(df_t)
        else:
            print(dt)

    #convert count stats from dict to df (index = key phrases, columns = date counts)
    cnt_summary_df = pd.DataFrame.from_dict(count_stats_dict).fillna(0)


    #Control limit start 
    try: 
        start_date_train = pd._libs.tslib.Timestamp(start_date)
        start_date_train_idx = list(cnt_summary_df.columns>=start_date_train).index(True)
    except Exception: 
        raise ValueError("An error occurred while processing the start date. Please ensure it's in the correct format.")



    try:
        end_date_train  = pd._libs.tslib.Timestamp(end_date)

        if end_date_train < max(cnt_summary_df.columns):
            logger.info(f"end date {end_date_train} is prior to last date in data {max(cnt_summary_df.columns)}")
            end_date_train_idx = list(cnt_summary_df.columns>end_date_train).index(True)-1
        if end_date_train == max(cnt_summary_df.columns):
            logger.info(f"end date {end_date_train} is the last date in data {max(cnt_summary_df.columns)}")
            end_date_train_idx = list(cnt_summary_df.columns>= end_date_train).index(True)
        if end_date_train > max(cnt_summary_df.columns):
            logger.info(f"end date {end_date_train} is after last date in data {max(cnt_summary_df.columns)}")
            nearest_date_end = nearest(cnt_summary_df.columns, end_date_train) 
            logger.info(f"closest date to end date is {nearest_date_end}")
            end_date_train_idx = list(cnt_summary_df.columns>=nearest_date_end).index(True)
        
    except Exception: 
        raise Exception("start and end date assignment failed")

    #id current date, or date closest to current date 
    cur_date_test = pd._libs.tslib.Timestamp(cur_date)

    try: 
        cur_date_test_idx = cnt_summary_df.columns.get_loc(cur_date_test)
    except: 
        nearest_date = nearest(cnt_summary_df.columns, cur_date_test)
        cur_date_test_idx = cnt_summary_df.columns.get_loc(nearest_date)



    #part 5: build control limit thresholds 

    #filter 
    DataDev = cnt_summary_df.iloc[:,start_date_train_idx:cur_date_test_idx+1].copy()
    DataDev = DataDev.loc[[v for v in ngram_list_final if v in DataDev.index.values]].fillna(0)

    #control limit derived inputs (how long back to use in establishing means)
    K = end_date_train_idx - start_date_train_idx + 1   # number of days to use in building control limits  
    curIdx = cur_date_test_idx-start_date_train_idx     # index for current day in DataDev datafarme
    showK = cur_date_test_idx - start_date_train_idx + 1 # tommorrows day in DataDev dataset (t+1)


    #calculate x mean 
    DataDev['x_bar'] = np.average(DataDev.iloc[:,:K],axis=1)

    #calc moving avg 
    MR_sum = 0 
    for i in range(K-1):
        MR_sum += abs(DataDev.iloc[:,i+1] - DataDev.iloc[:,i])

    DataDev['MR_avg'] = MR_sum/(K-1)
    DataDev['sigma'] = DataDev['MR_avg']/d2

    #UPPER AND LOWER CONTROL LIMITS(3std from moving avg) 
    DataDev['LCL'] = DataDev['x_bar'] - 3*DataDev['sigma']
    DataDev['UCL'] = DataDev['x_bar'] + 3*DataDev['sigma']  

    #Zone 1 Upper and lower limits 
    DataDev['LCL_ZoneA_l'] = DataDev['LCL']
    DataDev['LCL_ZoneA_u'] = DataDev['x_bar'] - 2*DataDev['sigma']
    DataDev['UCL_ZoneA_l'] = DataDev['x_bar'] + 2*DataDev['sigma'] 
    DataDev['UCL_ZoneA_u'] = DataDev['UCL']

    #Zone 2 Upper and Lower Limits 
    DataDev['LCL_ZoneB_l'] = DataDev['x_bar'] - 2*DataDev['sigma']
    DataDev['LCL_ZoneB_u'] = DataDev['x_bar'] -  DataDev['sigma'] 
    DataDev['UCL_ZoneB_l'] = DataDev['x_bar'] + DataDev['sigma'] 
    DataDev['UCL_ZoneB_u'] = DataDev['x_bar'] + 2*DataDev['sigma']

    #Zone 3 Upper and Lower Limits
    DataDev['LCL_ZoneC_l'] = DataDev['x_bar'] - DataDev['sigma']
    DataDev['LCL_ZoneC_u'] = DataDev['x_bar'] 
    DataDev['UCL_ZoneC_l'] = DataDev['x_bar']  
    DataDev['UCL_ZoneC_u'] = DataDev['x_bar'] + DataDev['sigma']


    #part 6: generate rules to identify key phrases that are emerging in some degree 

    #Rule 1: Beyond limits (one or more points beyond the control limits)
    DataDev['Rule1'] = DataDev.iloc[:,curIdx]>DataDev['UCL']
    DataDev['Rule1_pct'] = (DataDev.iloc[:,curIdx]-DataDev['UCL'])/DataDev['UCL']
    DataDev = DataDev.replace([np.inf, -np.inf], np.nan).sort_values(by='Rule1_pct', ascending=False)
    plot_rule1 = DataDev[DataDev['Rule1']==True].sort_values(by='Rule1_pct', ascending=False).iloc[:rule1_topn, :showK].transpose()
    logger.info(f"Rule 1 Ngrams: {list(plot_rule1)}")


    #Rule 2: Zone A(2 out of 3 consequtive points in Zone A or beyond)
    _tmp = DataDev.iloc[:,curIdx-2:curIdx+1]
    for i in range(3):
        _tmp.iloc[:,i] = (_tmp.iloc[:,i]>DataDev['UCL_ZoneA_l'])

    DataDev['Rule2'] = (_tmp.sum(axis=1)==2) & (_tmp.iloc[:,-1]==True)
    plot_rule2 = DataDev[DataDev['Rule2']==True].iloc[:,:showK].transpose()
    logger.info(f"Rule 2 Ngrams: {list(plot_rule2)}")


    #Rule 3: Zone B (4 out of 5 consecutive points in Zone B or beyond)
    _tmp = DataDev.iloc[:,curIdx-4:curIdx+1]
    for i in range(5):
        _tmp.iloc[:,i] = (_tmp.iloc[:,i]>DataDev['UCL_ZoneB_l'])

    DataDev['Rule3'] = (_tmp.sum(axis=1)==4) & (_tmp.iloc[:,-1]==True)
    plot_rule3 = DataDev[DataDev['Rule3']==True].iloc[:,:showK].transpose()
    logger.info(f"Rule 3 Ngrams: {list(plot_rule3)}")


    #Rule 4: Zone C (7 or more consecutive points on one side of the average(in Zone C or beyond))
    _tmp = DataDev.iloc[:,curIdx-6:curIdx+1]
    for i in range(7):
        _tmp.iloc[:,i] = (_tmp.iloc[:,i]>DataDev['UCL_ZoneC_l'])

    DataDev['Rule4'] = (_tmp.sum(axis=1)==7) 
    plot_rule4 = DataDev[DataDev['Rule4']==True].iloc[:,:showK].transpose()
    logger.info(f"Rule 4 Ngrams: {list(plot_rule4)}")

    #Rule 5(trending up): Trend(7 conseecutive points trending up or trending down)
    _tmp = DataDev.iloc[:,:curIdx+1]
    for i in range(curIdx-5, curIdx+1):
        _tmp.iloc[:,i] = (DataDev.iloc[:,i]>DataDev.iloc[:,i-1])


    DataDev['Rule5_trend'] = (_tmp.iloc[:,curIdx-5:].sum(axis=1)==6)
    plot_rule5 = DataDev[DataDev['Rule5_trend']==True].iloc[:,:showK].transpose()
    logger.info(f"Rule 5 Ngrams: {list(plot_rule5)}")

    return DataDev, plot_rule1, plot_rule2, plot_rule3, plot_rule4, plot_rule5
