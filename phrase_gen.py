import polars as pl
import os, nltk, spacy , sys, logging, datetime
from pathlib import Path
from itertools import chain

import nltk
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize, sent_tokenize
#from spello.model import SpellCorrectionModel

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')


#logger
logger = logging.getLogger(__name__)
logging.basicConfig()
logging.getLogger().setLevel(logging.DEBUG)



cfg = {'data_path':'/home/zjc1002/Mounts/data/cfpb/cfpb_complaints.csv'
       , 'cache_dir':"/home/zjc1002/Mounts/temp/"
       , 'project_root': '/home/zjc1002/Mounts/code/key_phrase_tracking'
       , 'incols':['Date received', 'Product','Consumer complaint narrative','Company public response']
       , 'text_cols':['Consumer complaint narrative','Company public response']

       , 'spell_correct_model_url': "https://haptik-website-images.haptik.ai/spello_models/en_large.pkl.zip"
       , 'spacy_model_path' : "en_core_web_lg"
       , 'stop_words_add':['my','xxxxxxxx','xxxx','xxx','xx']
       , 'stop_grams_add': ['mailbox_full']
       , 'n_samp':1000

       , 'date_col':'Date received'
       , 'incorp_ngrams_2_txt': True
       , 'ngram_len' : 2
       , 'min_freq' : 10 
       , 'pmi_thresh' : .5
       , 'tthresh': .01

       , 'start_date': '2015-04-01'
       , 'end_date': '2015-04-30'
       , 'cur_date': '2015-04-01'

       , 'ngram_cutoff_periods': 6
       , 'ngram_pv_thresh': .01
       , 'ngram_pmi_thresh': .7
       , 'max_tfidf_ngrams': 100 
       , 'rule1_topn': 15
       , 'd2':1.128

       }

sys.path.append(cfg['project_root'])
from utils import  map_stems_to_orig, get_topN_keywords_YAKE
from polars_utils import preprocess_text, gen_control_limits

#manually download spacy model to disk for use in future 
cache_dir= cfg['cache_dir']
spacy_model_path=cfg['spacy_model_path']
spell_correct_model_url = cfg['spell_correct_model_url']

data_path = cfg['data_path']
incols = cfg['incols']
text_cols = cfg['text_cols']
date_col = cfg['date_col']
n_samp = cfg['n_samp'] #number of documents to sample per date 

stop_words_add = cfg['stop_words_add']
stopgrams = cfg['stop_grams_add']

incorp_ngrams_2_txt = cfg['incorp_ngrams_2_txt']
ngram_len = cfg['ngram_len']
min_freq = cfg['min_freq']
pmi_thresh = cfg['pmi_thresh']
tthresh = cfg['tthresh']

start_date = datetime.datetime.strptime(cfg['start_date'],'%Y-%m-%d')
end_date = datetime.datetime.strptime(cfg['end_date'],'%Y-%m-%d')
cur_date = datetime.datetime.strptime(cfg['cur_date'],'%Y-%m-%d')

ngram_cutoff_periods = cfg['ngram_cutoff_periods']
ngram_pv_thresh = cfg['ngram_pv_thresh']
ngram_pmi_thresh = cfg['ngram_pmi_thresh']
max_tfidf_ngrams = cfg['max_tfidf_ngrams']
rule1_topn = cfg['rule1_topn']
d2 = cfg['d2']



#stemmer 
stemmer = PorterStemmer()


#download spacy model 
if not os.path.exists(Path(cache_dir,spacy_model_path).as_posix()):
    spacy.cli.download(spacy_model_path)

#nltk.download('stopwords')
#from nltk.corpus import stopwords # For removing stopwords
stopwords = nltk.corpus.stopwords.words('english')
stopwords= set([str(s_t_r).lower() for s_t_r in list(stopwords) + stop_words_add if str(s_t_r).lower() ])

#load spacy model 
nlp = spacy.load(spacy_model_path)
nlp.to_disk(os.path.join(cache_dir,spacy_model_path))
nlp = spacy.load(os.path.join(cache_dir,spacy_model_path))
#nlp.add_pipe('textrank')

#stemmer & stopwords
stemmer = PorterStemmer()
stopwords = nltk.corpus.stopwords.words('english')
stopwords= set([str(s_t_r).lower() for s_t_r in list(stopwords) + cfg['stop_words_add'] if str(s_t_r).lower() not in ['up','down']])

#download spacy model 
if not os.path.exists(Path(cache_dir,spacy_model_path).as_posix()):
    spacy.cli.download(spacy_model_path)

#load spacy model 
nlp = spacy.load(spacy_model_path)
nlp.to_disk(os.path.join(cache_dir,spacy_model_path))
nlp = spacy.load(os.path.join(cache_dir,spacy_model_path))
#nlp.add_pipe('textrank')

## ONLY USE IF NEEDED
# #download the spell correction model (just copy in paste if behind firewall) 
# if os.path.exists(cache_dir):
#     print("The spell check model folder exists.")
# else:
#     download_and_unzip(spell_correct_model_url, cache_dir)

# #spell correction model, we dont use it 
# sp = SpellCorrectionModel(language='en')
# sp.load(Path(cache_dir,spell_correct_model_url.split('/')[-1].replace('.zip','')))
# sp.config.min_length_for_spellcorrection = 4

# Read the CSV file into a polars DataFrame (and take a sample)
df = (pl.read_csv(data_path, has_header=True)[incols]
      ).drop_nulls(subset=text_cols
                   ).filter(
                       pl.int_range(pl.len()
                                    ).shuffle().over(date_col)<n_samp)

# Create a new column 'input_txt' by concatenating 'Consumer complaint narrative' and 'Company public response'
# AND sent tokenize 
df = (df.with_columns(
        pl.concat_str([pl.col(x) for x in text_cols]).alias('input_col'))
        ).select(pl.col("*").exclude(text_cols)
             ).with_columns(sent_tokenized = pl.col("input_col").map_elements(
                 sent_tokenize, strategy= 'thread_local')
                 )
          

#create map of stems to original words
stem_map = map_stems_to_orig([word_tokenize(' '.join(sent.to_list()
                                                     ).lower()) for sent in  df['sent_tokenized'] ], stemmer) 


#word tokenize and clean 
df = (df.with_columns(
    word_tokenized = pl.col("sent_tokenized").map_elements(
        lambda x: [word_tokenize(preprocess_text(sent)) for sent in  x ] , strategy= 'thread_local'))
        ).with_columns(
            input_col_clean = pl.col('input_col').map_elements(
                lambda x: preprocess_text(x)))

#phrase generation with YAKE 
df = df.with_columns(
    yake_keywords = pl.col('input_col_clean').map_elements(
        lambda x : get_topN_keywords_YAKE(x,30).split(',')))

#if you want to create unigrams from the ngrams identified from one of above processes 
if incorp_ngrams_2_txt: 
    
    df = df.with_columns(
        pl.struct(['word_tokenized','yake_keywords']).map_elements(
        
            lambda x: list(chain.from_iterable(
                [[word_tokenize(' '.join(x['word_tokenized'][i]).replace(y,y.replace(' ','_')))
                   for y in x['yake_keywords']] for i in range(0,len(x['word_tokenized']))]))
    ).alias('word_tokens'))



gen_control_limits(df
                       , date_col
                       , start_date
                       , end_date
                       , cur_date

                       , stopwords 
                       , stopgrams 
                       , ngram_len
                       , min_freq

                       , ngram_cutoff_periods
                       , ngram_pv_thresh
                       , ngram_pmi_thresh
                       , max_tfidf_ngrams
                       , rule1_topn
                       , d2)
