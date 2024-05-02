import pandas as pd
import polars as pl
from spello.model import SpellCorrectionModel
import io, re, string, spello,requests , zipfile, os, nltk, spacy , pytextrank, sys, math, logging, datetime
import numpy as np 
from pathlib import Path

from bs4 import BeautifulSoup # For removing HTML
import contractions # For expanding contractions
from unidecode import unidecode # For handling accented words
from fuzzywuzzy import process
from functools import reduce
from scipy.stats import t , chi2

import yake 
from  typing import List
import itertools
import collections
from collections import defaultdict

import nltk
from nltk import ngrams 
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize, sent_tokenize
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')

sys.path.append(cfg['project_root'])
from utils import download_and_unzip, map_stems_to_orig, get_top_idf_terms, incorp_static_phrases, nearest, build_wordfreq_stat_dict
from polars_utils import prep_data, preprocess_text, remove_duplicate_words, ngram_phraser

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
       }


#manually download spacy model to disk for use in future 
cache_dir= cfg['cache_dir']
spacy_model_path=cfg['spacy_model_path']
spell_correct_model_url = cfg['spell_correct_model_url']

incols = cfg['incols']
text_cols = cfg['text_cols']
n_samp = cfg['n_samp']

stop_words_add = cfg['stop_words_add']
stop_grams_add = cfg['stop_grams_add']





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
nlp.add_pipe('textrank')


#stemmer 
stemmer = PorterStemmer()


#from nltk.corpus import stopwords # For removing stopwords
stopwords = nltk.corpus.stopwords.words('english')
stopwords= set([str(s_t_r).lower() for s_t_r in list(stopwords) + cfg['stop_words_add'] if str(s_t_r).lower() not in ['up','down']])
stopgrams = cfg['stop_grams_add']

#download spacy model 
if not os.path.exists(Path(cache_dir,spacy_model_path).as_posix()):
    spacy.cli.download(spacy_model_path)

#load spacy model 
nlp = spacy.load(spacy_model_path)
nlp.to_disk(os.path.join(cache_dir,spacy_model_path))
nlp = spacy.load(os.path.join(cache_dir,spacy_model_path))
nlp.add_pipe('textrank')




