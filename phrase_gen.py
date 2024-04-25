import pandas as pd
import polars as pl
from spello.model import SpellCorrectionModel
import io, re, string, spello,requests , zipfile, os, nltk, spacy , pytextrank
from pathlib import Path
from bs4 import BeautifulSoup # For removing HTML
import contractions # For expanding contractions
from unidecode import unidecode # For handling accented words
from fuzzywuzzy import process
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from functools import reduce
stemmer = PorterStemmer()

cfg = {'data_path':'/home/zjc1002/Mounts/data/cfpb/cfpb_complaints.csv'
       , 'cache_dir':"/home/zjc1002/Mounts/temp/"
       , 'incols':['Date received', 'Product','Consumer complaint narrative','Company public response']
       , 'text_cols':['Consumer complaint narrative','Company public response']
       , 'spell_correct_model_url': "https://haptik-website-images.haptik.ai/spello_models/en_large.pkl.zip"
       , 'spacy_model_path' : "en_core_web_lg"
       , 'n_samp': 100 
       }


#manually download spacy model to disk for use in future 
cache_dir= cfg['cache_dir']
spacy_model_path=cfg['spacy_model_path']
spell_correct_model_url = cfg['spell_correct_model_url']
incols = cfg['incols']
text_cols = cfg['text_cols']
n_samp = cfg['n_samp']


#download spacy model 
if not os.path.exists(Path(cache_dir,spacy_model_path).as_posix()):
    spacy.cli.download(spacy_model_path)

#nltk.download('stopwords')
#from nltk.corpus import stopwords # For removing stopwords
stopwords = nltk.corpus.stopwords.words('english')
stopwords= set([str(s_t_r).lower() for s_t_r in list(stopwords) + ['xxxxxxxx','xxxx','xxx','xx'] if str(s_t_r).lower() not in ['up','down']])

#load spacy model 
nlp = spacy.load(spacy_model_path)
nlp.to_disk(os.path.join(cache_dir,spacy_model_path))
nlp = spacy.load(os.path.join(cache_dir,spacy_model_path))
nlp.add_pipe('textrank')





### SPELLING CORRECTION OPTIONAL 
##download the spell correction model (just copy in paste if behind firewall) 
## Check if the cache directory exists
#if os.path.exists(cache_dir):
#    print("The spell check model folder exists.")
#else:
#    # Download and unzip the spell correction model
#    download_and_unzip(spell_correct_model_url, cache_dir)
## Spell correction model, we don't use it
#sp = SpellCorrectionModel(language='en')
#sp.load(Path(cache_dir,spell_correct_model_url.split('/')[-1].replace('.zip','')))
#sp.config.min_length_for_spellcorrection = 4


r