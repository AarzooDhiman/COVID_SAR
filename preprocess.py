import pandas as pd
import numpy as np
pd.set_option('display.max_colwidth', None)
from math import sqrt 
import matplotlib.pyplot as plt 
import pickle5 as pickle
from multiprocessing import Pool
from nltk import pos_tag, pos_tag_sents
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
tqdm.pandas()
from nltk.tokenize import TweetTokenizer
import spacy
from spacy.matcher import Matcher
nlp = spacy.load('en_core_web_sm')
from nltk.tokenize import TweetTokenizer
import nltk
from nltk import wordpunct_tokenize
from nltk.corpus import words
words = set(nltk.corpus.words.words())
import contractions
from functools import partial
import datetime
from datetime import datetime
import itertools
import glob
import os
import re
import string 
import tempfile
import concurrent.futures
import swifter
punctuations = list(string.punctuation)
import sys
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix

from scipy import sparse

from sklearn.metrics import classification_report, recall_score, accuracy_score, precision_score, roc_curve, plot_roc_curve
from sklearn.model_selection import KFold 

#from ludwig.api import LudwigModel, kfold_cross_validate
#import logging

import tweepy
import json
import time
from sklearn.linear_model import LogisticRegression

from textblob.classifiers import NaiveBayesClassifier
from sklearn.model_selection import train_test_split
from textblob import TextBlob, WordList
from functools import partial
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize 
from pandarallel import pandarallel
import csv
from sklearn.feature_extraction.text import TfidfVectorizer
import emoji
import bz2

import _pickle as cPickle



import warnings
warnings.filterwarnings("ignore")

import shutil

USER_ID = "USER_ID"
USER_ID_STR = "USER_ID_STR"
TWEET_ID = "TWEET_ID"
TWEET_ID_STR = "TWEET_ID_STR"
LOCATION = "LOCATION"
TIMESTAMP = "TIMESTAMP"
TWEET_TEXT = "TWEET_TEXT" 
PROCESSED = "TWEET_TEXT_PROCESSED"
MATCHES = "MATCHES"
EXL_MATCHES = "EXCLUSION_MATCHES"
LABEL = "label"

columns = [TWEET_ID, TWEET_ID_STR, USER_ID, USER_ID_STR, LOCATION, TIMESTAMP, TWEET_TEXT]




def read_pickle1(pth):
    with open(pth, "rb") as fh:
        data = pickle.load(fh)
    return (data)





covid_restrictive_regex = [ 
    "(covid)", "(\\bc19\\b)"
    "((corona).*(virus))|((virus).*(corona))",
    "((sars).*(cov))",
    "(anosmia)", "(anosmia)", 
    "((smell).*(loss))|((loss).*(smell))",
    "((taste).*(loss))|((loss).*(taste))",
    "(headache)", "(sick)", "(migraine)", "(nausea)", "((tested).*(positive))",
    "(cold)", "(cough)", "((sore).*(throat))"
]

covid_restrictive_regex = [ 
    "(covid)", 
    "((corona).*(virus))|((virus).*(corona))",
]

restrictive_pattern = "|".join(covid_restrictive_regex)


def contract(tweet):
    new_tweet = []
    for t in tweet.split():
        try:
            new_tweet.append(contractions.fix(t))
        except:
            print (t)
            new_tweet.append(t)
    return (" ".join(new_tweet))

def regex_clean(df):
    temp = df.copy()
    #Remove contractions
    print (temp.head())
    temp[PROCESSED] = temp[TWEET_TEXT].apply(lambda x: contract(x))
    #replace emojis with texts
    temp[PROCESSED] = temp[PROCESSED].apply(lambda x: emoji.demojize(x))
    #Remove URLs
    temp[PROCESSED] = temp[PROCESSED].apply(lambda x: re.sub(r'https?:\/\/.*[\r\n]*', '', x))
    #Remove incorrectly formatted &
    temp[PROCESSED] = temp[PROCESSED].apply(lambda x: re.sub(r'&amp;', '', x))
    #Remove retweets and tags at the start. This also removes tweets which have no words but only tags and mentions
    temp[PROCESSED] = temp[PROCESSED].apply(lambda x: re.sub(r'^((RT|rt) )?(((@)\w*) ?:? ?)*', '', x))
    #Replace ... with .
    temp[PROCESSED] = temp[PROCESSED].apply(lambda x: re.sub('(\.{3})', '.', x))
    # Remove ..., @, and # characters
    temp[PROCESSED] = temp[PROCESSED].apply(lambda x: re.sub('(@|…|\\n)', '', x))
    # Convert the tweets to lowercase
    temp[PROCESSED] = temp[PROCESSED].apply(lambda x: x.lower())
    return temp


my_stopwords = set(stopwords.words('english')).difference(["i"])



#drop Tweets not in English
def english_or_Not(tweet_text, words):
    tokens = wordpunct_tokenize(tweet_text)
    punctuation_count = len([w for w in tokens if w in punctuations])
    if len(tokens) - punctuation_count < 3:
        return True
    eng_count = len([w for w in tokens if w.lower() in words or not w.isalpha()])
    if eng_count/len(tokens) < 0.70:
        return True
    return False

def clean_tweets(df_in,covid=False, remove_duplicates=True):
    df = df_in
    #print(f"{df.shape[0]:,}")
    
    df = regex_clean(df)
    
    df['HASHTAGS'] = df.TWEET_TEXT.apply(lambda x: [x for x in x.split(" ") if x.startswith("#")])
    
    texts = df[PROCESSED].tolist()
    tt = TweetTokenizer()
    '''tagged_texts = pos_tag_sents(tt.tokenize(texts))
    tagged_texts2=[]
    for tag in tagged_texts:
        temp = []
        for x in tag:
            if len(x)!=0 and x[1].isalpha():
                temp.append(x)
                
        tagged_texts2.append(temp)'''
        
    tagged_texts2=[]
    for t in texts:
        word_list = tt.tokenize(t)
        tag = nltk.pos_tag(word_list)
        
        temp = []
        for x in tag:
            if len(x)!=0 and x[1].isalpha():
                temp.append(x[0]+"_"+x[1])
                
        tagged_texts2.append(" ".join(temp))
            
    #tagged_texts2 = [x for x in tagged_texts if x[1]!='.']
    df['POS'] = tagged_texts2
    
    if remove_duplicates:
        df["DUPLICATE"] = df.duplicated(subset=PROCESSED, keep=False)
        #print((df["DUPLICATE"] == True).sum())
        df = df[(df["DUPLICATE"] == False)]

    df["NON_ENGLISH"] = df[PROCESSED].apply(lambda x: english_or_Not(x, words))
    #print((df["NON_ENGLISH"] == True).sum())

    df["IF"] = df[PROCESSED].apply(lambda x: False if re.match('if i get', x) is None else True)
    #print((df["IF"] == True).sum())

    df = df[(df["NON_ENGLISH"] == False) ]#& (df["IF"] == False)
    
    #print ("=====non english")
    #print (df.shape)
    df["COVID"] = df[PROCESSED].apply(lambda x: False if re.match(fr".*({restrictive_pattern}).*", x, re.IGNORECASE) is None else True)
    #df = df[df["COVID"] == covid]
    
    #print ("=====non covid")
    #print (df.shape)
    
    df['TWEET_TEXT_PROCESSED_POSTAGGED'] = df[[PROCESSED, 'POS']].T.agg(' '.join)
    
    #print(f"{df.shape[0]:,}")
    return df


if __name__ == "__main__":
    df1 = read_pickle1('/home/adhiman/SAR-z/to_label_data/code/final_labels/labeled_SET1.pkl')
    df2 = read_pickle1('/disks/sdb/adhiman/SAR-z/labelbox itr2/with_consensus/itr2.pkl')
    df2 = df2[list(df1.columns.values)]
    df2 = df2.replace('NO', 0)
    df2 = df2.replace('YES', 1)
    df = pd.concat([df1,df2])


    df = clean_tweets(df, remove_duplicates=True)
    #print (df.head())
    df.to_pickle('/home/adhiman/SAR-z/labeled data/itr1_2.pkl')
    df.to_csv('/home/adhiman/SAR-z/labeled data/itr1_2.csv', index = False)




        
        
