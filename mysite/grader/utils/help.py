import pandas as pd
import lime
from lime import lime_text
import lime.lime_tabular
from sklearn.pipeline import make_pipeline
import nltk
nltk.download('stopwords')
nltk.download('words')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.corpus import words
from nltk.corpus import wordnet
import re
import numpy as np
import imgkit
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn import preprocessing, utils, metrics, ensemble
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR, SVC
import string
import re, collections
from collections import defaultdict
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, cross_val_score, KFold
import pickle 
from sklearn.metrics import cohen_kappa_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from io import BytesIO
import base64
import math

# Removes numeric sequences, @ etc.
def clean_text(essay):
    essay = str(essay)
    result = re.sub(r'http[^\s]*', '', essay)
    result = re.sub('[0-9]+', '', result).lower()
    result = re.sub('@[a-z0-9]+', '', result)
    return re.sub('[%s]*' % string.punctuation, '', result)

# Removes all non-ascii characters from essays
def de_emojify(essay):
    return essay.encode('ascii', 'ignore').decode('ascii')

def count_bigrams(essay):
    sentences = nltk.sent_tokenize(essay)
    bigram_count = 0
    for sentence in sentences :
      bigrams = [grams for grams in nltk.ngrams(sentence.split(), 2)]
      bigram_count += len([(item, bigrams.count(item)) for item in sorted(set(bigrams))])
    return bigram_count

def get_wordlist(sentence):    
    clean_sentence = re.sub("[^A-Z0-9a-z]"," ", sentence)
    wordlist = nltk.word_tokenize(clean_sentence)    
    return wordlist

def tokenize(essay):
    stripped_essay = essay.strip()
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    raw_sentences = tokenizer.tokenize(stripped_essay)    
    tokenized_sentences = []
    for raw_sentence in raw_sentences:
        if len(raw_sentence) > 0:
            tokenized_sentences.append(get_wordlist(raw_sentence))    
    return tokenized_sentences

def avg_word_len(essay):    
    clean_essay = re.sub(r'\W', ' ', essay)
    words = nltk.word_tokenize(clean_essay)    
    return sum(len(word) for word in words) / len(words)

def word_count(essay):    
    clean_essay = re.sub(r'\W', ' ', essay)
    words = nltk.word_tokenize(clean_essay)    
    return len(words)


def char_count(essay):    
    clean_essay = re.sub(r'\s', '', str(essay).lower())    
    return len(clean_essay)

def sent_count(essay):    
    sentences = nltk.sent_tokenize(essay)    
    return len(sentences)

def count_pos(essay):
    tokenized_sentences = tokenize(essay)
    noun_count = 0
    adj_count = 0
    verb_count = 0
    adv_count = 0
    for sentence in tokenized_sentences:
        tagged_tokens = nltk.pos_tag(sentence)
        for token_tuple in tagged_tokens:
            pos_tag = token_tuple[1]
            if pos_tag.startswith('N'): 
                noun_count += 1
            elif pos_tag.startswith('J'):
                adj_count += 1
            elif pos_tag.startswith('V'):
                verb_count += 1
            elif pos_tag.startswith('R'):
                adv_count += 1
    return noun_count, adj_count, verb_count, adv_count

def extract_features(data):
    features = data.copy()
    features['sent_count'] = features['essay'].apply(sent_count)
    features['essay'] = features['essay'].apply(clean_text)
    features['essay'] = features['essay'].apply(de_emojify)
    features['word_count'] = features['essay'].apply(word_count)
    features['bigram_count'] = features['essay'].apply(count_bigrams)
    features['noun_count'], features['adj_count'], features['verb_count'], features['adv_count'] = zip(*features['essay'].map(count_pos))
    return features


def f_importances(coef, names):
    imp = coef
    imp,names = zip(*sorted(zip(imp,names)))
    plt.barh(range(len(names)), imp, align='center')
    plt.yticks(range(len(names)), names)
    plt.tight_layout()
    plt.savefig('grader/static/features.png')


