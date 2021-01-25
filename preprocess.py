#!/usr/bin/env python3

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import string


import nltk
from nltk.corpus import wordnet
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, SnowballStemmer, WordNetLemmatizer


class PreprocessNLP():
    def __init__(self, data_frame, X):
        self.data_frame = data_frame
        self.X = X
        self.stemmer = SnowballStemmer('english')
        self.lemm = WordNetLemmatizer()
        self._stop_words = set(stopwords.words('english'))

    # source: https://stackoverflow.com/questions/31016540/
        # lemmatize-plural-nouns-using-nltk-and-wordnet/31212056#31212056
    def get_wordnet_pos(self, treebank_tag):
        if treebank_tag.startswith('J'):
            return wordnet.ADJ
        elif treebank_tag.startswith('V'):
            return wordnet.VERB
        elif treebank_tag.startswith('N'):
            return wordnet.NOUN
        elif treebank_tag.startswith('R'):
            return wordnet.ADV
        else:
            return wordnet.NOUN

    def lower(self):
        self.data_frame[self.X] = self.data_frame[self.X].apply(lambda x: x.lower())
        print("lower finished")

    def punctuation(self):
        self.data_frame[self.X] =self.data_frame[self.X].apply(lambda x: ''.join([ i for i in x
                                                  if i not in string.punctuation] ) )

        print("punctuation done")

    def stop_words(self):
        self.data_frame[self.X] = self.data_frame[self.X].apply(lambda x: ' '.join( \
                        [item for item in x.split() if item not in self._stop_words]))
        print("stop words done")

    def stem(self):
        self.data_frame[self.X] =\
                self.data_frame[self.X].apply(lambda x: [self.stemmer.stem(y) for y in x])
        print("stemming done")

    def lemmatize(self):
        self.data_frame[self.X] =\
                self.data_frame[self.X].apply(lambda x: [self.lemm.lemmatize(y, \
                            self.get_wordnet_pos(nltk.pos_tag(y)[0][1]) ) for y in x])
        print("lemmatize done")

    def tokenize(self):
        self.data_frame[self.X] =\
                self.data_frame[self.X].apply(nltk.word_tokenize)
        print("tokenize done")

    def join(self):
        self.data_frame[self.X] =\
                self.data_frame[self.X].apply(lambda x: ' '.join(x))
        print("join finished")
