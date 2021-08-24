from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.snowball import ItalianStemmer
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
from typing import List
import seaborn as sns
import pandas as pd
import numpy as np
import json, re


class NltkTextProcessing:
    def __init__(self):
        self.stemmer = ItalianStemmer(ignore_stopwords=True)
        self.stopwords = set(stopwords.words("italian"))
        self.increase_stopwords()

    def preprocess_text(self, text, stem: bool = False) -> List:
        tokenized, res = word_tokenize(text=text, language='it'), list()
        for token in tokenized:
            if token not in self.stopwords and not token.isdigit() and len(token) > 2 and not token[0].isdigit():
                if not token.startswith('ah') and 'juve' not in token and 'inter' not in token and not token.startswith('tweet'):
                    if token == 'vaccini' or token == 'vaccinato' or token == 'vaccinati': token = 'vaccino'
                    if token == 'falsi': token = 'falso'
                    if token == 'grnpass': token = 'greenpass'
                    res.append(token)
        if stem: res = [self.stemmer.stem(word) for word in res]
        return res

    def process_df_text_column(self, df: pd.DataFrame, stem: bool, save: bool = False) -> pd.DataFrame:
        df['tweet_text'] = df['tweet_text'].apply(self.preprocess_text, stem=stem)
        if save: df.to_csv('tweets.csv')
        return df

    def unique_hashtags(self, df: pd.DataFrame):
        wl=set()
        for tweets in np.unique(df['hashtags'].dropna()):
            for hashs in tweets:
                if len(hashs) > 3: wl.add(hashs)
        return wl

    def remove_hashtag_from_text(self, df: pd.DataFrame):
        hashtag_set = self.unique_hashtags(df)
        def remove(s):
            for word in s.split(' '):
                if word in hashtag_set: s = s.replace(word, '')
            for iel in range(1,5): s = s.replace('  ' * iel, '')
            return s
        df['tweet_text'] = df['tweet_text'].apply(remove)
        return df

    def increase_stopwords(self) -> None:
        stopwords_={'ce', 'fa', 'tanto', 'comunque', 'ecco', 'sempre', 'perche', 'va', 'co', 't', 'vuole',
                    'dopo', 'https', 'poi', 'vedere', 'te', 'quest', 'do', 'no', 'pero', 'piu', 'quando',
                    'adesso', 'ogni', 'so', 'essere', 'tutta', 'senza', 'fatto', 'essere', 'oggi', 'cazzi',
                    'altri', 'ah', 'quindi', 'gran', 'solo', 'ora', 'grazie', 'cosa', 'gia', 'me', '-',
                    'altro', 'nome', 'prima', 'anno', 'pure', 'qui', 'fate', 'sara', 'proprio', 'sa', 'de', 'fare',
                    'nuova', 'molto', 'mette', 'dire', 'tali', 'puo', 'uso', 'cioe', 'alta', 'far', 'qualsiasi',
                    'cosi', 'chiamano', 'capito', 'cazzo', 'raga', 'mai', 'avere', 'andare', 'invece', 'mesi', 'ancora',
                    'invece', 'a0xlp74lne', 'a4otny4rhy', 'aaa', 'aacmgmzanzio', 'aanzibma3f', 'ajgsd0w7mx', 'parli',
                    'vai','allegri', 'qusta', 'qusto', 'anch', 'prch', 'com', 'snza', 'dir', 'qlli', 'no'}
        self.stopwords = self.stopwords.union(stopwords_)

    @staticmethod
    def process_df_hash_column(df: pd.DataFrame, save: bool = False) -> pd.DataFrame:
        def hash_process(hashes):
            if hashes: return [hashs.lower() for hashs in hashes]
        df['hashtags'] = df['hashtags'].apply(hash_process)
        if save: df.to_csv('tweets.csv')
        return df

    @staticmethod
    def keep_unique(df: pd.DataFrame, save: bool = False) -> pd.DataFrame:
        check, res = set(), list()
        for (idx, row) in df.iterrows():
            if tuple(row.tweet_text) in check:
                pass
            else:
                check.add(tuple(row.tweet_text))
                res.append(row)
        unique_df = pd.DataFrame(res)
        if save: unique_df.to_csv('unique.csv')
        return unique_df

    @staticmethod
    def frequency_dist(df: pd.DataFrame, obj: str = 'tweet') -> FreqDist:
        res = FreqDist()
        bag = df['tweet_text'] if obj == 'tweet' else df['hashtags']
        for text in bag:
            if text:
                for word in text: res[word] += 1
        return res

    @staticmethod
    def extract_external_url(df: pd.DataFrame) -> dict:
        res = dict()
        for url in df['external_url']:
            if url and 'twitter' not in url:
                slash_idx = url.find('/') + 2
                idx = url.find(".")
                if 'www' in url:
                    sdx = url.find('.', idx + 1)
                    web = url[slash_idx:sdx + 4]
                else:
                    second_slash = url.find('/', slash_idx) + 1
                    web = url[slash_idx:second_slash]
                if web != '':
                    if web in res:
                        res[web] += 1
                    else: res[web] = 1
        return res

    @staticmethod
    def prepare_text_to_vectorize(df: pd.DataFrame, afil: bool = False, obj = 'tweet') -> list:
        def filter_(txt):
            extra_filter={'nogreenpass', 'iononmivaccino'}
            tmp = ' '.join(txt)
            for word in extra_filter:
                tmp = tmp.replace(word, '')
                tmp = tmp.replace('  ', ' ')
            return tmp
        if obj == 'tweet':
            if afil: return df['tweet_text'].apply(filter_).values.tolist()
            else: return df['tweet_text'].apply(lambda x: ' '.join(x)).values.tolist()
        else: return df['hashtags'].apply(lambda x: ' '.join(x)).values.tolist()


def count_barplot(count: dict, thresold: int = 20) -> None:
    fig = plt.figure(figsize=(20,20))
    sns.set_style('white')
    word, freq = [], []
    for key in count:
        if count[key] > thresold:
            word.append(key)
            freq.append(count[key])
    df = pd.DataFrame(freq, word).reset_index().\
        rename(columns={'index': 'words', 0: 'freq'}).sort_values(by='freq', ascending=False)
    sns.barplot(y='words', x="freq", data=df)
    plt.title('Most Frequent External URL\n', fontsize=25)
    plt.xlabel('')
    plt.ylabel('')
    plt.show()

def update_parameter() -> None:
    large, med = 22, 16
    sns.set_style('white')
    params={'axes.titlesize': large,'legend.fontsize': med,
            'axes.labelsize': med, 'xtick.labelsize': large,
            'ytick.labelsize': large, 'figure.titlesize': large}
    plt.rcParams.update(params)
    sns.set_style('whitegrid')

