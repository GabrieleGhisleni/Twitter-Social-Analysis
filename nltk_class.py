import json
import pandas as pd
from typing import List
from nltk.corpus import stopwords
from nltk.probability import FreqDist
from nltk.stem.snowball import ItalianStemmer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
import seaborn as sns


class NltkTextProcessing:
    def __init__(self):
        self.stemmer = ItalianStemmer(ignore_stopwords=True)
        self.stopwords = set(stopwords.words("italian"))
        self.increase_stopwords()

    def preprocess_text(self, text) -> List:
        tokenized, res = word_tokenize(text=text, language='it'), list()
        for token in tokenized:
            if token not in self.stopwords and not token.isdigit() and len(token) > 2 and not token[0].isdigit():
                if not token.startswith('ah'):
                    if token == 'vaccini' or token == 'vaccinato' or token == 'vaccinati': token = 'vaccino'
                    if token == 'falsi': token = 'falso'
                    res.append(token)
        # res = [self.stemmer.stem(word) for word in res]
        return res

    def process_df_text_column(self, df: pd.DataFrame, save: bool = False) -> pd.DataFrame:
        df['tweet_text'] = df['tweet_text'].apply(self.preprocess_text)
        if save: df.to_csv('tweets.csv')
        return df

    def process_df_hash_column(self, df: pd.DataFrame, save: bool = False) -> pd.DataFrame:
        def hash_process(hashes):
            if hashes: return [hashs.lower() for hashs in hashes]
        df['hashtags'] = df['hashtags'].apply(hash_process)
        if save: df.to_csv('tweets.csv')
        return df

    def keep_unique(self, df: pd.DataFrame, save: bool = False) -> pd.DataFrame:
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

    def frequency_dist(self, df: pd.DataFrame, obj: str = 'tweet') -> FreqDist:
        res = FreqDist()
        bag = df['tweet_text'] if obj == 'tweet' else df['hashtags']
        for text in bag:
            if text:
                for word in text: res[word] += 1
        return res

    def increase_stopwords(self) -> None:
        stopwords_={'ce', 'fa', 'tanto', 'comunque', 'ecco', 'sempre', 'perche', 'va', 'co', 't', 'vuole',
                    'dopo', 'https', 'poi', 'vedere', 'te', 'quest', 'do', 'no', 'pero', 'piu', 'quando',
                    'adesso', 'ogni', 'so', 'essere', 'tutta', 'senza', 'fatto', 'essere', 'oggi', 'cazzi',
                    'altri', 'ah', 'quindi', 'gran', 'solo', 'ora', 'grazie', 'cosa', 'gia', 'me', '-',
                    'altro', 'nome', 'prima', 'anno', 'pure', 'qui', 'fate', 'sara', 'proprio', 'sa', 'de', 'fare',
                    'nuova', 'molto', 'mette', 'dire', 'tali', 'puo', 'uso', 'cioe', 'alta', 'far', 'qualsiasi',
                    'cosi', 'chiamano', 'capito', 'cazzo', 'raga', 'mai', 'avere', 'andare', 'invece', 'mesi', 'ancora',
                    'invece', 'a0xlp74lne', 'a4otny4rhy', 'aaa', 'aacmgmzanzio', 'aanzibma3f', 'ajgsd0w7mx'}
        self.stopwords = self.stopwords.union(stopwords_)

    def extract_external_url(self, df: pd.DataFrame) -> dict:
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


def count_barplot(count, thresold = 20):
    fig = plt.figure(figsize=(20,20))
    word, freq = [], []
    for key in count:
        if count[key] > thresold:
            word.append(key)
            freq.append(count[key])
    df = pd.DataFrame(freq, word).reset_index().\
        rename(columns={'index': 'words', 0: 'freq'}).sort_values(by='freq', ascending=False)
    sns.barplot(y='words', x="freq", data=df)
    plt.show()

def update_parameter():
    large, med = 22, 16
    params={'axes.titlesize': large,'legend.fontsize': med,
            'axes.labelsize': large, 'xtick.labelsize': large,
            'ytick.labelsize': large, 'figure.titlesize': large}
    plt.rcParams.update(params)
    sns.set_style('whitegrid')

if __name__ == '__main__':
    with open('twitter.json', 'r') as file:
        tweet = pd.DataFrame(json.load(file))
    nlp = NltkTextProcessing()
    from pprint import pprint
    count_barplot(    nlp.extract_external_url(tweet))


