import json
import pandas as pd
from typing import List
from nltk.corpus import stopwords
from nltk.probability import FreqDist
from nltk.stem.snowball import ItalianStemmer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer


class NltkTextProcessing:
    def __init__(self):
        self.stemmer = ItalianStemmer(ignore_stopwords=True)
        self.stopwords = set(stopwords.words("italian"))
        self.increase_stopwords()
        self.vectorizer = TfidfVectorizer(stop_words=self.stopwords)

    def preprocess_text(self, text) -> List:
        tokenized, res = word_tokenize(text=text, language='it'), list()
        for token in tokenized:
            if token not in self.stopwords and not token.isdigit() and len(token) > 2:
                if token == 'vaccini' or token == 'vaccinato' or token == 'vaccinati': token = 'vaccino'
                res.append(token)
        #token = [self.stemmer.stem(word) for word in token]
        return res

    def add_processed_column(self, df: pd.DataFrame, save: bool = False) -> pd.DataFrame:
        df['tweet_text'] = df['tweet_text'].apply(self.preprocess_text)
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
                    'invece'}
        self.stopwords = self.stopwords.union(stopwords_)

    def vectorized_dataframe(self, df: pd.DataFrame):
        return self.vectorizer.fit_transform(df['tweet_text'])


if __name__ == '__main__':
    with open('twitter.json', 'r') as file:
        tweet = pd.DataFrame(json.load(file))
    nlp = NltkTextProcessing()
    tweet_df = nlp.add_processed_column(tweet)
    nlp.frequency_dist(tweet_df)
