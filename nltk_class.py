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
        self.fdist = FreqDist()
        self.vectorizer = TfidfVectorizer(stop_words=self.stopwords)

    def preprocess_text(self, text) -> List:
        tokenized = word_tokenize(text=text, language='it')
        token = [token for token in tokenized if token not in self.stopwords and not token.isdigit()]
        #token = [self.stemmer.stem(word) for word in token]
        return token

    def frequency_dist(self, df: pd.DataFrame) -> FreqDist:
        for text in df['tweet_text']:
            for word in self.preprocess_text(text):
                self.fdist[word] += 1
        return self.fdist

    def increase_stopwords(self) -> None:
        stopwords = {'ce', 'fa', 'tanto', 'comunque','ecco','sempre','perche','va'}
        self.stopwords = self.stopwords.union(stopwords)

    def vectorized_dataframe(self, df: pd.DataFrame):
        return self.vectorizer.fit_transform(df['tweet_text'])

    def add_processed_column(self, df: pd.DataFrame) -> None:
        df['tweet_text'] = df['tweet_text'].apply(self.preprocess_text)
        df.to_csv('tweets.csv')


if __name__ == '__main__':
    with open('twitters.json', 'r') as file:
        tweet = pd.DataFrame(json.load(file))
    NltkTextProcessing().add_processed_column(tweet)
