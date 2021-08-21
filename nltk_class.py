import json
from nltk.corpus import stopwords
from nltk.probability import FreqDist
from nltk.stem.snowball import ItalianStemmer
from nltk.tokenize import word_tokenize
import pandas as pd
from typing import List


class NltkTextProcessing:
    def __init__(self):
        self.stopwords = set(stopwords.words("italian"))
        self.increase_stopwords()
        self.stemmer = ItalianStemmer(ignore_stopwords=True)
        self.fdist = FreqDist()

    def preprocess_text(self, text) -> List:
        tokenized = word_tokenize(text=text, language='it')
        token = [token for token in tokenized if token not in self.stopwords and not token.isnumeric()]
        return token
        #return [self.stemmer.stem(word) for word in tokenized if word not in self.stopwords]

    def frequency_dist(self, df: pd.DataFrame):
        for text in df['tweet_text']:
            for word in self.preprocess_text(text):
                self.fdist[word] += 1
        return self.fdist

    def increase_stopwords(self):
        stopwords = {'ce', 'fa', 'tanto', 'comunque','ecco','sempre','perche','va'}
        self.stopwords = self.stopwords.union(stopwords)


if __name__ == '__main__':
    with open('twitters.json', 'r') as file:
        tweet = json.load(file)
    df = pd.DataFrame(tweet)
