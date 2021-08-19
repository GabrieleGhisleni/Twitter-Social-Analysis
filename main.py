import tweepy
import os, json, time
from typing import List

class Tweet:
    def __init__(self):
        raise NotImplementedError


class JsonManager:
    def __init__(self, path: str = 'twitter.json'):
        if not os.path.exists(path):
            with open(path, 'w') as f:
                json.dump([], f)
        self.path = path

    def load(self):
        with open(self.path) as file:
            return json.load(file)

    def save(self, news: List[dict]):
        storico = self.load()
        storico.append(news)
        with open(self.path, 'w') as f:
            json.dump(storico, f)


class TwitterAPI:
    def __init__(self):
        self.auth = tweepy.OAuthHandler(os.getenv("CSS_KEY"), os.getenv("CSS_SECRET_KEY"))
        self.auth.set_access_token(os.getenv("CSS_TOKEN"), os.getenv("CSS_SECRET_TOKEN"))
        self.api = tweepy.API(self.auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True)

    def tweet_stream(self, q: List, s: str):
        query = f"{q} -filter:retweets" #avoid retweets
        jsonDB = JsonManager()
        stream = tweepy.Cursor(self.api.search,
                                q=query, since = s,
                                result_type="mixed",
                                lang="it", tweet_mode="extended").items()

        for tweet in self.limit_handled(stream):
            jsonDB.save(tweet._json)
            time.sleep(1)

    def limit_handled(self, cursor):
        while True:
            try: yield cursor.next()
            except tweepy.RateLimitError: time.sleep(15 * 60)

if __name__ == '__main__':
    hashtags = ['#iononmivaccino OR #dittaturasanitaria OR #nogreenpass OR #vienegiututto']
    since = "2021-8-12" # no more than one week
    TwitterAPI().tweet_stream(q=hashtags, s=since)

