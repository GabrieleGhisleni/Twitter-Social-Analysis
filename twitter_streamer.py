import datetime
import tweepy
import os, json, time
from typing import List
from json_manager import JsonManager
from tweet_class import Tweet
from keys import get_api_aws
import traceback
from pprint import pprint


class TwitterStreamAPI(tweepy.StreamListener):
    def __init__(self):
        super().__init__()
        self.json = JsonManager()
        self.tweets_id_collected = self.json.check_initial_id()

    def on_status(self, status):
        tweet = Tweet.from_api_to_class(status)
        if self.json.check_id(tweet.id, self.tweets_id_collected):
            self.json.save(tweet.to_repr())
            print(f'Tweets Collected: {len(self.tweets_id_collected)}')

    def on_limit(self, status):
        print("Rate Limit Exceeded, Sleep for 15 Mins")
        time.sleep(15 * 60)
        return True

    def on_error(self, status_code):
            return True


def get_api() -> tweepy.API:
    auth = tweepy.OAuthHandler(os.getenv("CSS_KEY"), os.getenv("CSS_SECRET_KEY"))
    auth.set_access_token(os.getenv("CSS_TOKEN"), os.getenv("CSS_SECRET_TOKEN"))
    return tweepy.API(auth,
                      wait_on_rate_limit=True,
                      wait_on_rate_limit_notify=True)

def log_loop(fun):
    while True:
        try:
            print(f'Start looping at {datetime.datetime.now()}')
            fun()
        except Exception as e:
            now = datetime.datetime.now().strftime('%Y-%m-%d | %H:%M:%S')
            print(f'Error: {e} at {now}')
            with open("exceptions.log", "a") as logfile:
                logfile.write(f"\n\n {now} \n")
                traceback.print_exc(file=logfile)
            time.sleep(10*60)

@log_loop
def main():
    q = ['iononmivaccino','nogreenpass','dittaturasanitaria','bigpharma','nocavie','nessunacorrelazione',
         'meluzzi', 'Montagnier', 'obbligovaccinale', 'governocriminale', 'nogreenpassobbligatorio',
         'terzadose', 'PassSanitaire','somministrazioneDiCortesia']
    myStream = tweepy.Stream(auth=get_api_aws().auth, listener=TwitterStreamAPI())
    myStream.filter(track=q, languages=['it'])


if __name__ == '__main__':
    main()

