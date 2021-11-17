from json_manager import JsonManager
from tweet_class import Tweet
from keys import get_api_aws
from typing import List
import traceback
import argparse
import datetime
import tweepy
import json
import time
import os


class TwitterStreamAPI(tweepy.StreamListener):
    def __init__(self, path_json = 'twitter.json', first_clean: bool = False):
        super().__init__()
        self.json = JsonManager(path=path_json)
        self.tweets_id_collected = self.json.check_initial_id()
        self.first_clean = first_clean
        self.file_name = path_json

    def on_status(self, status):
        tweet = Tweet.from_api_to_class(status, first_clean=self.first_clean)
        if self.json.check_id(tweet.id, self.tweets_id_collected):
            self.json.save(tweet.to_repr())
            print(f'Tweets Collected: {len(self.tweets_id_collected)}, file name = {self.file_name}')

    def on_limit(self, status):
        print("Rate Limit Exceeded, Sleep for 15 Mins")
        time.sleep(15 * 60)
        return True

    def on_error(self, status_code):
            return True

    @staticmethod
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
    arg_parse = argparse.ArgumentParser(description="Twitter Streamer!")
    arg_parse.add_argument('-c', '--clean', action='store_true')
    arg_parse.add_argument('-p', '--path', default='twitter.json')
    args = arg_parse.parse_args()

    q = ['iononmivaccino','nogreenpass','dittaturasanitaria','bigpharma','nocavie','nessunacorrelazione',
         'meluzzi', 'Montagnier', 'obbligovaccinale', 'governocriminale', 'nogreenpassobbligatorio',
         'terzadose', 'PassSanitaire','somministrazioneDiCortesia']

    listener = TwitterStreamAPI(path_json=args.path, first_clean=args.clean)
    myStream = tweepy.Stream(auth=get_api_aws().auth, listener=listener)
    myStream.filter(track=q, languages=['it'])

if __name__ == '__main__':
    main()

