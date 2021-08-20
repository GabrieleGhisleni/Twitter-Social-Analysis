from typing import List
import os
import json
from tweet_class import *
from tqdm import tqdm

class JsonManager:
    def __init__(self, path: str = 'twitter.json'):
        self.path = path

    def save(self, news: List[dict]) -> None:
        with open(file=self.path, mode='a') as storico:
            json.dump(news, storico)

    def load_unprocess(self) -> List:
        with open(file=self.path, mode='r') as file:
            return json.load(file)

    def check_initial_id(self) -> set:
        try: storico, check_set = self.load_unprocess(), set()
        except Exception: return set()
        for i in storico:
            check_set.add(i['id_str'])
        print(f"Unique ID founded: {len(check_set)}")
        return check_set

    def check_id(self, tweet_id: dict, check: set) -> bool:
        if tweet_id not in check:
            check.add(tweet_id)
            return True

    def initial_from_raw_to_processed(self, save: bool = True) -> None or List :
        raw_storico, res = self.load_unprocess(), list()
        for tweet in tqdm(raw_storico):
            obj = Tweet.from_dict_to_class(tweet)
            obj_d = obj.to_repr()
            res.append(obj_d)
        if save:
            with open(file=self.path, mode='w') as storico:
                json.dump(res, storico)
        else: return res


if __name__ == '__main__':
    JsonManager().initial_from_raw_to_processed()