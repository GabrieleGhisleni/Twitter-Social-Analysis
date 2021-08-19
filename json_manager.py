from typing import List
import os, json


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

    def check_initial_id(self) -> set:
        storico, check_set = self.load(), set()
        for i in storico:
            check_set.add(i['id_str'])
        print(f"Unique ID founded: {len(check_set)}")
        return check_set

    def check_id(self, tweet_id: dict, check: set) -> bool:
        if tweet_id not in check:
            check.add(tweet_id)
            return True