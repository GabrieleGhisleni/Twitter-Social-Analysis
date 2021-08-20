from __future__ import annotations
import re
import requests
from PIL import Image
from pytesseract import pytesseract
import tweepy
import unicodedata

class Tweet:
    def __init__(self, created_at, id, tweet_text, is_reply, reply_count, retweet_count, hashtags, external_url,
                 author_followers, author_follow, author_loc, media_text):
        self.created_at = created_at
        self.id = id
        self.tweet_text = tweet_text
        self.is_reply = is_reply
        self.reply_count = reply_count
        self.retweet_count = retweet_count
        self.hashtags = hashtags
        self.external_url = external_url
        self.author_followers = author_followers
        self.author_follow = author_follow
        self.author_loc = author_loc
        self.media_text = media_text

    def __str__(self):
        print(self.id)

    def to_repr(self):
        return dict(created_at=self.created_at,
                    id=self.id,
                    tweet_text=self.tweet_text,
                    is_reply=self.is_reply,
                    reply_count=self.reply_count,
                    retweet_count=self.retweet_count,
                    hashtags=self.hashtags,
                    external_url=self.external_url,
                    author_followers=self.author_followers,
                    author_follow=self.author_follow,
                    author_loc=self.author_loc,
                    media_text=self.media_text)

    @staticmethod
    def from_api_to_class(status: tweepy.api.API) -> Tweet:
        is_a_reply, hashtags = False, None
        text = extended_tweet(status._json)
        media_text = check_and_extract_image_text(status)
        external_url = find_external_url(status._json)
        if status.in_reply_to_status_id or status.in_reply_to_user_id: is_a_reply = True
        if status.entities['hashtags']: hashtags = [i['text'] for i in status.entities['hashtags']]
        return Tweet( created_at=str(status.created_at),
                      id=str(status.id),
                      tweet_text=text,
                      is_reply=is_a_reply,
                      reply_count= status.reply_count,
                      retweet_count=status.retweet_count,
                      author_followers=status.author.followers_count,
                      author_follow=status.author.friends_count,
                      author_loc=status.author.location,
                      hashtags=hashtags,
                      external_url=external_url,
                      media_text=media_text)


def extract_text_from_image(url, timeout=5, path_to_tesseract=r"C:\Program Files\Tesseract-OCR\tesseract.exe") -> str or None:
    pytesseract.tesseract_cmd = path_to_tesseract
    raw_image = requests.get(url, stream=True).raw
    try:
        text = pytesseract.image_to_string(Image.open(raw_image), timeout=timeout)
        return text_first_clean(text[:-1])
    except RuntimeError:
        print(f'Exceeded timeout ({timeout}s)')
    except Exception as e:
        print(e, url)


def check_media(tweet: dict) -> str or None:
    original, retweet = False, False
    if 'media' in tweet['entities']:
        original = tweet['entities']['media'][0]['media_url_https']
    elif 'retweeted_status' in tweet:
        if 'media' in tweet['retweeted_status']['entities']:
            retweet = (tweet['retweeted_status']['entities']['media'][0]['media_url_https'])
        elif 'extended_tweet' in tweet['retweeted_status']:
            if 'media' in tweet['retweeted_status']['extended_tweet']['entities']:
                retweet = tweet['retweeted_status']['extended_tweet']['entities']['media'][0]['media_url_https']
    if retweet and not original: return retweet
    elif original and not retweet: return original
    elif original and retweet: return original
    else: return None


def check_and_extract_image_text(tweet: tweepy.api.API) -> str or None:
    media_text = check_media(tweet._json)
    if media_text:
        media_text = extract_text_from_image(media_text)
    return media_text


def extended_tweet(tweet: dict) -> str:
    if 'extended_tweet' in tweet:
        text = tweet['extended_tweet']['full_text']
    elif 'retweeted_status' in tweet:
        if 'extended_tweet' in tweet['retweeted_status']:
            text = tweet['retweeted_status']['extended_tweet']['full_text']
        elif 'text' in tweet['retweeted_status']:
            text = tweet['retweeted_status']['text']
    else: text = tweet['text']
    return text_first_clean(text)


def text_first_clean(text: str) -> str:
    for word in text.split(' '):
        if word.startswith('@') or word.startswith('http'): text = text.replace(word, '')
    text = unicodedata.normalize('NFD', text)
    text = text.encode('ascii', 'ignore')
    text = text.decode("utf-8")
    text = text.replace("\n", '')
    text = text.lower()
    text = re.sub("[^a-zA-Z-0-9]+", ' ', text)
    return text


def find_external_url(tweet: dict) -> str or None:
    if 'retweeted_status' in tweet:
        if 'extended_tweet' in tweet['retweeted_status']:
            if tweet['retweeted_status']['extended_tweet']['entities']['urls']:
                return tweet['retweeted_status']['extended_tweet']['entities']['urls'][0]['expanded_url']
            elif tweet['retweeted_status']['entities']['urls']:
                return tweet['retweeted_status']['entities']['urls'][0]['expanded_url']
        elif 'quoted_status' in tweet['retweeted_status']:
            if 'extended_entities' in tweet['retweeted_status']['quoted_status']:
                if tweet['retweeted_status']['quoted_status']['extended_entities']:
                    return tweet['retweeted_status']['quoted_status']['extended_entities']['media'][0]['expanded_url']
            elif tweet['quoted_status']['entities']['urls']:
                return tweet['quoted_status']['entities']['urls'][0]['expanded_url']
    elif tweet['entities']['urls']:
        return tweet['entities']['urls'][0]['expanded_url']
    elif 'quoted_status' in tweet:
        if tweet['quoted_status']['entities']['urls']:
            return tweet['quoted_status']['entities']['urls'][0]['expanded_url']





