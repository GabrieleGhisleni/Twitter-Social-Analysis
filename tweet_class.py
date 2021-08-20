from __future__ import annotations
import re
import requests
from PIL import Image
from pytesseract import pytesseract
import tweepy


class Tweet:
    def __init__(self, created_at, id, tweet_text, is_reply, reply_count, retweet_count, hashtags, external_url, author_followers, author_follow, author_loc, media_text):
        self.created_at = created_at
        self.id = id
        self.tweet_text = tweet_text
        self.is_reply = is_reply
        self.reply_count = reply_count
        self.retweet_count = retweet_count
        self.hashtags = hashtags
        self.external_url = external_url
        self.author_followers = author_followers
        self.author_followe = author_follow
        self.author_loc = author_loc
        self.media_text = media_text

    def __str__(self):
        print(self.id)

    @staticmethod
    def from_api_to_class(status: tweepy.api.API) -> Tweet:
        if status.truncated:
            text = status._json['extended_tweet']['full_text']
        else: text = status.text
        is_a_reply = True if status.in_reply_to_status_id or status.in_reply_to_user_id else False
        media_text = check_media(status._json)
        if media_text: media_text = extract_text_from_image(media_text)
        return Tweet( created_at=status.created_at,
                      id=str(status.id),
                      tweet_text=text,
                      is_reply=is_a_reply,
                      reply_count= status.reply_count,
                      retweet_count=status.retweet_count,
                      author_followers=status.author.followers_count,
                      author_follow=status.author.friends_count,
                      author_loc=status.author.location,
                      hashtags=status.entities['hashtags'],
                      external_url=status.entities['urls'],
                      media_text=media_text )

def extract_text_from_image(url, timeout=5, path_to_tesseract=r"C:\Program Files\Tesseract-OCR\tesseract.exe") -> str or None:
    pytesseract.tesseract_cmd = path_to_tesseract
    raw_image = requests.get(url, stream=True).raw
    try:
        text = pytesseract.image_to_string(Image.open(raw_image), timeout=timeout)
        text = text[:-1].replace("\n", '')
        text = re.sub("[^a-zA-Z]+", ' ', text)
        text = text.lower()
        return text
    except RuntimeError:
        print(f'Exceeded timeout ({timeout}s)')
    except Exception as e:
        print(e, url)

def check_media(tweet: dict) -> str or None:
    original, retweet = False, False
    if 'media' in tweet['entities']:
        original = tweet['entities']['media'][0]['media_url_https']
    if 'retweeted_status' in tweet:
        if 'media' in tweet['retweeted_status']['entities']:
            retweet = (tweet['retweeted_status']['entities']['media'][0]['media_url_https'])
    if retweet and not original: return retweet
    elif original and not retweet: return original
    elif original and retweet: return original
    else: return None