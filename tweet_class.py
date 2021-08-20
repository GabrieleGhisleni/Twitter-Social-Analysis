class Tweet:
    def __init__(self, created_at, id, tweet_text, is_reply, reply_count, retweet_count,
                 hashtags, external_url, author_followers, author_follow, author_loc):
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

    def __str__(self):
        print(self.id)

    @staticmethod
    def from_api_to_class(status):
        if status.truncated:
            text = status._json['extended_tweet']['full_text']
        else: text = status.text
        is_a_reply = True if status.in_reply_to_status_id or status.in_reply_to_user_id else False
        return Tweet(created_at=status.created_at,
                      id=str(status.id),
                      tweet_text=text,
                      is_reply=is_a_reply,
                      reply_count= status.reply_count,
                      retweet_count=status.retweet_count,
                      author_followers = status.author.followers_count,
                      author_follow=status.author.friends_count,
                      author_loc=status.author.location,
                      hashtags = status.entities['hashtags'],
                      external_url = status.entities['urls'])
