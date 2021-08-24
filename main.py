from graph_class import *
from nltk_class import *
from text_mining import *

if __name__ == '__main__':
    nlp = NltkTextProcessing()
    net = NetworkPlot
    with open('twitter_.json', 'r') as file:
        raw_tweet = pd.DataFrame(json.load(file))
    tweet_no_hash = nlp.remove_hashtag_from_text(raw_tweet)
    tweet_df = nlp.process_df_text_column(tweet_no_hash, stem=False)  # steem
    tweet_unique_df = nlp.keep_unique(tweet_df)
    freq_distrib_tweet = nlp.frequency_dist(tweet_df, obj='tweet')

    graph_tweet = net.graph_filtered_dist(tweet_unique_df, freq_distrib_tweet, 2)
    graph_tweet_filter = net.filter_pairwise_words(graph_tweet, 30)
    net.keep_connected_components(graph_tweet_filter, min_degree=5)

    NetworkPlot(graph=graph_tweet_filter, frequency_dist=freq_distrib_tweet, label_thresold=100,
                color_thresold=900).plot()