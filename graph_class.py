import networkx
import matplotlib.pyplot as plt
import pandas as pd


class TextNetwork:
    def __init__(self):
        self.graph = networkx.Graph()
        self.hashgraph = networkx.Graph()

    def create_undirected_graph(self, df: pd.DataFrame) -> networkx.Graph:
        for tweet in df['tweet_text']:
            tmp = set(tweet)
            for word in tmp:
                for pair in tmp:
                    if word != pair:
                        if not self.graph.has_edge(word, pair):
                            self.graph.add_edge(word, pair, count=1)
                        else:
                            self.graph[word][pair]['count'] += 1
        return self.graph

    def create_hashtag_graph(self, df: pd.DataFrame) -> networkx.Graph:
        for tweet in df['hashtags']:
            tmp = set(tweet)
            for word in tmp:
                for pair in tmp:
                    if word != pair:
                        if not self.hashgraph.has_edge(word, pair):
                            self.hashgraph.add_edge(word, pair, count=1)
                        else:
                            self.hashgraph[word][pair]['count'] += 1
        return self.hashgraph


def filter_by_min(graph, min: int) -> networkx.Graph:
    res = []
    for (u, v, d) in graph.edges(data=True):
        if d['count'] > min:
            res.append((u,v, dict(count=d['count'])))
    return networkx.Graph(res)

def filter_by_top(graph, min: int) -> networkx.Graph:
    res = set()
    for (u, v, d) in graph.graph.edges(data=True): res.add(d['count'])
    ls = list(res)
    ls.sort()
    res = []
    for (u, v, d) in graph.graph.edges(data=True):
        if d['count'] > ls[-min]:
            res.append((u, v, dict(count=d['count'])))
    return networkx.Graph(res)

def ego_word_network(self, word: str) -> dict:
    return self.graph[word]

def plot_all_network(graph, min: int):
    graph = filter_by_min(graph, min)
    dim = [1000]*len(graph.nodes())
    plt.figure(3,figsize=(15,15))
    pos=networkx.spring_layout(graph)
    networkx.draw(graph, pos, cmap=plt.get_cmap('jet'),node_size=dim)
    networkx.draw_networkx_labels(graph, pos)
    plt.show()
