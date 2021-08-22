import networkx
import matplotlib.pyplot as plt
import pandas as pd
from nltk.probability import FreqDist


class NetworkPlot:
    def __init__(self, graph: networkx.Graph, frequency_dist: FreqDist, label_thresold: int, color_thresold: int):
        self.freq_dist = frequency_dist
        self.graph = graph
        self.label_thresold = label_thresold
        self.color_thresold = color_thresold

    def get_size(self, word: str) -> int:
        return self.freq_dist.get(word)

    def get_labels(self) -> dict:
        labels={}
        for node in self.graph.nodes():
            if self.get_size(node) > self.label_thresold: labels[node] = node
        return labels

    def get_node_size(self):
        return [self.get_size(i) for i in self.graph.nodes()]

    def get_node_color(self):
        return [['#1f78b4', 'lightblue'][self.get_size(node) > self.color_thresold] for node in self.graph.nodes()]

    def plot(self):
        plt.figure(3, figsize=(22, 22))
        layout = networkx.spring_layout(self.graph)

        networkx.draw(G=self.graph,
                      pos=layout,
                      cmap=plt.get_cmap('autumn'),
                      node_size=self.get_node_size(),
                      node_color=self.get_node_color())

        networkx.draw_networkx_labels(self.graph,
                                      pos=layout,
                                      labels=self.get_labels(),
                                      font_size=25,
                                      font_color='firebrick')
        plt.show()


def graph_filtered_dist(df: pd.DataFrame, distrib: FreqDist, thresold: int, obj: str = 'tweet') -> networkx.Graph:
    def check_thresold(word, distrib: FreqDist, value: int):
        return distrib.get(word) > value
    res = networkx.Graph()
    bag = df['tweet_text'] if obj == 'tweet' else df['hashtags']
    for tweet in bag:
        if tweet:
            for word in tweet:
                if check_thresold(word=word, distrib=distrib, value=thresold):
                    for pair in tweet:
                        if word != pair:
                            if not res.has_edge(word, pair):
                                res.add_edge(word, pair, count=1)
                            else:
                                res[word][pair]['count'] += 1
    return res

def filter_pairwise_words(graph: networkx.Graph, thresold: int) -> networkx.Graph:
    res = []
    for (u, v, d) in graph.edges(data=True):
        if d['count'] > thresold:
            res.append((u,v, dict(count=d['count'])))
    return networkx.Graph(res)

def filter_by_top(graph: networkx.Graph, min: int) -> networkx.Graph:
    res = set()
    for (u, v, d) in graph.graph.edges(data=True): res.add(d['count'])
    ls = list(res)
    ls.sort()
    res = []
    for (u, v, d) in graph.graph.edges(data=True):
        if d['count'] > ls[-min]:
            res.append((u, v, dict(count=d['count'])))
    return networkx.Graph(res)



