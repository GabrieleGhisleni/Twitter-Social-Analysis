from sklearn.cluster import SpectralClustering
from nltk.probability import FreqDist
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import operator
import networkx



class NetworkPlot:
    def __init__(self, graph: networkx.Graph, frequency_dist: FreqDist, label_thresold: int, color_thresold: int = None, labels: list = None):
        self.freq_dist = frequency_dist
        self.graph = graph
        self.label_thresold = label_thresold
        self.color_thresold = color_thresold
        self.labels = labels
        self.colors = list(mcolors.TABLEAU_COLORS.values())

    def get_size(self, word: str) -> int:
        return self.freq_dist.get(word)

    def get_labels(self) -> dict:
        labels={}
        for node in self.graph.nodes():
            if self.get_size(node) > self.label_thresold: labels[node] = node
        return labels

    def get_node_size(self):
        return [self.get_size(i) for i in self.graph.nodes()]

    def get_node_thresold_color(self):
        return [['#1f78b4', 'lightblue'][self.get_size(node) > self.color_thresold] for node in self.graph.nodes()]

    def get_node_color_clustering(self, rnd: bool = False, colors: list = False):
        res, tmp = [],{}
        if rnd:
            for i in np.unique(self.labels):
                tmp[i] = self.colors[np.random.randint(0, len(self.colors))]
            for i in self.labels:
                res.append(tmp[i])
        elif colors:
            for i in self.labels:
                res.append(colors[i])
        else:
            for i in self.labels:
                res.append(self.colors[i])
        return res

    def plot(self):
        plt.figure(3, figsize=(22, 22))
        layout = networkx.spring_layout(self.graph)

        networkx.draw(G=self.graph,
                      pos=layout,
                      cmap=plt.get_cmap('autumn'),
                      node_size=self.get_node_size(),
                      node_color=self.get_node_thresold_color() if self.color_thresold else self.get_node_color_clustering())

        networkx.draw_networkx_labels(self.graph,
                                      pos=layout,
                                      labels=self.get_labels(),
                                      font_size=25,
                                      font_color='firebrick')
        plt.show()

    @staticmethod
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
                            if check_thresold(word=pair, distrib=distrib, value=thresold):
                                if word != pair:
                                    if not res.has_edge(word, pair):
                                        res.add_edge(word, pair, count=1)
                                    else:
                                        res[word][pair]['count'] += 1
        return res

    @staticmethod
    def filter_pairwise_words(graph: networkx.Graph, thresold: int) -> networkx.Graph:
        res = []
        for (u, v, d) in graph.edges(data=True):
            if d['count'] > thresold:
                res.append((u,v, dict(count=d['count'])))
        return networkx.Graph(res)

    @staticmethod
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

    @staticmethod
    def keep_connected_components(graph: networkx.Graph, min_degree: int) -> None:
        for component in list(networkx.connected_components(graph)):
            if len(component) < min_degree:
                for node in component:
                    graph.remove_node(node)

    @staticmethod
    def spectral_clustering(graph: networkx.Graph, n_cluster: int) -> list:
        adj_matrix  = networkx.to_numpy_matrix(graph)
        spectral_clustering = SpectralClustering(n_cluster, affinity='precomputed', n_init=100, assign_labels='discretize')
        spectral_clustering.fit(adj_matrix)
        return spectral_clustering.labels_

    @staticmethod
    def plot_centrality(levels_of_centrality: list):
        names = ["Degree Centrality", "Degree Betwenness"]
        fig, axes = plt.subplots(2,1, figsize=(12, 10))
        for iel in range(len(levels_of_centrality)):
            if iel == 0: to = 65
            else: to = 30
            tmp = dict(sorted(levels_of_centrality[iel].items(), key = operator.itemgetter(1), reverse=True)[:to])
            df = pd.DataFrame(tmp, index = [0]).T.reset_index()
            sns.barplot(data=df, x='index', y=0, palette='viridis', ax = axes[iel])
            axes[iel].tick_params(labelrotation=90)
            axes[iel].set_title(names[iel], fontsize=25)
            axes[iel].set_xlabel('')
            axes[iel].set_ylabel('')
            axes[iel].set_yticks([])
        fig.tight_layout()
        plt.show()

    @staticmethod
    def count_barplot(count: dict, thresold: int = 20) -> None:
        fig=plt.figure(figsize=(20, 15))
        sns.set_style('white')
        word, freq=[], []
        for key in count:
            if count[key] > thresold:
                word.append(key)
                freq.append(count[key])
        df=pd.DataFrame(freq, word).reset_index(). \
            rename(columns={'index': 'words', 0: 'freq'}).sort_values(by='freq', ascending=False)
        sns.barplot(y='words', x="freq", data=df, palette='viridis')
        plt.title('Most Frequent External URL\n\n', fontsize=35)
        plt.xlabel('')
        plt.ylabel('')
        plt.show()