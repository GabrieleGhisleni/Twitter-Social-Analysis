from sklearn.cluster import SpectralClustering
from sklearn.metrics import silhouette_score
from nltk.probability import FreqDist
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from typing import Tuple
import seaborn as sns
import pandas as pd
import numpy as np
import operator
import networkx



class NetworkPlot:
    def __init__(self, graph: networkx.Graph, frequency_dist: FreqDist = None, label_thresold: int = None, color_thresold: int = None, labels: list = None):
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

    def get_node_size_centrality_and_labels(self, centrality: dict, mul_factor: int, upper=False) -> Tuple[list,dict]:
        node_sizes, labelsss=[], {}
        for u in self.graph.nodes():
            flag=set()
            if u in centrality:
                node_sizes.append(self.freq_dist[u] * mul_factor)
                if upper: labelsss[u.upper()]=u.upper()
                else: labelsss[u]=u
                flag.add(u)
            else:
                node_sizes.append(15)
        return node_sizes, labelsss

    def plot_main_centrality(self, res, mul_factor: int = 5, save: str = None, upper=True, k=2, i=50, w='count', ka=True, c=None):
        fig, axes= plt.subplots(1,1, figsize=(20, 20))
        plt.style.use('seaborn-white')
        if ka: layout = networkx.kamada_kawai_layout(self.graph, weight=w)
        else: layout = networkx.spring_layout(self.graph, weight=w, k=k, iterations=i)
        node_sizes, labels = self.get_node_size_centrality_and_labels(res, mul_factor, upper)

        networkx.draw_networkx_nodes(G=self.graph,
                                  pos=layout,
                                  cmap=plt.get_cmap('autumn'),
                                  node_size=node_sizes,
                                  node_color='lightblue' if not np.array(self.labels).any() else self.get_node_color_clustering(colors=c),
                                  ax=axes,
                                  alpha=0.8)

        networkx.draw_networkx_edges(G=self.graph,
                                     edge_color='black',
                                     # edge_color= 'black' if not np.array(self.labels).any() else self.get_node_color_clustering(),
                                     pos=layout,
                                     width = [self.graph[u][v]['count'] / 7 for u,v in self.graph.edges],
                                     alpha=0.10)

        networkx.draw_networkx_labels(self.graph,
                                      pos=layout,
                                      labels=labels,
                                      font_size=17,
                                      ax=axes,
                                      font_color='black',
                                      font_weight='bold',
                                      verticalalignment='top')
        axes.set_axis_on()
        axes.grid(False)
        plt.title('Semantic Network Representation', fontsize=25)
        if save:
            plt.savefig(f'photos/{save}.png', format='png', dpi=300)
        plt.show()

    @staticmethod
    def extract_top_centrality_words(centrality, percentage):
        res = {}
        for centr in centrality:
            q, perc = 0, (len(centr) / 100 * percentage)
            for i in centr.items():
                if q < perc:
                    if i[0] in res:
                        if res[i[0]] < i[1]: res[i[0]]=i[1]
                    else:
                        res[i[0]]=i[1]
                q+=1
        return res

    @staticmethod
    def create_graph_from_top_centrality(graph, res):
        tmp=[]
        for (u, v, d) in graph.edges(data=True):
            if u in res or v in res:
                tmp.append((u, v, dict(count=d['count'])))
        return networkx.Graph(tmp)

    @staticmethod
    def graph_filtered_dist(df: pd.DataFrame, distrib: FreqDist, thresold: int, obj: str = 'tweet') -> networkx.Graph:
        def check_thresold(word, distrib: FreqDist, value: int):
            return distrib.get(word) > value
        res = networkx.Graph()
        if obj=='tweet': bag= df['tweet_text']
        elif obj =='hash': bag = df['hashtags']
        else: bag = df
        for tweet in bag:
            if (tweet):
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
    def keep_minimun_degree(graph: networkx.Graph, min_degree: int = 2) -> networkx.Graph:
        res = networkx.Graph.copy(graph)
        for component in graph.degree:
            if component[1] < min_degree:
                res.remove_node(component[0])
        return res

    @staticmethod
    def get_set_top_centrality_words(centrality, top=20):
        res = set()
        for i in centrality:
            tmp=dict(sorted(i.items(), key=operator.itemgetter(1), reverse=True)[:top])
            for i in list(tmp.keys()): res.add(i)
        return res

    @staticmethod
    def spectral_clustering(graph: networkx.Graph, n_cluster: int, k: int = None, gamma: float = 1.0, w = 'count', check = False):
        adj_matrix  = networkx.to_numpy_matrix(graph, weight=w)
        if k: spectral_clustering=SpectralClustering(n_cluster, affinity='nearest_neighbors', n_init=500, assign_labels='discretize', n_neighbors=k)
        else: spectral_clustering=SpectralClustering(n_cluster, affinity='rbf', assign_labels='discretize', gamma=gamma)
        spectral_clustering.fit(adj_matrix)
        counts, sil =  np.unique(spectral_clustering.labels_, return_counts=True)[1].tolist() ,silhouette_score(adj_matrix, spectral_clustering.labels_)
        if check: return silhouette_score(adj_matrix, spectral_clustering.labels_)
        else:
            print(f"""Cluster distribution: {counts}, Silhouette Score: {sil}""")
            return spectral_clustering, spectral_clustering.labels_

    @staticmethod
    def plot_centrality(levels_of_centrality: list, save: bool = False, h=10):
        names = ["Degree Centrality", "Degree Betwenness", 'Degree Closeness']
        fig, axes = plt.subplots(len(levels_of_centrality),1, figsize=(12, h))
        for iel in range(len(levels_of_centrality)):
            if iel == 0: to = 40
            else: to = 30
            tmp = dict(sorted(levels_of_centrality[iel].items(), key = operator.itemgetter(1), reverse=True)[:to])
            df = pd.DataFrame(tmp, index = [0]).T.reset_index()
            df['index'] = df['index'].apply(lambda x: x[:7]+'[..]' if len(x) > 12 else x)
            sns.barplot(data=df, x='index', y=0, palette='viridis', ax = axes[iel])
            axes[iel].tick_params(labelrotation=90)
            axes[iel].set_title(names[iel], )
            axes[iel].set_xlabel('')
            axes[iel].set_ylabel('')
            axes[iel].set_yticks([])
        fig.tight_layout()
        if save:
            plt.savefig(f'photos/centrality_levels.eps', format='eps', dpi=300)
        plt.show()

    @staticmethod
    def plot_single_centrality_v(levels_of_centrality: dict, name, to, save: bool = False, h=15, lb=15):
        fig = plt.figure(figsize=(12, h))
        tmp = dict(sorted(levels_of_centrality.items(), key=operator.itemgetter(1), reverse=True)[:to])
        df = pd.DataFrame(tmp, index=[0]).T.reset_index()
        df['index'] = df['index'].apply(lambda x: x[:7] + '[..]' if len(x) > 10 else x)
        df['index'] = df['index'].apply(lambda x: x.upper())
        sns.barplot(data=df, y='index', x=0, palette='viridis')
        plt.tick_params(labelrotation=0)
        plt.title(f"Highest {name}")
        plt.xlabel('')
        plt.ylabel('')
        plt.xticks([])
        plt.tick_params(axis='both', which='major', labelsize=lb)
        fig.tight_layout()
        if save:
            plt.savefig(f'photos/centrality_levels_v.eps', format='eps', dpi=300)
        plt.show()

    @staticmethod
    def plot_centrality_v(levels_of_centrality: list, save: bool = False, h=15, names="Degree Centrality,Degree Betwenness,Degree Closeness", lb=15):
        fig, axes=plt.subplots(len(levels_of_centrality), 1, figsize=(8, h))
        for iel in range(len(levels_of_centrality)):
            if iel == 0:
                to=15
            else:
                to=15
            tmp=dict(sorted(levels_of_centrality[iel].items(), key=operator.itemgetter(1), reverse=True)[:to])
            df=pd.DataFrame(tmp, index=[0]).T.reset_index()
            df['index']=df['index'].apply(lambda x: x[:7] + '[..]' if len(x) > 10 else x)
            df['index']=df['index'].apply(lambda x: x.upper())
            sns.barplot(data=df, y='index', x=0, palette='viridis', ax=axes[iel])
            axes[iel].tick_params(labelrotation=0)
            axes[iel].set_title(f"Highest {names.split(',')[iel]}")
            axes[iel].set_xlabel('')
            axes[iel].set_ylabel('')
            axes[iel].set_xticks([])
            axes[iel].tick_params(axis='both', which='major', labelsize=lb)
        fig.tight_layout()
        if save:
            plt.savefig(f'photos/centrality_levels_v.eps', format='eps', dpi=300)
        plt.show()

    @staticmethod
    def count_barplot(count: dict, name, title, ticks, to: int = 10, save = False, h=20, w=20, r=False) -> None:
        def filters_k(s):
            if s.startswith('www'): s=s[4:]
            if s.endswith('/'): s=s[:-1]
            return s.upper()
        freq = dict(sorted(count.items(), key=operator.itemgetter(1), reverse=True)[:to])
        df = pd.DataFrame(freq, index=[0]).T.reset_index().rename(columns={'index': 'words', 0: 'freq'})
        df.loc[:, 'words'] = df.words.apply(filters_k)
        fig = plt.figure(figsize=(w, h))
        sns.barplot(y='words', x="freq", data=df, color='royalblue', edgecolor='blue', alpha=0.4)
        plt.title(f'{title}', fontsize=35)
        plt.xlabel('')
        plt.ylabel('')
        for line in range(0, len(df)):
            if line < 5 and r: ha = 'right'
            else: ha = 'left'
            plt.annotate(df.words[line].upper(),
                             (df.freq[line], line),
                             textcoords="offset points",
                             xytext=(2, -10),
                             ha=ha,
                             weight='bold',
                             size=25)
        plt.yticks([])
        plt.xticks(ticks)
        if save:
            plt.savefig(f'photos/{name}.png', format='png', dpi=300)
        plt.show()


    @staticmethod
    def find_best_k_and_cluster(graph: networkx.Graph, k, gamma) -> dict:
        n_cluster = [3, 4, 5, 6, 7, 8, 9, 10, 15]
        res = {}
        for i in n_cluster:
            res[i], tmp, tmp_e = {}, 0, 0
            for j in range(len(k)):
                try:
                    score = NetworkPlot.spectral_clustering(graph, n_cluster=i, k=k[j], w='count', check=True)
                    if score > tmp:
                        tmp = score
                        z = (k[j], score)
                except Exception:
                    z = None
                try:
                    score = NetworkPlot.spectral_clustering(graph, n_cluster=i, gamma=gamma[j], w='count', check=True)
                    if score > tmp_e:
                        tmp_e = score
                        z_e = (gamma[j], score)
                except Exception:
                    z_e = None
            res[i]= z, z_e
        return res

    @staticmethod
    def extend_top_word(graph, thresold, res):
        ret = res.copy()
        for u, v in graph.edges:
            if graph[u][v]['count'] > thresold:
                ret.add(u)
                ret.add(v)
        return ret
