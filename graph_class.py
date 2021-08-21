import networkx
import matplotlib.pyplot as plt
import pandas as pd

class TextNetwork:
    def __init__(self):
        self.graph = networkx.Graph()

    def create_undirected_graph(self, df: pd.DataFrame) -> networkx.Graph:
        for tweet in df['tweet_t']:
            tmp = set(tweet)
            for word in tmp:
                for pair in tmp:
                    if word != pair:
                        if not self.graph.has_edge(word, pair):
                            self.graph.add_edge(word, pair, count=1)
                        else:
                            self.graph[word][pair]['count'] += 1
        return self.graph

    def filter_by_min(self, min: int) -> networkx.Graph:
        res = []
        for (u, v, d) in self.graph.edges(data=True):
            if d['count'] > min:
                res.append((u,v, dict(count=d['count'])))
        return networkx.Graph(res)

    def filter_by_top(self, min: int) -> networkx.Graph:
        res = set()
        for (u, v, d) in self.graph.edges(data=True): res.add(d['count'])
        ls = list(res)
        ls.sort()
        res = []
        for (u, v, d) in self.graph.edges(data=True):
            if d['count'] > ls[-min]:
                res.append((u, v, dict(count=d['count'])))
        return networkx.Graph(res)

    def ego_word_network(self, word: str) -> dict:
        return self.graph[word]

    def plot_all_network(self, min: int):
        dim = [1000]*len(self.graph.nodes())
        plt.figure(3,figsize=(15,15))
        pos=networkx.spring_layout(self.graph)
        networkx.draw(self.graph, pos, cmap=plt.get_cmap('jet'),node_size=dim)
        networkx.draw_networkx_labels(self.graph, pos)
        plt.show()


if __name__ == '__main__':
    ""