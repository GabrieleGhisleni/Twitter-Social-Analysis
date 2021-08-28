from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer
from sklearn.pipeline import make_pipeline
from nltk.tokenize import word_tokenize
from sklearn.cluster import KMeans
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from matplotlib import cm
from typing import Tuple
import seaborn as sns
import pandas as pd
import numpy as np
import umap


class TextMining:
    def __init__(self, ngram_range: int = 1):
        self.tfid_vectorizer = TfidfVectorizer(ngram_range=(1, ngram_range), norm='l2', max_df = 0.999, min_df = 1)
        self.count_vectorizer = CountVectorizer()

    def vectorized_text(self, text_to_vectorize: list):
        res = self.tfid_vectorizer.fit_transform(text_to_vectorize)
        print(f'Shape of the Sparse Matrix {res.shape}, type: {type(res)}')
        return res

    def get_features_names(self):
        return self.tfid_vectorizer.get_feature_names()

    def lda_topic_modeling(self, encoded, topics: int) -> LatentDirichletAllocation:
        lda = LatentDirichletAllocation(n_components=topics, max_iter=5,
                                          learning_method='online',
                                          learning_offset=50.,
                                          random_state=42)
        lda.fit(encoded)
        return lda

    def plot_lda_topic(self, model: LatentDirichletAllocation, topics: int, n_top_words: int, save: bool = False) -> None:
        if topics < 10: height = 20
        elif topics < 20: height = 35
        elif topics < 30: height = 55
        else: height = 65
        fig, axes = plt.subplots(topics, 1, figsize=(20, height))
        features_names, i = self.tfid_vectorizer.get_feature_names(), 0
        for topic_idx, topic in enumerate(model.components_):
            top_features_ind = topic.argsort()[:-n_top_words - 1:-1]
            top_features = [features_names[i] for i in top_features_ind]
            weights = topic[top_features_ind]
            sns.barplot(y=top_features, x=weights, ax=axes[i], palette='viridis')
            axes[i].tick_params(axis='y', which='minor', labelsize=7)
            axes[i].set_title(f'Topic {i} characterizing words', fontsize=20)
            axes[i].set_xticks([])
            sns.set_style('white')
            i += 1
        fig.tight_layout()
        if save:
            plt.savefig(f'photos/lda_topics.eps', format='eps')
        plt.show()

    def word_cloud_dict(self, model: LatentDirichletAllocation) -> dict:
        features_names, tmp = self.tfid_vectorizer.get_feature_names(), {}
        for topic_idx, topic in enumerate(model.components_):
            tmp[topic_idx] = {}
            top_features_ind = topic.argsort()[::-1]
            top_features = [features_names[i] for i in top_features_ind]
            weights = topic[top_features_ind]
            for iel in range(len(weights)):
                tmp[topic_idx][top_features[iel]] = int(weights[iel])
        return tmp

    def clustering_kmeans(self, reduced_data: np.array, n_cluster:int = 3) -> KMeans:
        cluster_model = KMeans(n_clusters=n_cluster, init='k-means++', max_iter=100, n_init=15)
        cluster_model.fit(reduced_data)
        return cluster_model

    def plot_multiple_umaps(self, data: list, k: list, n_cluster: int = 3) -> None:
        def plot_single(data, n_cluster, axes, k):
            umap_df=pd.DataFrame(data, columns=[f'Component {i + 1}' for i in range(2)])
            kmeans_umap=self.clustering_kmeans(umap_df, n_cluster)
            sns.scatterplot(data=umap_df, x='Component 1', y='Component 2', hue=kmeans_umap.labels_, palette='viridis', ax=axes)
            axes.set_title(f'Perplexity = {k}', fontsize=25)
        fig, axes = plt.subplots(2, 2, figsize=(15, 15))
        sns.set_style('white')
        fig.suptitle('UMAP dimensionality reduction\n', fontsize=35)
        plot_single(data[0], n_cluster, axes[0][0], k=k[0])
        plot_single(data[1], n_cluster, axes[0][1], k=k[1])
        plot_single(data[2], n_cluster, axes[1][0], k=k[2])
        plot_single(data[3], n_cluster, axes[1][1], k=k[3])
        fig.tight_layout()
        plt.show()

    def plot_umaps(self, data: pd.DataFrame, k: list, n_cluster: int = 3, palette: str = 'viridis', save: str = None) -> None:
        fig = plt.figure(figsize=(15, 15))
        kmeans_umap = self.clustering_kmeans(data.loc[:, ['Component 1','Component 2']], n_cluster)
        data['cluster'] = kmeans_umap.labels_
        sns.scatterplot(data=data, x='Component 1', y='Component 2', hue=data['cluster'], palette=palette)
        legend = plt.legend(fontsize="large")
        for line in range(0, len(data)):
            plt.annotate(data.labels[line].upper(),
                         (data['Component 1'][line], data['Component 2'][line]),
                         textcoords="offset points",
                         xytext=(0, 5),
                         ha='center',
                         size='large',
                         color = legend.legendHandles[data['cluster'][line]]._facecolors[0])
        plt.title(f'UMAP dimensionality reduction k = {k}', fontsize=25)
        plt.xticks([])
        plt.yticks([])
        plt.tight_layout()
        sns.set_style('white')
        if save:
            plt.savefig(f'photos/{save}-umap-{k}.eps', format='eps')
        plt.show()

    def get_wordcloud_lsa(self, svd_model, topics) -> list:
        svd_df = pd.DataFrame(svd_model.components_, columns=(self.tfid_vectorizer.get_feature_names())).T
        res = []
        for i in range(topics):
            d = {}
            tmp = svd_df.loc[(svd_df[i] > 0.001), [i]][0:250].reset_index().sort_values(by=i, ascending=False)
            tmp[i] = tmp[i].apply(lambda x: x * 100)
            for a, x in tmp.values: d[a]=x
            res.append(d)
        return res

    def plot_lsa_topic(self, svd_model, topics, top_word, save: bool = False):
        fig, axes=plt.subplots(topics, 1, figsize=(20, 20))
        svd_df=pd.DataFrame(svd_model.components_, columns=(self.tfid_vectorizer.get_feature_names())).T
        for i in range(topics):
            tmp = svd_df.loc[(svd_df[i] > 0.1), [i]][0:top_word].reset_index().sort_values(by=i, ascending=False)
            sns.barplot(data=tmp, y='index', x=i, palette='viridis', ax=axes[i])
            axes[i].set_title(f'Topic {i} characterizing words', fontsize=20)
            axes[i].set_ylabel('')
            axes[i].set_xlabel('')
        fig.tight_layout()
        if save:
            plt.savefig(f'photos/lsa-topic.eps', format='eps')
        plt.show()

    @staticmethod
    def latent_semantic_analysis(encoded, components: int) -> Tuple[TruncatedSVD,np.array]:
        svd = TruncatedSVD(n_components=components, n_iter=10)
        normalizer = Normalizer(norm='l2', copy=False)
        lsa = make_pipeline(svd, normalizer)
        svd_result = lsa.fit_transform(encoded)
        print(f"Explained variance of the SVD step: {svd.explained_variance_ratio_.sum()}%")
        return svd, svd_result

    @staticmethod
    def plot_lsa(svd_result: np.array, kmeans_model: KMeans, n_components: int, save: bool = False) -> None:
        svd_df=pd.DataFrame(svd_result, columns=[f'Component {i + 1}' for i in range(n_components)])
        svd_df['cluster']=kmeans_model.labels_
        fig, axes=plt.subplots(1, 3, figsize=(23, 10))
        axes[0].set_title('Latent Semantic Analisys 1-2\n', fontsize=25)
        axes[1].set_title('Latent Semantic Analisys 1-3\n', fontsize=25)
        axes[2].set_title('Latent Semantic Analisys 2-3\n', fontsize=25)
        sns.scatterplot(ax=axes[0], data=svd_df, x='Component 1', y='Component 2', hue='cluster', palette='tab10')
        sns.scatterplot(ax=axes[1], data=svd_df, x='Component 1', y='Component 3', hue='cluster', palette='tab10')
        sns.scatterplot(ax=axes[2], data=svd_df, x='Component 2', y='Component 3', hue='cluster', palette='tab10')
        fig.tight_layout()
        if save:
            plt.savefig(f'photos/lsa.eps', format='eps')
        plt.show()

    @staticmethod
    def umaps(data, k=None, dist=None) -> np.array:
        kk, ddist = 2, 0.1
        if k: kk = k
        if dist: ddist = dist
        reducer=umap.UMAP(random_state=42, n_neighbors=kk, min_dist=ddist, metric='euclidean', n_components=2)
        reducer.fit(data)
        return reducer.transform(data)

    @staticmethod
    def plot_wordcloud(data, n_topics, save: str = None):
        if n_topics == 4: fig, axes=plt.subplots(2, 2, figsize=(20, 14))
        elif n_topics == 6: fig, axes=plt.subplots(3, 2, figsize=(20, 16))
        else: fig, axes=plt.subplots(4, 2, figsize=(20, 18))
        wordcloud=WordCloud(margin=10, background_color='white', colormap='inferno', width=640, height=400,
                            max_words=150)
        for iel in range(n_topics):
            if iel == 0: ax, ax1 = 0, 0
            elif iel == 1: ax, ax1 = 0, 1
            elif iel == 2: ax, ax1 = 1, 0
            elif iel == 3: ax, ax1 = 1, 1
            elif iel == 4: ax, ax1 = 2, 0
            elif iel == 5: ax, ax1 = 2, 1
            elif iel == 6: ax, ax1 = 3, 0
            elif iel == 7: ax, ax1 = 3, 1
            word_clouded=wordcloud.generate_from_frequencies(data[iel])
            axes[ax][ax1].set_xticks([])
            axes[ax][ax1].set_yticks([])
            axes[ax][ax1].set_title(f'Topic {iel+1} characterizing words', fontsize=25)
            axes[ax][ax1].imshow(word_clouded)
        fig.tight_layout()
        if save:
            plt.savefig(f'photos/{save}.eps', format='eps')
        plt.show()

    @staticmethod
    def sort_coo(adj):
        coo_matrix=adj.tocoo()
        tuples=zip(coo_matrix.col, coo_matrix.data)
        return sorted(tuples, key=lambda x: (x[1], x[0]), reverse=True)

    @staticmethod
    def extract_topn_from_vector(feature_names, sorted_items, topn=10):
        sorted_items=sorted_items[:topn]
        score_vals=[]
        feature_vals=[]
        for idx, score in sorted_items:
            score_vals.append(round(score, 3))
            feature_vals.append(feature_names[idx])
        results={}
        for idx in range(len(feature_vals)):
            results[feature_vals[idx]]=score_vals[idx]
        print(f"Keywords founded: {len(results)}")
        return results

