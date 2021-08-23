from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer
from sklearn.pipeline import make_pipeline
from sklearn.cluster import KMeans
import umap
from typing import Tuple
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import scipy

class TextMining:
    def __init__(self, ngram_range: int = 1):
        self.tfid_vectorizer = TfidfVectorizer(ngram_range=(1, ngram_range), norm='l2')
        self.count_vectorizer = CountVectorizer()

    def vectorized_text(self, text_to_vectorize: list):
        return self.tfid_vectorizer.fit_transform(text_to_vectorize)

    def lda_topic_modeling(self, encoded, topics: int) -> Tuple[LatentDirichletAllocation, list]:
        lda = LatentDirichletAllocation(n_components=topics, max_iter=5,
                                          learning_method='online',
                                          learning_offset=50.,
                                          random_state=42)
        lda.fit(encoded)
        return lda, self.tfid_vectorizer.get_feature_names()

    def plot_lda_topic(self, model: LatentDirichletAllocation, n_top_words: int) -> None:
        fig, axes = plt.subplots(1, 3, figsize=(20, 7))
        fig.suptitle('Latent Dirichlet Allocation')
        i = 0
        for topic_idx, topic in enumerate(model.components_):
            top_features_ind=topic.argsort()[:-n_top_words - 1:-1]
            top_features=[self.tfid_vectorizer.get_feature_names()[i] for i in top_features_ind]
            weights=topic[top_features_ind]
            sns.barplot(y=top_features, x=weights, ax=axes[i])
            axes[i].set_title(f'Topic Number: {i}')
            i += 1

    def clustering_kmeans(self, reduced_data: np.array, n_cluster:int = 3) -> KMeans:
        cluster_model = KMeans(n_clusters=n_cluster, init='k-means++', max_iter=100, n_init=1)
        cluster_model.fit(reduced_data)
        return cluster_model

    def plot_umaps(self, data: list, n_cluster: int = 3) -> None:
        def plot_single(data, n_cluster, axes):
            umap_df=pd.DataFrame(data, columns=[f'Component {i + 1}' for i in range(2)])
            kmeans_umap=self.clustering_kmeans(umap_df, n_cluster)
            sns.scatterplot(data=umap_df, x='Component 1', y='Component 2', hue=kmeans_umap.labels_, palette='viridis', ax=axes)
            axes.set_title('nasd')
        fig, axes = plt.subplots(2, 2, figsize=(15, 15))
        plot_single(data[0], n_cluster, axes[0][0])
        plot_single(data[1], n_cluster, axes[0][1])
        plot_single(data[2], n_cluster, axes[1][0])
        plot_single(data[3], n_cluster, axes[1][1])
        plt.show()

    @staticmethod
    def latent_semantic_analysis(encoded, components: int) -> np.array:
        svd = TruncatedSVD(n_components=components, n_iter=10)
        normalizer = Normalizer(norm='l2', copy=False)
        lsa = make_pipeline(svd, normalizer)
        svd_result = lsa.fit_transform(encoded)
        print(f"Explained variance of the SVD step: {svd.explained_variance_ratio_.sum()}%")
        return svd_result

    @staticmethod
    def plot_lsa(svd_result: np.array, kmeans_model: KMeans, n_components: int) -> None:
        svd_df=pd.DataFrame(svd_result, columns=[f'Component {i + 1}' for i in range(n_components)])
        svd_df['cluster']=kmeans_model.labels_
        fig, axes=plt.subplots(1, 3, figsize=(22, 10))
        axes[0].set_title('Latent Semantic Analisys 1-2\n')
        axes[1].set_title('Latent Semantic Analisys 1-3\n')
        axes[2].set_title('Latent Semantic Analisys 2-3\n')
        sns.scatterplot(ax=axes[0], data=svd_df, x='Component 1', y='Component 2', hue='cluster', palette='viridis')
        sns.scatterplot(ax=axes[1], data=svd_df, x='Component 1', y='Component 3', hue='cluster', palette='viridis')
        sns.scatterplot(ax=axes[2], data=svd_df, x='Component 2', y='Component 3', hue='cluster', palette='viridis')
        plt.show()

    @staticmethod
    def umaps(data, k=None, dist=None) -> np.array:
        kk, ddist = 2, 0.1
        if k: kk = k
        if dist: ddist = dist
        reducer=umap.UMAP(random_state=42, n_neighbors=kk, min_dist=ddist, metric='euclidean', n_components=2)
        reducer.fit(data)
        return reducer.transform(data)

