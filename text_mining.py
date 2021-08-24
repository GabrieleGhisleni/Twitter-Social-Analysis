from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer
from sklearn.pipeline import make_pipeline
from sklearn.cluster import KMeans
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from typing import Tuple
import seaborn as sns
import pandas as pd
import numpy as np
import umap


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
        fig, axes = plt.subplots(3, 1, figsize=(20, 20))
        features_names, i = self.tfid_vectorizer.get_feature_names(), 0
        fig.suptitle('Latent Dirichlet Allocation\n', fontsize=35)
        for topic_idx, topic in enumerate(model.components_):
            top_features_ind = topic.argsort()[:-n_top_words - 1:-1]
            top_features = [features_names[i] for i in top_features_ind]
            weights = topic[top_features_ind]
            sns.barplot(y=top_features, x=weights, ax=axes[i], palette='viridis')
            axes[i].tick_params(axis='y', which='minor', labelsize=7)
            axes[i].set_title(f'Topic Number {i+1}', fontsize=25)
            axes[i].set_xticks([])
            sns.set_style('white')
            i += 1

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
        cluster_model = KMeans(n_clusters=n_cluster, init='k-means++', max_iter=100, n_init=1)
        cluster_model.fit(reduced_data)
        return cluster_model

    def plot_umaps(self, data: list, k: list, n_cluster: int = 3) -> None:
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
        fig, axes=plt.subplots(1, 3, figsize=(23, 10))
        axes[0].set_title('Latent Semantic Analisys 1-2\n', fontsize=25)
        axes[1].set_title('Latent Semantic Analisys 1-3\n', fontsize=25)
        axes[2].set_title('Latent Semantic Analisys 2-3\n', fontsize=25)
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

    @staticmethod
    def word_cloud_create_and_show(data, title):
        fig=plt.figure(figsize=(20, 7))
        wordcloud = WordCloud(margin=0, background_color='white', colormap='inferno',
                              contour_width=10, contour_color='black', width=2000, height=1000)
        word_clouded=wordcloud.generate_from_frequencies(data)
        plt.imshow(word_clouded, interpolation='bilinear')
        plt.title(f'{title}\n', fontdict=dict(size=20, style='italic'))
        plt.axis("off")
        plt.show()

