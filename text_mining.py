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

class TextMining:
    def __init__(self, ngram_range: int = 1):
        self.tfid_vectorizer = TfidfVectorizer(ngram_range=(1, ngram_range), norm='l2')
        self.count_vectorizer = CountVectorizer()


    def vectorized_text(self, text_to_vectorize) -> list:
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
        i = 0
        for topic_idx, topic in enumerate(model.components_):
            top_features_ind=topic.argsort()[:-n_top_words - 1:-1]
            top_features=[self.tfid_vectorizer.get_feature_names()[i] for i in top_features_ind]
            weights=topic[top_features_ind]
            sns.barplot(y=top_features, x=weights, ax=axes[i])
            i += 1


    def latent_semantic_analysis(self, encoded, components: int):
        svd = TruncatedSVD(n_components=components, n_iter=10)
        normalizer = Normalizer(norm='l2', copy=False)
        lsa = make_pipeline(svd, normalizer)
        svd_result = lsa.fit_transform(encoded)
        print(f"Explained variance of the SVD step: {svd.explained_variance_ratio_.sum()}%")
        return svd_result

    def clustering_over_lsa(self, reduced_data, n_cluster) -> KMeans:
        cluster_model = KMeans(n_clusters=n_cluster, init='k-means++', max_iter=100, n_init=1)
        cluster_model.fit(reduced_data)
        return cluster_model

    def plot_lsa(self, svd_result, kmeans_model):
        svd_df = pd.DataFrame(svd_result)
        svd_df['cluster'] = kmeans_model.labels_
        sns.set_style('white')
        fig, axes=plt.subplots(1, 3, figsize=(20, 10))
        sns.scatterplot(ax=axes[0], data=svd_df, x=0, y=1, hue='cluster', palette='viridis')
        sns.scatterplot(ax=axes[1], data=svd_df, x=0, y=2, hue='cluster', palette='viridis')
        sns.scatterplot(ax=axes[2], data=svd_df, x=2, y=1, hue='cluster', palette='viridis')
        plt.show()

    def umaps(self, data, k=None, dist=None):
        if k: reducer = umap.UMAP(random_state=42, n_neighbors=k, metric='euclidean', n_components=2)
        else: reducer = umap.UMAP(random_state=42, min_dist=dist,  metric='euclidean', n_components=2)
        reducer.fit(data)
        return reducer, reducer.transform(data)
    #
    # def plot_umaps(self, data):
    #     from umap.plot import connectivity
    #     ax =  connectivity(reducer,
    #                        theme='fire',
    #                        width=800, height=500)
    #     ax.set_title("Graph oriented")
    #     plt.show()
