import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.preprocessing import StandardScaler


class KolamClusterEngine:
    """Handles feature scaling, dimensionality reduction (PCA & t-SNE),
    unsupervised clustering, and internal cluster validation for Kolam typologies.
    """

    def __init__(self, n_clusters=3, perplexity=15, random_state=42):
        self.n_clusters = n_clusters
        self.base_perplexity = perplexity
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=2, random_state=random_state)
        self.tsne = TSNE(
            n_components=2,
            perplexity=perplexity,
            random_state=random_state,
            init="pca",
            learning_rate="auto",
        )
        self.kmeans = KMeans(
            n_clusters=n_clusters, random_state=random_state, n_init=10
        )
        self._last_X_scaled = None
        self._last_labels = None

    def fit_transform(self, feature_df):
        """Processes extracted feature DataFrame and returns dimensionality-reduced
        coordinates alongside cluster assignments.
        """
        numeric_cols = feature_df.select_dtypes(include=[np.number]).columns
        X = feature_df[numeric_cols]

        # 1. Scale features
        X_scaled = self.scaler.fit_transform(X)
        self._last_X_scaled = X_scaled

        # 2. PCA transformation
        pca_coords = self.pca.fit_transform(X_scaled)

        # 3. t-SNE transformation (adapt perplexity safely for small sample counts)
        if len(feature_df) <= self.base_perplexity:
            self.tsne.perplexity = max(2, min(len(feature_df) - 1, self.base_perplexity))
        else:
            self.tsne.perplexity = self.base_perplexity
        tsne_coords = self.tsne.fit_transform(X_scaled)

        # 4. K-Means clustering
        cluster_labels = self.kmeans.fit_predict(X_scaled)
        self._last_labels = cluster_labels

        results_df = feature_df.copy()
        results_df["cluster"] = [f"Typology {label+1}" for label in cluster_labels]
        results_df["pca_x"] = pca_coords[:, 0]
        results_df["pca_y"] = pca_coords[:, 1]
        results_df["tsne_x"] = tsne_coords[:, 0]
        results_df["tsne_y"] = tsne_coords[:, 1]

        return results_df

    def get_variance_ratio(self):
        """Returns explained variance ratio of top 2 principal components."""
        return self.pca.explained_variance_ratio_

    def get_validation_metrics(self):
        """Computes Silhouette Score and Davies-Bouldin Index on the scaled features."""
        if self._last_X_scaled is None or self._last_labels is None:
            return {"silhouette": None, "davies_bouldin": None}

        unique_labels = np.unique(self._last_labels)
        n_samples = len(self._last_labels)
        if len(unique_labels) < 2 or n_samples <= len(unique_labels):
            return {"silhouette": None, "davies_bouldin": None}

        sil = float(silhouette_score(self._last_X_scaled, self._last_labels))
        dbi = float(davies_bouldin_score(self._last_X_scaled, self._last_labels))
        return {"silhouette": sil, "davies_bouldin": dbi}


if __name__ == "__main__":
    print("Clustering engine ready. Import and use KolamClusterEngine.")
