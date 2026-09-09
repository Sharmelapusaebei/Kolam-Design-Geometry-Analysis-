import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler


class KolamClusterEngine:
    """Handles feature scaling, dimensionality reduction (PCA & t-SNE),

    and unsupervised clustering for Kolam geometric typologies.
    """

    def __init__(self, n_clusters=3, random_state=42):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=2, random_state=random_state)
        self.tsne = TSNE(
            n_components=2,
            perplexity=15,
            random_state=random_state,
            init="pca",
            learning_rate="auto",
        )
        self.kmeans = KMeans(
            n_clusters=n_clusters, random_state=random_state, n_init=10
        )

    def fit_transform(self, feature_df):
        """Processes extracted feature DataFrame and returns dimensionality-reduced

        coordinates alongside cluster assignments.
        """
        # Separate metadata (like image paths/names) from numerical features
        numeric_cols = feature_df.select_dtypes(include=[np.number]).columns
        X = feature_df[numeric_cols]

        # 1. Scale Features
        X_scaled = self.scaler.fit_transform(X)

        # 2. PCA Transformation (preserves variance drivers)
        pca_coords = self.pca.fit_transform(X_scaled)

        # 3. t-SNE Transformation (captures local non-linear neighborhood structures)
        # Adjust perplexity automatically if sample size is very small
        if len(feature_df) <= 15:
            self.tsne.perplexity = max(2, len(feature_df) - 1)
        tsne_coords = self.tsne.fit_transform(X_scaled)

        # 4. K-Means Clustering for Typology Discovery
        cluster_labels = self.kmeans.fit_predict(X_scaled)

        # Build output result DataFrame
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


if __name__ == "__main__":
    print("Clustering engine ready. Import and use KolamClusterEngine.")
