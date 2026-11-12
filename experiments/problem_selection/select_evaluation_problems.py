"""
Select one benchmark problem instance per ELA cluster: the evaluation set in
src/edaptive/problems/custom_test_set.py.

Follows the instance selection using clustering in Cenikj et al. (2022),
"SELECTOR: Selecting a Representative Benchmark Suite for Reproducible
Statistical Comparison", GECCO '22 (Sections 3.1, 3.2 and 4.3):

1. Each problem instance is represented by the median of its ELA features
   over the 30 feature-calculation runs (ic.eps.s excluded).
2. Agglomerative hierarchical clustering with cosine distance partitions the
   instances into 12 clusters.
3. The largest cluster is clustered again into 10 subclusters, giving 21
   clusters in total.
4. Each cluster is represented by the instance whose features are most
   similar (cosine) to the cluster centroid, the mean of each feature over
   the cluster's instances. Problems in ela_benchmark_suite.py are not
   eligible, so the selected suite shares no problems with it.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity

# ELA features of each problem instance (10-D), one row per feature-calculation
# run.
ELA_CSV = Path(__file__).resolve().parent / "ela_features.csv"

# Columns that identify a problem instance.
PROBLEM_COLS = ["suite", "dimensions", "fid", "iid"]

# Columns that are not ELA features. "cluster" holds labels from an
# earlier clustering.
METADATA_COLS = ["Unnamed: 0", "run_id", "cluster", *PROBLEM_COLS]

# ic.eps.s is excluded because it is undefined for some instances.
EXCLUDED_FEATURES = ["ic.eps.s"]

# The paper chose 12 clusters from the silhouette score and the need for at
# least 10 instances in the statistical analysis, and 10 subclusters of the
# largest cluster by inspecting the silhouette scores. The silhouette score
# alone peaks at 2 for both, so these are fixed; the curves are printed for
# reference.
N_CLUSTERS = 12
N_SUBCLUSTERS = 10
SILHOUETTE_CLUSTER_COUNTS = range(2, 21)

# Problems in ela_benchmark_suite.py, as (suite, fid, iid).
ELA_BENCHMARK_PROBLEMS = {
    ("bbob", 2, 1),
    ("bbob", 6, 1),
    ("bbob", 16, 1),
    ("bbob", 17, 2),
}

# Seed for the t-SNE projection used in the plot.
TSNE_SEED = 136


def load_median_features(path):
    """
    Return one row per problem instance with the median of each ELA feature
    over the runs, and the list of feature columns.
    """
    df = pd.read_csv(path)

    # Rows with suite "all" repeat the instances of the individual suites.
    df = df[df["suite"] != "all"]

    feature_cols = [
        col
        for col in df.columns
        if col not in METADATA_COLS and col not in EXCLUDED_FEATURES
    ]

    instances = df.groupby(PROBLEM_COLS)[feature_cols].median().reset_index()

    if not np.isfinite(instances[feature_cols].to_numpy()).all():
        raise ValueError("ELA features contain missing or infinite values")

    return instances, feature_cols


def problem_name(instance):
    """Class name of the problem in the edaptive problem suites."""
    if instance["suite"] == "bbob":
        return f"BBOB_FID{instance['fid']}_IID{instance['iid']}"

    year = instance["suite"].removeprefix("cec")
    return f"CEC_{year}_FID{instance['fid']}"


def agglomerative_clustering(X, n_clusters):
    """Hierarchical clustering with cosine distance and complete linkage."""
    return AgglomerativeClustering(
        n_clusters=n_clusters,
        metric="cosine",
        linkage="complete",
    ).fit_predict(X)


def silhouette_curve(X, cluster_counts):
    """Cosine silhouette score of the clustering for each number of clusters."""
    return pd.Series(
        {
            n_clusters: silhouette_score(
                X,
                agglomerative_clustering(X, n_clusters),
                metric="cosine",
            )
            for n_clusters in cluster_counts
        },
        name="silhouette",
    )


def cluster_instances(X):
    """
    Cluster the instances into N_CLUSTERS clusters, then split the largest
    into N_SUBCLUSTERS subclusters.

    The initial clusters keep their labels 0 ... N_CLUSTERS - 1, except the
    largest, whose instances get the subcluster labels
    N_CLUSTERS ... N_CLUSTERS + N_SUBCLUSTERS - 1.

    Returns the final labels and the mask of the largest initial cluster.
    """
    initial_labels = agglomerative_clustering(X, N_CLUSTERS)

    largest_mask = initial_labels == np.bincount(initial_labels).argmax()

    sub_labels = agglomerative_clustering(X[largest_mask], N_SUBCLUSTERS)

    labels = initial_labels.copy()
    labels[largest_mask] = N_CLUSTERS + sub_labels

    return labels, largest_mask


def select_representatives(instances, X, labels):
    """
    Return the instance closest (cosine) to each cluster's centroid,
    skipping problems in ela_benchmark_suite.py, ordered by cluster.
    """
    eligible = np.array(
        [
            (suite, fid, iid) not in ELA_BENCHMARK_PROBLEMS
            for suite, fid, iid in zip(
                instances["suite"], instances["fid"], instances["iid"], strict=True
            )
        ]
    )

    selected = []

    for cluster in np.unique(labels):
        in_cluster = labels == cluster

        centroid = X[in_cluster].mean(axis=0, keepdims=True)
        similarity = cosine_similarity(X, centroid).ravel()

        candidates = np.flatnonzero(in_cluster & eligible)

        if len(candidates) == 0:
            raise ValueError(
                f"Cluster {cluster} only contains problems from ela_benchmark_suite.py"
            )

        selected.append(candidates[np.argmax(similarity[candidates])])

    representatives = instances.iloc[selected].copy()
    representatives["cluster"] = labels[selected]

    return representatives.reset_index(drop=True)


def plot_tsne(X, labels):
    """Plot the clusters on a 2-D t-SNE projection with cosine distance."""
    X_tsne = TSNE(
        n_components=2,
        metric="cosine",
        init="random",
        learning_rate="auto",
        random_state=np.random.RandomState(TSNE_SEED),
    ).fit_transform(X)

    plt.figure(figsize=(12, 9))

    for cluster in np.unique(labels):
        in_cluster = labels == cluster

        plt.scatter(
            X_tsne[in_cluster, 0],
            X_tsne[in_cluster, 1],
            label=f"Cluster {cluster}",
            alpha=0.75,
            s=50,
        )

    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.title("t-SNE Visualization of ELA Feature Clusters")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.show()


def main():
    instances, feature_cols = load_median_features(ELA_CSV)
    X = instances[feature_cols].to_numpy()

    labels, largest_mask = cluster_instances(X)
    instances["cluster"] = labels

    print("Silhouette scores of the initial clustering:")
    print(silhouette_curve(X, SILHOUETTE_CLUSTER_COUNTS))

    print("\nSilhouette scores of the subclustering of the largest cluster:")
    print(silhouette_curve(X[largest_mask], SILHOUETTE_CLUSTER_COUNTS))

    print("\nInstances per suite in each cluster:")
    print(
        pd.crosstab(
            instances["cluster"],
            instances["suite"],
            margins=True,
        )
    )

    representatives = select_representatives(instances, X, labels)

    print("\nSelected problem per cluster:")
    for _, instance in representatives.iterrows():
        print(f"{instance['cluster']:>3}  {problem_name(instance)}")

    plot_tsne(X, labels)


if __name__ == "__main__":
    main()
