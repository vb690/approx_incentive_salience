from sklearn.cluster import KMeans, MiniBatchKMeans

from ...utils.model_utils.clusterers import auto_elbow


def auto_kmeans(X, min_k, max_k, verbose=0, fast=True, save_name='',
                optimal_k=None, **kwargs):
    """Function for perfroming k-means partitioning. The number of k can be
    manually set or it will be chosen using the auto_elbow function.
    """
    # decide weather to use faster but less accurate mini batch K-means
    if fast:
        kmeans = MiniBatchKMeans
    else:
        kmeans = KMeans

    # if an optimal k value is not defined we test a range of values
    # and chose the one that maximize the curvature of the inertia with
    # respect to the k value
    if optimal_k is None:
        inertias = []
        n_clusters = [i for i in range(min_k, max_k + 1)]
        for k in n_clusters:

            clusterer = kmeans(
                n_clusters=k,
                **kwargs
            )
            clusterer.fit(X)
            inertias.append(clusterer.inertia_)

        optimal_k = auto_elbow(
            n_clusters=n_clusters,
            inertias=inertias,
            verbose=verbose,
            save_name=save_name
        )
    else:
        optimal_k = optimal_k

    # we fit the again the k-means but using the optimal k value
    clusterer = kmeans(
        n_clusters=optimal_k,
        **kwargs
    )
    clusterer.fit(X)
    labels = clusterer.labels_
    centroids = clusterer.cluster_centers_
    return clusterer, labels, centroids
