from joblib import dump

import numpy as np

from sklearn.cluster import KMeans, MiniBatchKMeans

from ...utils.general_utils.utilities import generate_dir, save_objects
from ...utils.model_utils.clusterers import auto_elbow


def auto_kmeans(X, min_k, max_k, verbose=0, fast=True, save_name='',
                optimal_k=None, **kwargs):
    """
    Function for perfroming k-means partitioning. The number of k can be
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


def hierarchical_kmeans(X, min_k, max_k, tag, precomputed_root=None,
                        max_levels=2, verbose=0, **kwargs):
    """
    """
    clst_path = 'results\\saved_clusterers\\{}\\clusterer'.format(tag)
    labl_path = 'results\\saved_clusterers\\{}\\labels'.format(tag)
    generate_dir(clst_path)
    generate_dir(labl_path)
    hierarchy = {}
    for iter in range(max_levels):

        if verbose > 0:
            print('')
            print('Computing clusters for the {}th level'.format(iter))
            print('')
        if iter == 0 and precomputed_root is None:
            clusterer, labels, centroids = auto_kmeans(
                X=X,
                min_k=min_k,
                max_k=max_k,
                verbose=verbose,
                save_name='level_{}'.format(iter),
                **kwargs
            )
            dump(
                clusterer,
                '{}\\root_0_0.joblib'.format(clst_path)
            )
            hierarchy[iter] = {
                'labels': labels,
                'centroids': centroids
            }
        elif iter == 0 and precomputed_root is not None:
            hierarchy[iter] = {
                'labels': precomputed_root,
                'centroids': None
            }
        else:
            upstream = iter - 1
            unique_elements = np.unique(hierarchy[upstream]['labels'])
            unique_elements = unique_elements.flatten()
            total_centroids = []
            total_labels = np.zeros(shape=X.shape[0])
            for element in unique_elements:

                indices = np.argwhere(hierarchy[upstream]['labels'] == element)
                indices = indices.flatten()
                clusterer, labels, centroids = auto_kmeans(
                    X=X[indices],
                    min_k=min_k,
                    max_k=max_k,
                    verbose=verbose,
                    save_name='level_{}'.format(iter),
                    **kwargs
                )
                dump(
                    clusterer,
                    '{}\\lev_{}_{}.joblib'.format(clst_path, iter, element)
                )
                labels = labels + (total_labels.max() + 1)
                total_labels[indices] = labels
                total_centroids.append(centroids)

            total_centroids = np.vstack(total_centroids)
            hierarchy[iter] = {
                'labels': total_labels,
                'centroids': total_centroids
            }

    save_objects(
        objects={
            'hierar_labels_centr': hierarchy
        },
        dir_name='saved_clusterers\\{}\\labels'.format(tag)
    )
    return hierarchy
