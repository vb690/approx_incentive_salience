import numpy as np

from kneed import KneeLocator

from ..general_utils.visualizers import visualize_auto_elbow


def auto_elbow(n_clusters, inertias, save_name, verbose=0):
    """Function for individuating the elbow in an elbow plot.
    The elbow is set as the number of centroids maximizing the distance
    to the gradient line.

    Args:
        - n_clusters:
        - inertias:
        - save_name:
        - verbose:

    Returns:
        - optimal_k:
    """
    y = (inertias[0], inertias[-1])
    x = (n_clusters[0], n_clusters[-1])

    kneedle = KneeLocator(
        n_clusters,
        inertias,
        S=1.0,
        curve='convex',
        direction='decreasing'
    )
    print(n_clusters)
    print(inertias)
    print(kneedle.knee)

    alpha, beta = np.polyfit(
        x,
        y,
        1
    )
    grad_line = [beta+(alpha*k) for k in n_clusters]
    optimal_k = np.argmax([grad - i for grad, i in zip(grad_line, inertias)])

    print(optimal_k)
    optimal_k = n_clusters[optimal_k]
    if verbose > 0:
        print('Optimal k found at {}'.format(optimal_k))
        visualize_auto_elbow(
            n_clusters=n_clusters,
            inertias=inertias,
            grad_line=grad_line,
            optimal_k=optimal_k,
            save_name=save_name
        )
    return optimal_k
