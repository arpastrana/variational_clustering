from variational_clustering.datastructures import Queue

from variational_clustering.clustering import get_new_clusters
from variational_clustering.clustering import output_cls

from variational_clustering.operations import merge_split



__all__ = ['k_means']


def k_means(clusters, faces, iters, mergesplit=True, callback=None):
    """
    Find directional clusters using variational k-means.

    Parameters
    ----------
    clusters : `dict` of `Cluster`
        A dictionary with the initial clusters.
    faces : `dict` of `Face`
        A dictionary with the face objects to cluster.
    iters : `int`
        The number of iterations to run the algorithm for.
    mergesplit : `bool`. Optional
        A flag to activate the merge splitting operation for ambiguous clusters.
        Defaults to `True`.
    callback : `bool`. Optional
        A function to run at every iteration.

    Returns
    -------
    all_clusters : `list` of `dict`
        A list with all the clusters generated per iteration.

    Notes
    -----

    An outline of the process is as follows:

        1. Run for N given iterations only.
        2. Create Cluster Colors. Make global *counter*.
        3. Create Proxys with *get_proxy*
        4. Test whether it is the 1st iteration or not with global *counter*.
        5. If 1st, get proxies from clusters through from *get_proxy_seed*
        6. If not, proxies are the regions. Or the other way around.
        7. Build a queue with the seeds' adjacent faces through *build_queue*
        8. Grow up a cluster with *assign_to_region* method.
        9. Create new proxies from created regions with *grow_seeds* method.
        10. New proxies become the proxies.
        11. Repeat
    """

    all_clusters = []

    for it in range(iters):

        new_clusters, new_faces = get_new_clusters(clusters, faces)

        q = Queue(new_clusters, new_faces)
        q.init_queue()
        q.assign()
        clusters = q.get_clusters()
        all_clusters.append(output_cls(clusters))

        if mergesplit is True:
            if it < iters - 1:
                clusters = merge_split(clusters)

        if callback:
            callback(k=it, clusters=clusters)

    return all_clusters


if __name__ == "__main__":
    pass
