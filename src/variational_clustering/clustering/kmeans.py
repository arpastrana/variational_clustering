from variational_clustering.datastructures import Queue

from variational_clustering.clustering import get_new_clusters
from variational_clustering.clustering import output_cls

from variational_clustering.operations import merge_split



__all__ = ['k_means']


def k_means(clusters, faces, iters, mergesplit=True, callback=None):
    '''
    2. Run for N given iterations only.
    1. Create Cluster Colors. Make global *counter*.
    2B. Create Proxys with *get_proxy*
    3. Test whether it is the 1st iteration or not with global *counter*.
    4. If 1st, get proxies from clusters through from *get_proxy_seed*
    4B. If not, proxies are the regions. Or the other way around.
    5. Build a queue with the seeds' adjacent faces through *build_queue*
    6. Grow up a cluster with *assign_to_region* method.
    7. Create new proxies from created regions with *grow_seeds* method.
    8. New proxies become the proxies.
    9. Repeat
    '''
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
