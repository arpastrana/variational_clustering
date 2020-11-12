import math

from functools import reduce

from variational_clustering.datastructures import Proxy

from variational_clustering.operations import cluster_adjacency
from variational_clustering.operations import get_cluster_to_split
from variational_clustering.operations import split_cluster

from variational_clustering.metrics import errors
from variational_clustering.metrics import distortion


__all__ = ["get_clusters_to_merge", "simulate_merge", "merge_clusters", "merge_split", "execute_merge_split"]


def merge_split(clusters):
    adj_cl = cluster_adjacency(clusters)  # returns objects list
    to_merge = get_clusters_to_merge(adj_cl)  # returns objects tuple
    to_split = get_cluster_to_split(clusters)  # returns object single

    if execute_merge_split(to_merge, to_split):
        clusters, new_id = merge_clusters(to_merge, clusters)
        clusters = split_cluster(to_split, clusters, new_id)
    return clusters


def get_clusters_to_merge(adj_clusters):
    """
    get the adjacent cluster pair with that produces the smallest distortion
    after merging
    """
    return min(adj_clusters, key=lambda x: simulate_merge(x[0], x[1]))


def simulate_merge(cluster_1, cluster_2):
    """
    favors merging of two small clusters,
    as opposed of one big to absorb a small one?
    """
    _proxy = Proxy()
    t_faces = set(cluster_1.faces + cluster_2.faces)
    e = errors(t_faces, _proxy(t_faces))
    return distortion(e)


def execute_merge_split(t_m, t_s, threshold=0.5):
    """
    ???
    1. get the combined error of the two clusters to merge (a + b)
    2. get the error of the merged cluster (ab)
    3. calculate difference ab - (a + b)
    4. if the absolute value of the error difference is less than half
    the error of the worst error, return True
    5. otherwise, return False
    """
    # to_merge_err = reduce(lambda x, y: x+y, [x.get_distortion() for x in t_m])

    tma, tmb = t_m
    to_merge_err = tma.get_distortion() + tmb.get_distortion()
    
    merged_err = simulate_merge(tma, tmb)

    dif = merged_err - to_merge_err

    worst_err = t_s.get_distortion()

    # print()
    # print('combined error', to_merge_err)
    # print('simulated merge error', merged_err)
    # print('error difference', dif)
    # print('worst error', worst_err)

    if math.fabs(dif) < threshold * worst_err:  # 0.5, not squared
        # print('merge-split is True')
        return True

    # print('merge-split is False')
    return False


def merge_clusters(t_m, clusters):
    resilient = t_m[0]
    resilient.absorb_cluster(t_m[1])
    new_clusters = {v.id: v for k, v in clusters.items() if v.id != t_m[1].id}
    new_clusters[resilient.id] = resilient
    return new_clusters, t_m[1].id


if __name__ == "__main__":
    pass
