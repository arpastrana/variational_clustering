import math

from functools import reduce

from variational_clustering.datastructures import get_proxy

from variational_clustering.operations import cluster_adjacency
from variational_clustering.operations import get_cluster_to_split
from variational_clustering.operations import split_cluster

from variational_clustering.metrics import errors
from variational_clustering.metrics import distortion


__all__ = ["get_clusters_to_merge", "simulate_merge", "merge_clusters", "merge_split", "execute_merge_split"]


def get_clusters_to_merge(adj_clusters):
    return min(adj_clusters, key=lambda x: simulate_merge(x[0], x[1]))


def simulate_merge(cluster_1, cluster_2):
    t_faces = set(cluster_1.faces + cluster_2.faces)
    e = errors(t_faces, get_proxy(t_faces))
    return distortion(e)


def merge_clusters(t_m, clusters):
    resilient = t_m[0]
    resilient.absorb_cluster(t_m[1])
    new_clusters = {v.id: v for k, v in clusters.items() if v.id != t_m[1].id}
    new_clusters[resilient.id] = resilient
    return new_clusters, t_m[1].id


def merge_split(clusters):
    adj_cl = cluster_adjacency(clusters)  # returns objects list
    to_merge = get_clusters_to_merge(adj_cl)  # returns objects tuple
    to_split = get_cluster_to_split(clusters)  # returns object single

    if execute_merge_split(to_merge, to_split) is True:
        clusters, new_id = merge_clusters(to_merge, clusters)
        clusters = split_cluster(to_split, clusters, new_id)
    return clusters


def execute_merge_split(t_m, t_s):
    to_merge_err = reduce(lambda x, y: x+y, [x.get_distortion() for x in t_m])
    merged_err = simulate_merge(t_m[0], t_m[1])
    dif = merged_err - to_merge_err
    worst_err = t_s.get_distortion()

    if math.fabs(dif) < 0.50 * worst_err:  # 0.5, not squared
        print('merge-split is True')
        return True

    else:
        print('merge-split is False')
        return False


if __name__ == "__main__":
    pass
