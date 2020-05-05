from variational_clustering.datastructures import Cluster


__all__ = ["get_cluster_to_split", "split_cluster"]


def get_cluster_to_split(clusters):
    return max(clusters.items(), key=lambda x: x[1].get_distortion())[1]


def split_cluster(s_cluster, clusters, new_id=None):
    new_fkey = s_cluster.get_worst_seed()
    s_cluster.remove_face(new_fkey)

    if new_id is None:
        new_id = max(clusters.keys()) + 1

    clusters[new_id] = Cluster(new_id, new_fkey)
    clusters[s_cluster.id] = s_cluster
    return clusters


if __name__ == "__main__":
    pass
