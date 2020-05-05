from variational_clustering.datastructures import Cluster


__all__ = ["get_cluster_to_split", "split_cluster"]


def get_cluster_to_split(clusters):
	"""
	finds cluster with largest error
	if not normalized, would that be the biggest cluster?
	"""
	return max(clusters.items(), key=lambda x: x[1].get_distortion())[1]


def split_cluster(s_cluster, clusters, new_id=None):
	new_fkey = s_cluster.get_worst_seed()
	s_cluster.remove_face(new_fkey)

	if new_id is None:
		new_id = max(clusters.keys()) + 1

	clusters[new_id] = Cluster(new_id, new_fkey)
	return clusters


if __name__ == "__main__":
	pass
