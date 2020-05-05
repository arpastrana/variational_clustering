from itertools import combinations


__all__ = ["cluster_adjacency"]


def cluster_adjacency(clusters):
    comb = combinations(clusters.keys(), 2)

    adjacency = []
    for ka, kb in comb:
    	ca, cb = clusters.get(ka), clusters.get(kb)

    	if not is_adjacent_vertices(ca, cb):
    		continue

    	adjacency.append((ca, cb))

    return adjacency


def is_adjacent_halfedges(cluster_a, cluster_b):
	a = cluster_a.get_faces_halfedges()
	b = cluster_b.get_faces_halfedges()
	ints = len(a.intersection(b))
	return ints > 0  # if they share more than one edge

def is_adjacent_vertices(cluster_a, cluster_b):
	a = cluster_a.get_faces_vertices()
	b = cluster_b.get_faces_vertices()
	ints = len(a.intersection(b))	
	return ints > 1  # if they share more than two vertices together


if __name__ == "__main__":
    pass
