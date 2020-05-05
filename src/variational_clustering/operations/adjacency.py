from itertools import combinations


__all__ = ["cluster_adjacency", "is_adjacent"]


def cluster_adjacency(clusters):
    comb = combinations(range(len(clusters)), 2)
    cl_comb = [(clusters.get(x[0]), clusters.get(x[1])) for x in comb]
    return list(filter(is_adjacent, cl_comb))


def is_adjacent(cluster_pair):
    vert_1 = cluster_pair[0].get_faces_halfedges()
    vert_2 = cluster_pair[1].get_faces_halfedges()
    return len(vert_1.intersection(vert_2)) > 0


if __name__ == "__main__":
    pass
