from variational_clustering.datastructures import Cluster
from variational_clustering.datastructures import Queue
from variational_clustering.datastructures import Face

from variational_clustering.operations import get_cluster_to_split
from variational_clustering.operations import split_cluster


__all__ = ["furthest_init", "output_cls", "get_new_clusters", "clear_clusters", "make_faces"]


def furthest_init(num, faces, callback=None):  # no dependency

    seed = min(list(faces.keys()))
    clusters = {0: Cluster(id_=0, seed_face=seed)}
    all_clusters = []

    for i in range(num):
        new_clusters, new_faces = get_new_clusters(clusters, faces)

        q = Queue(new_clusters, new_faces)
        q.init_queue()
        q.assign()
        clusters = q.get_clusters()
        all_clusters.append(output_cls(clusters))

        if i < num-1:
            t_s = get_cluster_to_split(clusters)
            clusters = split_cluster(t_s, clusters)

        if callback:
            callback(k=i, clusters=clusters)

    return all_clusters


def output_cls(clusters):
    new_clusters = {}
    for c_key, cluster in clusters.items():
        new_cl = Cluster(cluster.id, cluster.seed)
        new_cl.copy_cluster(cluster)
        new_clusters[c_key] = new_cl
    return new_clusters


def get_new_clusters(clusters, faces):
    n_clusters = {}

    for key, cluster in clusters.items():
        cluster.harvest_faces(faces)
        cluster.set_proxy()

        n_cluster = Cluster(cluster.id, cluster.get_best_new_seed())
        n_clusters[n_cluster.id] = n_cluster
        cluster.clear_faces()  # clears out cluster and face relationship

    return n_clusters, faces


def clear_clusters(faces):
    for f_key, face in faces.items():
        face.clear_cluster()


def make_faces(mesh, vectors):  # no dep
    faces = {}

    for f_key in mesh.faces():
        face = Face(f_key)

        vector = vectors[f_key]
        halfedges = mesh.face_halfedges(f_key)

        face.set_halfedges([tuple(sorted(h)) for h in halfedges])
        face.set_vertices(mesh.face_vertices(f_key))
        face.set_area(mesh.face_area(f_key))
        face.set_neighbours(mesh.face_neighbors(f_key))

        face.set_vector(vector)
        face.set_weighed_vector(vector)
        face.set_angle(vector)

        face.set_coordinates(mesh.face_coordinates(f_key))
        face.set_centroid(mesh.face_centroid(f_key))

        faces[f_key] = face

    return faces


if __name__ == "__main__":
    pass

