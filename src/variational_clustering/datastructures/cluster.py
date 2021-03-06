import compas.geometry as cg

from functools import reduce

from compas.utilities import geometric_key

from variational_clustering.datastructures import Proxy
from variational_clustering.metrics import distortion

from variational_clustering.helpers import align_vector


__all__ = ['Cluster']


class Cluster():
    def __init__(self, id_, seed_face):
        self.id = id_
        self.seed = seed_face

        self.faces = []
        self.faces_keys = []

        self.proxy = None
        self.distortion = None
        self.add_face_key(self.seed)

    def remove_face(self, fkey):
        self.faces_keys = [k for k in self.faces_keys if k != int(fkey)]
        self.faces = [f for f in self.faces if f.fkey != fkey]
        self.set_proxy()

    def absorb_cluster(self, other_cluster):
        for o_face in other_cluster.faces:
            self.add_face(o_face)

        self.set_proxy()
        self.set_faces_in_cluster()

    def copy_cluster(self, other_cluster):
        self.faces_keys = list(set(other_cluster.faces_keys))
        self.proxy = other_cluster.proxy
        self.distortion = other_cluster.distortion

    def add_face_key(self, f_key):
        if f_key not in self.faces_keys:
            self.faces_keys.append(f_key)

    def add_seed(self, faces):
        seed_face = faces.get(self.seed)
        seed_face.set_cluster(self.id)
        self.faces.append(seed_face)

    def add_face(self, face):
        if face.fkey not in self.faces_keys:
            face.set_cluster(self.id)
            self.faces_keys.append(face.fkey)
            self.faces.append(face)

    def harvest_faces(self, faces):
        for key in self.faces_keys:
            self.faces.append(faces.get(key))

    def get_vectors(self):
        # return [face.w_vector for face in self.faces]
        return [face.vector for face in self.faces]

    def get_angles(self):
        return [face.angle for face in self.faces]        

    def set_faces_in_cluster(self):
        for face in self.faces:
            face.cluster = self.id

    def set_proxy(self):
        _proxy = Proxy()
        self.proxy = _proxy(self.faces)

    def get_errors(self):
        return [face.get_error(self.proxy) for face in self.faces]

    def get_best_new_seed(self):
        best_face = min(self.faces, key=lambda x: x.get_error(self.proxy))
        return best_face.fkey

    def get_worst_seed(self):
        worst_face = max(self.faces, key=lambda x: x.get_error(self.proxy))
        return worst_face.fkey

    def get_face_keys(self):
        return [f.fkey for f in self.faces]

    def get_faces_halfedges(self):
        face_halfedges = set()
        for f in self.faces:
            face_halfedges.update(f.halfedges)
        return face_halfedges

    def get_faces_vertices(self):
        face_vertices = set()
        for face in self.faces:
            face_vertices.update(face.vertices)
        return face_vertices

    def clear_faces(self):
        for face in self.faces:
            face.clear_cluster()
        self.faces[:] = []

    def get_distortion(self):
        return distortion(self.get_errors())

    def set_distortion(self):
        self.distortion = self.get_distortion()

    def __repr__(self):
        f = len(self.faces)
        fk = len(self.faces_keys)
        s = self.seed
        dst = self.distortion
        return 'id:{0} seed:{1} distortion:{4} faces:{2}, keys:{3}'.format(self.id, s, f, fk, dst)


if __name__ == "__main__":
    cluster = Cluster(0, 0)
    print(cluster)
