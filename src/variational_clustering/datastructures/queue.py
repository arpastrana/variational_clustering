from heapq import heapify
from heapq import heappop
from heapq import heappush

from variational_clustering.datastructures import KeyDict


__all__ = ['Queue']


class Queue():
    def __init__(self, clusters, faces):
        self.clusters = clusters
        self.faces = faces
        self.queue = []
        heapify(self.queue)

    def init_queue(self):
        for c_key, cluster in self.clusters.items():
            cluster.add_seed(self.faces)
            cluster.set_proxy()
            n_faces = self.get_neighbour_faces(cluster.seed)
            self.update_queue(n_faces, c_key)

    def update_queue(self, n_faces, c_key):
        for f in n_faces:
            if f.cluster is None:
                error = f.get_error(self.clusters.get(c_key).proxy)
                entry = {'fkey': f.fkey, 'cluster': c_key}
                heappush(self.queue, KeyDict(error, entry))

    def assign(self):
        while len(self.queue) > 0:
            entry = heappop(self.queue)
            cu_f = entry.dct.get('fkey')
            face = self.faces.get(cu_f)

            if face.cluster is None:
                c_key = entry.dct.get('cluster')
                cluster = self.clusters.get(c_key)
                cluster.add_face(face)
                self.update_queue(self.get_neighbour_faces(cu_f), c_key)

    def get_neighbour_faces(self, f_key):
        for nf in self.faces.get(f_key).neighbours:
            yield self.faces.get(nf)

    def get_clusters(self):
        for ckey, cluster in self.clusters.items():
            cluster.set_proxy()
            cluster.set_distortion()
        return self.clusters


if __name__ == "__main__":
    pass
