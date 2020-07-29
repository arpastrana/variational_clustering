import math
import compas.geometry as cg

from compas.utilities import geometric_key

from variational_clustering.helpers import align_vector


__all__ = ['Face']


class Face():
    def __init__(self, fkey):
        self.fkey = fkey
        self.vertices = None
        self.halfedges = None

        self.vector = None
        self.vector_length = None
        self.w_vector = None
        self.area = None
        self.neighbours = None
        self.error = None
        self.angle = None

        self.cluster = None

    def set_vertices(self, vertices):
        self.vertices = vertices

    def set_halfedges(self, halfedges):
        self.halfedges = halfedges

    def set_cluster(self, cluster):
        self.cluster = cluster

    def set_area(self, area):
        self.area = area

    def set_vector(self, vector):
        self.vector = cg.normalize_vector(vector)
        self.vector_length = cg.length_vector(vector)

    def set_weighed_vector(self, vector):
        vector = cg.normalize_vector(vector)
        self.w_vector = cg.scale_vector(vector, self.area)

    def set_angle(self, vector):
        angle = cg.angle_vectors([1.0, 0.0, 0.0], vector, deg=True)
        if angle > 90.0:
            angle = 180.0 - angle
        self.angle = angle

    def set_neighbours(self, neighbours):
        self.neighbours = [n for n in neighbours if n is not None]

    def set_coordinates(self, coordinates):
        self.coordinates = coordinates

    def set_centroid(self, centroid):
        self.centroid = centroid

    def get_error(self, proxy):
        func = self.get_error_vector
        # func = self.get_error_angle
        return func(proxy)

    def get_error_vector(self, proxy):
        vector = self.vector

        ali_vec = align_vector(vector, proxy)
        difference = cg.subtract_vectors(ali_vec, proxy)
        error = cg.length_vector_sqrd(difference)

        error *= self.area

        return error

    def get_error_angle(self, proxy):
        error = math.fabs(self.angle - proxy)
        if error > 90.0:
            error = math.fabs(180.0 - error)
        if error > 45:
            error = math.fabs(90.0 - error)
        return error

    def set_error(self, proxy):  # NEW
        self.error = self.get_error(proxy)

    def clear_cluster(self):
        self.cluster = None


if __name__ == "__main__":
    pass
