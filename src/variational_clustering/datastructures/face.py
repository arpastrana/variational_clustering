import math
import compas.geometry as cg

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

    def set_vector(self, vector):
        self.vector = cg.normalize_vector(vector)
        self.vector_length = cg.length_vector(vector)

    def set_weighed_vector(self, area_weight=False):
        if area_weight is True:
            self.w_vector = cg.scale_vector(self.vector, self.area)
        else:
            self.w_vector = self.vector

    def set_angle(self, vector):
        angle = cg.angle_vectors([1.0, 0.0, 0.0], vector, deg=True)
        if angle > 90.0:
            angle = 180.0 - angle
        self.angle = angle

    def set_area(self, area):
        self.area = area

    def set_neighbours(self, neighbours):
        self.neighbours = [n for n in neighbours if n is not None]

    def get_error(self, proxy):
        func = self.get_error_vector
        # func = self.get_error_angle
        return func(proxy)

    def get_error_vector(self, proxy, area_weight=False):
        ali_vec = align_vector(self.vector, proxy)
        difference = cg.subtract_vectors(ali_vec, proxy)
        error = cg.length_vector_sqrd(difference)  # original

        # error = cg.length_vector(difference)
        # if area_weight is True:
        #     error = self.area * cg.length_vector_sqrd(difference)

        # do something about weights
        # w_1 = 0.3
        # w_2 = 0.7
        # w_error = w_1 * error + w_2 * self.vector_length
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
