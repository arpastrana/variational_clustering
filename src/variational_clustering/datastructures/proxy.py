import numpy as np
import compas.geometry as cg

from functools import reduce

from variational_clustering.helpers import align_vector


__all__ = ['Proxy', 'get_proxy', 'get_proxy_angle', 'get_proxy_vector', 'proxy_maker']


class Proxy(object):
	pass


def get_proxy(faces):
    # return get_proxy_angle(faces)
    return get_proxy_vector(faces)


def get_proxy_angle(faces):
    angles = [face.angle for face in faces]
    return proxy_maker(angles)


def get_proxy_vector(faces):
    w_ve = [face.vector for face in faces]
    w_ve = list(map(lambda x: align_vector(x, w_ve[0]), w_ve))
    r_ve = reduce(lambda x, y: cg.add_vectors(x, y), w_ve)
    return cg.normalize_vector(r_ve)


def proxy_maker(values):
    return np.mean(values)  # mean, median?


if __name__ == "__main__":
    pass
