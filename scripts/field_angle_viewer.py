import math
import numpy as np

from random import choice
from time import time
from functools import partial

from compas.geometry import scale_vector
from compas.geometry import add_vectors
from compas.geometry import normalize_vector
from compas.geometry import length_vector
from compas.geometry import cross_vectors
from compas.geometry import angle_vectors

from compas.datastructures import Mesh
from compas.datastructures import mesh_unify_cycles

from compas.utilities import i_to_rgb
from compas.utilities import i_to_black
from compas.utilities import i_to_red
from compas.utilities import i_to_blue

from compas_plotters import MeshPlotter

from variational_clustering.clustering import make_faces
from variational_clustering.clustering import furthest_init
from variational_clustering.clustering import k_means
from variational_clustering.helpers import align_vector


def vector_lines_on_faces(mesh, vector_tag, uniform=True, factor=0.02):
    '''
    '''
    def line_sdl(start, direction, length):
        direction = normalize_vector(direction[:])
        a = add_vectors(start, scale_vector(direction, -length))
        b = add_vectors(start, scale_vector(direction, +length))

        return a, b

    lines = []
    for fkey, attr in mesh.faces(data=True):
        vector = attr.get(vector_tag)

        if not vector:
            raise ValueError('Vector {} not defined on face {}'.format(vector_tag, fkey))

        if uniform:
            vec_length = factor
        else:
            vec_length = length_vector(vector) * factor

        pt = mesh.face_centroid(fkey)
        lines.append(line_sdl(pt, vector, vec_length))
    return lines


def line_tuple_to_dict(line):
    '''
    '''
    a, b = line
    return {'start': a, 'end': b}


if __name__ == '__main__':

    # ==========================================================================
    # Constants
    # ==========================================================================

    # HERE = '../data/json_files/four_point_slab.json'  # interesting
    HERE = '../data/json_files/perimeter_supported_slab.json' # interesting
    # HERE = '../data/json_files/topleft_supported_slab.json'  # interesting

    # HERE = '../data/json_files/leftright_supported_slab.json'  # interesting

    # HERE = '../data/json_files/bottomleftright_supported_slab.json'  
    # HERE = '../data/json_files/middle_supported_slab_cantilever.json'


    tags = [
        'n_1',
        'n_2',
        'm_1',
        'm_2',
        'ps_1_top',
        'ps_1_bot',
        'ps_1_mid',
        'ps_2_top',
        'ps_2_bot',
        'ps_2_mid',
        'custom_1',
        'custom_2'
        ]

    vector_tag = 'ps_1_top'  # ps_1_top
    vector_tag_display = 'ps_2_top'  # ps_1_top
    smooth_iters = 20

    # ==========================================================================
    # Import mesh
    # ==========================================================================

    mesh = Mesh.from_json(HERE)
    mesh_unify_cycles(mesh)

    # ==========================================================================
    # 45 degrees field
    # ==========================================================================

    for fkey, attr in mesh.faces(True):
        vec_1 = attr[vector_tag]
        y = 1.0 / math.tan(math.radians(45.0))
        x_vec = vec_1
        y_vec = cross_vectors(x_vec, [0.0, 0.0, 1.0])  # global Z
        y_vec = scale_vector(y_vec, y)
        vec_3 = normalize_vector(add_vectors(x_vec, y_vec))
        
        mesh.face_attribute(fkey, name=vector_tag, value=vec_3)

    # ==========================================================================
    # Average smooth vector field
    # ==========================================================================

    # for _ in range(smooth_iters):
    #     averaged_vectors = {}
    #     for fkey in mesh.faces():
    #         nbrs = mesh.face_neighbors(fkey)
    #         vectors = mesh.faces_attribute(keys=nbrs, name=vector_tag)
    #         vectors.append(mesh.face_attribute(fkey, name=vector_tag))

    #         vectors = list(map(lambda x: align_vector(x, vectors[0]), vectors))

    #         vectors = np.array(vectors)
    #         avg_vector = np.mean(vectors, axis=0).tolist()
    #         averaged_vectors[fkey] = avg_vector

    #     for fkey in mesh.faces():
    #         mesh.face_attribute(fkey, name=vector_tag, value=averaged_vectors[fkey])

    # ==========================================================================
    # Process PS vectors
    # ==========================================================================

    angles = {}
    centroids = {}
    for fkey, attr in mesh.faces(data=True):
        vector = attr.get(vector_tag)
        # angle = angle_vectors([1.0, 0.0, 0.0], vector, deg=True)
        angle = angle_vectors([1.0, 1.0, 0.0], vector, deg=True)
        angles[fkey] = angle
        centroids[fkey] = np.array(mesh.face_centroid(fkey))


    print('max angle', min(angles.values()))
    print('min angle', max(angles.values()))

    for idx, angle in angles.items():
        if angle >= 90.0:
            angle = 180.0 - angle
        angles[idx] = angle

    for idx, angle in angles.items():
        if angle >= 45:
            angle = 90.0 - angle
        angles[idx] = angle

    print('max angle', min(angles.values()))
    print('min angle', max(angles.values()))

    # ==========================================================================
    # Average smooth angles
    # ==========================================================================

    for _ in range(smooth_iters):
        averaged_angles = {}

        for fkey in mesh.faces():
            nbrs = mesh.face_neighbors(fkey)
            nbrs.append(fkey)
            local_angles = [angles[key] for key in nbrs]
            
            averaged_angles[fkey] = np.mean(np.array(local_angles))

        for fkey in mesh.faces():
            angles[fkey] = averaged_angles[fkey]

    # ==========================================================================
    # Colors
    # ==========================================================================

    anglemax = max(angles.values())
    colors = {}
    for idx, angle in angles.items():
        color = i_to_rgb(angle / anglemax)
        colors[idx] = color

    # ==========================================================================
    # Create PS vector lines
    # ==========================================================================

    lines = vector_lines_on_faces(mesh, vector_tag_display, True, factor=0.05)

    lines = [line for line in map(line_tuple_to_dict, lines)]
    for line in lines:
        line['width'] = 0.60

    # ==========================================================================
    # Set up Plotter
    # ==========================================================================

    plotter = MeshPlotter(mesh, figsize=(12, 9))
    plotter.draw_lines(lines)
    plotter.draw_faces(facecolor=colors)
    plotter.show()
