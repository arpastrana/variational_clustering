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

    HERE = '../data/json_files/topleft_supported_slab_k25_i50_ms_ps1top.json'
    HERE = '../data/json_files/four_point_slab_k5_i20_ms_ps1top.json'
    vector_tag = 'ps_1_top'

    # ==========================================================================
    # Import mesh
    # ==========================================================================

    mesh = Mesh.from_json(HERE)
    mesh_unify_cycles(mesh)

    # ==========================================================================
    # Create PS vector lines
    # ==========================================================================

    lines = vector_lines_on_faces(mesh, vector_tag, True, factor=0.05)

    lines = [line for line in map(line_tuple_to_dict, lines)]
    for line in lines:
        line['width'] = 0.60

    # ==========================================================================
    # Define Callbacks
    # ==========================================================================

    def label_colors(mesh):
        num = max(mesh.faces_attribute(name='k_label', keys=mesh.faces()))

        colordict = {}
        for fkey, attr in mesh.faces(True):
            label = attr['k_label']
            color = [i / 255 for i in i_to_rgb(label / num)]
            colordict[fkey] = color

        return colordict

    # ==========================================================================
    # Find Vertices
    # ==========================================================================

    vertices = []
    for vkey in mesh.vertices():
        fkeys = mesh.vertex_faces(vkey)
        labels = mesh.faces_attribute(name='k_label', keys=fkeys)
        labels = set(labels)

        if len(labels) > 2:
            vertices.append(vkey)

        elif mesh.is_vertex_on_boundary(vkey):
            if len(labels) > 1:
                vertices.append(vkey)

    # ==========================================================================
    # Set up Plotter
    # ==========================================================================

    plotter = MeshPlotter(mesh, figsize=(12, 9))
    # plotter.draw_lines(lines)
    plotter.draw_faces(facecolor=label_colors(mesh))
    plotter.draw_vertices(facecolor=(255, 0, 0), keys=vertices, radius=0.05)
    plotter.show()
