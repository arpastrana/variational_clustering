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

    HERE = '../data/json_files/four_point_slab.json'  # interesting
    # HERE = '../data/json_files/perimeter_supported_slab.json' # interesting
    # HERE = '../data/json_files/topleft_supported_slab.json'  # interesting

    # HERE = '../data/json_files/leftright_supported_slab.json'  # interesting

    # HERE = '../data/json_files/bottomleftright_supported_slab.json'  
    # HERE = '../data/json_files/middle_supported_slab_cantilever.json'

    THERE = 'json_files/four_point_slab_k5_i20_ms_ps1top.json'

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

    NUM = 5
    ITERS = 20
    MERGESPLIT = True
    vector_tag = 'ps_1_top'  # ps_1_top

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

    def callback(k, plotter, clusters):
        num = len(list(clusters.keys()))

        facedict = {}
        for idx, cluster in clusters.items():
            color = [i / 255 for i in i_to_rgb(idx / num)]

            for fkey in cluster.faces_keys:
                facedict[fkey] = color

        plotter.update_faces(facedict)        
        plotter.update(pause=0.05)


    def colors_from_distortions(final_clusters):
        colors = {}
        max_distortion = max([cluster.distortion for _, cluster in final_clusters.items()])
        print('max dstr', max_distortion)
        for idx, cluster in final_clusters.items():
            distortion = cluster.distortion
            print('cluster dstr', distortion)

            color = [i / 255 for i in i_to_blue((distortion / max_distortion))]

            for fkey in cluster.faces_keys:
                colors[fkey] = color
        return colors

    # ==========================================================================
    # Set up Plotter
    # ==========================================================================

    # plotter = MeshPlotter(mesh, figsize=(12, 9))
    # plotter.draw_lines(lines)
    # plotter.draw_faces()
    # plotter.update(0.5)
    # callback = partial(callback, plotter=plotter)

    # ==========================================================================
    # K-Means
    # ==========================================================================

    c = None

    faces = make_faces(mesh, vector_tag)
    clusters = furthest_init(NUM, faces, callback=c)
    
    sel_clusters = clusters[-1]
    all_clusters = k_means(sel_clusters, faces, ITERS, MERGESPLIT, callback=c)

    final_clusters = all_clusters[-1]

    print('kmeant')

    # ==========================================================================
    # Export labeled json
    # ==========================================================================

    for idx, cluster in final_clusters.items():
        for fkey in cluster.faces_keys:
            mesh.face_attribute(fkey, name='k_label', value=idx)

    mesh.to_json(THERE)

    # ==========================================================================
    # Visualization
    # ==========================================================================

    # callback(k=0, clusters=final_clusters)

    # plotter.update_faces(colors_from_distortions(final_clusters))
    # plotter.update(0.5)

    # plotter.show()

    print('done')
