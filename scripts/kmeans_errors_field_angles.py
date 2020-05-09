import math
import numpy as np

from random import choice
from time import time
from functools import partial

from numpy import asarray
from numpy import meshgrid
from numpy import linspace
from numpy import amax
from numpy import amin
from scipy.interpolate import griddata
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans

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


def cluster(data, n_clusters, reshape=None, normalize=False, random_state=0, n_jobs=-1):
	assert isinstance(data, dict)

	np_data = [x[1] for x in data.items()]
	np_data = np.array(np_data, dtype=np.float64)

	if normalize:
		np_data /= np.amax(np_data)

	if reshape:
		np_data = np_data.reshape(reshape)

	print()
	print("fitting {} clusters".format(n_clusters))
	t0 = time()

	kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_jobs=n_jobs)
	kmeans.fit(np_data)

	print("done in {} s, number of iterations: {}".format((time() - t0), kmeans.n_iter_))
	return kmeans.labels_, kmeans.cluster_centers_


def faces_labels(labels, centers):
	centers[centers < 0.1] = 0  # to avoid float rounding precision issues
	clustered_data = centers[labels].tolist()

	face_labels = {}
	for idx, fkey in enumerate(mesh.faces()):
		c = clustered_data[idx]
		face_labels[fkey] = c

	return face_labels


def faces_bisector_field(mesh, ref_vector_tag, target_vector_tag):

	for fkey, attr in mesh.faces(True):
		vec_1 = attr[ref_vector_tag]
		y = 1.0 / math.tan(math.radians(45.0))
		x_vec = vec_1
		y_vec = cross_vectors(x_vec, [0.0, 0.0, 1.0])  # global Z
		y_vec = scale_vector(y_vec, y)
		vec_3 = normalize_vector(add_vectors(x_vec, y_vec))
		
		mesh.face_attribute(fkey, name=target_vector_tag, value=vec_3)


def smoothed_angles(angles, smooth_iters):
	for _ in range(smooth_iters):
		averaged_angles = {}

		for fkey in mesh.faces():
			nbrs = mesh.face_neighbors(fkey)
			nbrs.append(fkey)
			local_angles = [angles[key] for key in nbrs]
			
			averaged_angles[fkey] = np.mean(np.array(local_angles))

		for fkey in mesh.faces():
			angles[fkey] = averaged_angles[fkey]


def faces_angles(mesh, vector_tag, ref_vector):
	angles = {}

	for fkey, attr in mesh.faces(data=True):
		vector = attr.get(vector_tag)
		angle = angle_vectors(ref_vector, vector, deg=True)
		angles[fkey] = angle

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

	print()
	print('max angle', min(angles.values()))
	print('min angle', max(angles.values()))

	return angles

# ==============================================================================
# Compute loss
# ==============================================================================



if __name__ == '__main__':

	# ==========================================================================
	# Constants
	# ==========================================================================

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



	filepaths = [
	'../data/json_files/four_point_slab.json',  # interesting
	'../data/json_files/perimeter_supported_slab.json', # interesting
	'../data/json_files/topleft_supported_slab.json',  # interesting
	'../data/json_files/leftright_supported_slab.json',  # interesting
	'../data/json_files/bottomleftright_supported_slab.json' , 
	'../data/json_files/middle_supported_slab_cantilever.json',
	'../data/json_files/triangle_supported_slab_cantilever.json'
	]

	vector_tag = 'ps_1_top'  # ps_1_top

	smooth_iters = 20
	n_clusters_iters = 20

	# ==========================================================================
	# Loop over filepaths
	# ==========================================================================

	for filepath in filepaths:

	# ==========================================================================
	# Import mesh
	# ==========================================================================

		mesh = Mesh.from_json(filepath)
		mesh_unify_cycles(mesh)

	# ==========================================================================
	# 45 degrees field
	# ==========================================================================

		faces_bisector_field(mesh, vector_tag, vector_tag)

	# ==========================================================================
	# Process PS vectors
	# ==========================================================================

		angles = faces_angles(mesh, vector_tag, ref_vector=[1.0, 1.0, 0.0])

	# ==========================================================================
	# Average smooth angles
	# ==========================================================================

		smoothed_angles(angles, smooth_iters)

	# ==========================================================================
	# Loss Computation
	# ==========================================================================

		errors = []
		
		for i in range(1, n_clusters_iters + 1):
			k_error = 0.0

			labels, centers = cluster(angles, i, reshape=(-1, 1))
			cluster_labels = faces_labels(labels, centers)

			for fkey in mesh.faces():
				c_angle = cluster_labels[fkey]
				angle = angles[fkey]

				error = (angle - c_angle) ** 2
				k_error += error

			k_error /= len(cluster_labels.values())
			errors.append(k_error)

		
		plt.plot(errors, marker='o', label=filepath)
	
	# ==========================================================================
	# Show
	# ==========================================================================

	plt.axes()
	plt.grid(b=None, which='major', axis='both', linestyle='--')
	
	plt.xticks(list(range(0, n_clusters_iters)), list(range(1, n_clusters_iters + 1)))
	
	plt.xlabel("Number of Clusters")
	plt.ylabel("Error (MSE)")

	plt.legend()

	plt.show()
