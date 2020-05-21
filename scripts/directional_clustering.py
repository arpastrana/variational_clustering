import math
import numpy as np
import matplotlib.pyplot as plt

from time import time

from numpy import asarray
from numpy import meshgrid
from numpy import linspace
from numpy import amax
from numpy import amin

from scipy.interpolate import griddata

from sklearn.cluster import KMeans

from compas.geometry import scale_vector
from compas.geometry import add_vectors
from compas.geometry import dot_vectors
from compas.geometry import normalize_vector
from compas.geometry import length_vector
from compas.geometry import cross_vectors
from compas.geometry import angle_vectors
from compas.geometry import rotate_points
from compas.geometry import centroid_points
from compas.geometry import angle_vectors_signed

from compas.datastructures import Mesh
from compas.datastructures import mesh_unify_cycles

from compas.utilities import i_to_rgb
from compas.utilities import i_to_black
from compas.utilities import i_to_red
from compas.utilities import i_to_blue

from compas_plotters import MeshPlotter

from streamlines.basic_sandwich import ReinforcedShell
from streamlines.materials import Concrete
from streamlines.materials import Rebar


def line_sdl(start, direction, length, both_sides=True):

	direction = normalize_vector(direction[:])
	a = start
	b = add_vectors(start, scale_vector(direction, +length))
	if both_sides:
		a = add_vectors(start, scale_vector(direction, -length))	
	return a, b


def vector_lines_on_faces(mesh, vector_tag, uniform=True, factor=0.02):

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

	a, b = line
	return {'start': a, 'end': b}


def polygon_list_to_dict(polygon):

	return {'points': polygon}


def extract_polygons_from_contours(contours, levels):

	polygons = []
	for i in range(len(contours)):
		level = levels[i]
		contour = contours[i]

		for path in contour:
			for polygon in path:
				polygons.append(polygon[:-1])
	return polygons


def scalarfield_contours_numpy(xy, s, levels=50, density=100, method='cubic'):

	xy = asarray(xy)
	s = asarray(s)
	x = xy[:, 0]
	y = xy[:, 1]

	X, Y = meshgrid(linspace(amin(x), amax(x), 2 * density),
					linspace(amin(y), amax(y), 2 * density))

	S = griddata((x, y), s, (X, Y), method=method)

	fig = plt.figure()
	ax = fig.add_subplot(111, aspect='equal')
	c = ax.contour(X, Y, S, levels)

	contours = [0] * len(c.collections)
	levels = c.levels

	for i, coll in enumerate(iter(c.collections)):
		paths = coll.get_paths()
		contours[i] = [0] * len(paths)
		for j, path in enumerate(iter(paths)):
			polygons = path.to_polygons()
			contours[i][j] = [0] * len(polygons)
			for k, polygon in enumerate(iter(polygons)):
				contours[i][j][k] = polygon

	plt.close(fig)

	return levels, contours


def contour_polygons(mesh, centers, face_labels, density=100, method='nearest'):

	xy, s = [], []

	for fkey in mesh.faces():
		point = mesh.face_centroid(fkey)[:2]
		xy.append(point)
		a = face_labels[fkey]
		s.append(a)

	b = np.sort(centers, axis=0).flatten()

	levels, contours = scalarfield_contours_numpy(xy, s, levels=b, density=density, method=method)
	polygons = extract_polygons_from_contours(contours, levels)

	polygons = [p for p in map(polygon_list_to_dict, polygons)]
	for p in polygons:
		p['edgewidth'] = 1.0

	return polygons


def color_maker(data, callback, invert=False):

	assert isinstance(data, dict)

	dataarray = np.array(list(data.values()))
	valuemin = np.amin(dataarray)
	valuemax = np.amax(dataarray - valuemin)

	colors = {}
	for idx, value in data.items():
		centered_val = (value - valuemin)  # in case min is not 0
		if not invert:
			ratio = centered_val / valuemax  # tuple 0-255
		else:
			ratio = (valuemax - centered_val) / valuemax

		colors[idx] = callback(ratio)  # tuple 0-255

	return colors

def rgb_colors(data, invert=False):
	return color_maker(data, i_to_rgb, invert)


def black_colors(data, invert=False):
	return color_maker(data, i_to_black, invert)


def inverted_blue_colors(data, invert=True):
	return color_maker(data, i_to_blue, invert)


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
		c = clustered_data[idx][0]
		face_labels[fkey] = c

	return face_labels


def faces_bisector_field(mesh, vector_tag_1, vector_tag_2, target_vector_tag):

	for fkey, attr in mesh.faces(True):
		vec_1 = attr[vector_tag_1]
		vec_2 = attr[vector_tag_2]
		vec_3 = add_vectors(normalize_vector(vec_1), normalize_vector(vec_2))		
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


def vector_from_angle(angle, base_vector):

	rot_pts = rotate_points([base_vector], angle)
	return rot_pts[0]


def vectors_from_angles(angles, base_vector):

	vectors = {}
	
	for fkey, angle in angles.items():
		rot_pts = rotate_points([base_vector], angle)
		vectors[fkey] = vector_from_angle(angle, base_vector)

	return vectors


def plot_colored_vectors(angles, base_vector):

	vectors = vectors_from_angles(angles, base_vector)

	X = np.array(list(vectors.values()))
	x = X[:, 0]
	y = X[:, 1]

	c = np.array(list(kcolors.values())).reshape((-1, 3)) / 255.0

	plt.scatter(x, y, c=c.tolist())
	plt.show()


def cluster_center_vectors(mesh, centers, cluster_labels, base_vector, length=1.0):

	center_vectors = []
	for center in centers:

		angle = center

		centroids = []
		for fkey, label in cluster_labels.items():
			if label != center:
				continue
			centroids.append(mesh.face_centroid(fkey))

		cluster_centroid = centroid_points(centroids)
		vec = vector_from_angle(angle, base_vector)	
		
		for sign in (-1, 1):

			a = vector_from_angle(sign * math.pi / 4, vec)
			arrow = line_sdl(cluster_centroid, a, length, both_sides=False)

			center_vectors.append(arrow)

	return center_vectors


def cluster_arrows(arrows, width=1.0, color=None):

	arrows = list(map(line_tuple_to_dict, arrows))
	for arrow in arrows:
		arrow['width'] = width
		if color:
			arrow['color'] = color

	return arrows


def cos_angle(u, v):
	return angle_vectors(u, v)


def raw_cos_angle(u, v):
	return math.acos(dot_vectors(u, v) / (length_vector(u) * length_vector(v)))


def signed_angle(u, v):
	return angle_vectors_signed(u, v, normal=[0.0, 0.0, 1.0])


def det_vectors_xy(u, v):
	return u[0] * v[1] - v[0] * u[1]


def clockwise(u, v):
	dot = dot_vectors(u, v)
	det = det_vectors_xy(u, v)
	beta = math.atan2(det, dot)
	return (beta + 2 * math.pi) % (2 * math.pi)


def faces_angles(mesh, vector_tag, ref_vector, func):

	def twoway_angles(u, v, func, deg):
		if not deg:
			return [func(scale_vector(u, sign), v) for sign in [1, -1]]
		return [math.degrees(func(scale_vector(u, sign), v)) for sign in [1, -1]]


	angles = {}
	for fkey, attr in mesh.faces(data=True):
		vector = attr.get(vector_tag)
		u, v = vector, ref_vector
		twangles = twoway_angles(u, v, func=func, deg=False)
		angles[fkey] = min(twangles)

	print('min angle', min(angles.values()))
	print('max angle', max(angles.values()))

	return angles


def faces_clustered_field(mesh, cluster_labels, base_tag, target_tag, func):

	x = [1.0, 0.0, 0.0]
	for fkey, angle in cluster_labels.items():

		base_vec = mesh.face_attribute(fkey, name=base_tag)
		delta = angle - func(base_vec, x)  # or +? 

		vec = vector_from_angle(delta, base_vec)
		test_angle = func(vec, x)

		if math.fabs(test_angle - angle) > 0.001:
			vec = vector_from_angle(-delta, base_vec)

		mesh.face_attribute(fkey, name=target_tag, value=vec)


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

	HERE = '../data/json_files/four_point_slab'  # interesting
	HERE = '../data/json_files/perimeter_supported_slab' # interesting
	# HERE = '../data/json_files/topleft_supported_slab'  # interesting
	# HERE = '../data/json_files/bottomleftright_supported_slab'  # interesting
	# HERE = '../data/json_files/triangle_supported_slab_cantilever'  # interesting
	# HERE = '../data/json_files/middle_supported_slab_cantilever'
	# HERE = '../data/json_files/leftright_supported_slab'


	base_vector_tag = 'm_1'
	
	transformable_vector_tags = ['m_1']
	vector_cluster_tags = ['m_1_k']
	vector_display_tags = ['m_1', 'm_1_k']

	vector_display_colors = [(0, 0, 255), (255, 0, 0)]

	mode = cos_angle    # clockwise or cos_angle
	smooth_iters = 2
	n_clusters = 4

	data_to_color_tag = "uncolored"  # angles, clusters, uncolored

	plot_mesh = True

	draw_contours = True
	draw_vector_field = True

	draw_kmeans_colors = False  # 2d representation

	export_json = False

	# ==========================================================================
	# Import mesh
	# ==========================================================================

	mesh = Mesh.from_json(HERE + ".json")
	mesh_unify_cycles(mesh)

	# ==========================================================================
	# Process PS vectors
	# ==========================================================================

	angles = faces_angles(mesh, base_vector_tag, ref_vector=[1.0, 0.0, 0.0], func=mode)

	# ==========================================================================
	# Average smooth angles
	# ==========================================================================

	smoothed_angles(angles, smooth_iters)

	# ==========================================================================
	# Kmeans angles
	# ==========================================================================

	labels, centers = cluster(angles, n_clusters, reshape=(-1, 1))

	print('centers')
	for idx, c in enumerate(centers):
		print(idx, math.degrees(c), "deg")

	# ==========================================================================
	# Quantized Colors
	# ==========================================================================

	cluster_labels = faces_labels(labels, centers)

	for fkey, label in cluster_labels.items():
		mesh.face_attribute(fkey, name="k_label", value=label)

	# ==========================================================================
	# Register clustered field
	# ==========================================================================
	
	for ref_tag, target_tag in zip(transformable_vector_tags, vector_cluster_tags):
		faces_clustered_field(mesh, cluster_labels, ref_tag, target_tag, func=mode)

	# ==========================================================================
	# data to plot
	# ==========================================================================

	data_to_color = {
		"clusters": rgb_colors(cluster_labels),
		"angles": rgb_colors(angles),
		"uncolored": {}
		}
	
	datacolors = data_to_color[data_to_color_tag]
	
	# ==========================================================================
	# Kmeans plot 2d
	# ==========================================================================

	if draw_kmeans_colors:
		plot_colored_vectors(angles, base_vector=[1.0, 0.0, 0.0])

	# ==========================================================================
	# Set up Plotter
	# ==========================================================================

	plotter = MeshPlotter(mesh, figsize=(12, 9))
	plotter.draw_edges(keys=list(mesh.edges_on_boundary()))
	plotter.draw_faces(facecolor=datacolors)
	# plotter.draw_faces(facecolor=datacolors, text=angles)
	
	# ==========================================================================
	# Scalar Contouring
	# ==========================================================================

	if draw_contours:
		polygons = contour_polygons(mesh, centers, cluster_labels, 100, 'nearest')
		plotter.draw_polylines(polygons)

	# ==========================================================================
	# Create PS vector lines
	# ==========================================================================

	if draw_vector_field:

		lines = []
		length = 0.05

		for vector_display_tag, c in zip(vector_display_tags, vector_display_colors):
			_lines = vector_lines_on_faces(mesh, vector_display_tag, True, factor=length)
			_lines = [line for line in map(line_tuple_to_dict, _lines)]

			for line in _lines:
				line['width'] = 0.5
				line['color'] = c

			lines.extend(_lines)

		plotter.draw_lines(lines)

	# ==========================================================================
	# Export json
	# ==========================================================================

	if export_json:
		out = HERE + "_k_{}_smooth_{}.json".format(n_clusters, smooth_iters)
		mesh.to_json(out)
		print("Exported mesh to: {}".format(out))

	# ==========================================================================
	# Show
	# ==========================================================================

	if plot_mesh:
		plotter.show()
