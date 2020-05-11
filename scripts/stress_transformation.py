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
from compas.geometry import normalize_vector
from compas.geometry import length_vector
from compas.geometry import cross_vectors
from compas.geometry import angle_vectors
from compas.geometry import rotate_points
from compas.geometry import centroid_points

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

	valuemax = np.amax(np.array(list(data.values())))

	colors = {}
	for idx, value in data.items():
		if not invert:
			ratio = value / valuemax  # tuple 0-255
		else:
			ratio = (valuemax - value) / valuemax

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


def vector_from_angle(angle, base_vector=[1.0, 0.0, 0.0]):

	rot_pts = rotate_points([base_vector], math.radians(angle))
	return rot_pts[0]


def vectors_from_angles(angles, base_vector=[1.0, 0.0, 0.0]):

	vectors = {}
	
	for fkey, angle in angles.items():
		rot_pts = rotate_points([base_vector], math.radians(angle))
		vectors[fkey] = vector_from_angle(angle, base_vector)

	return vectors


def plot_colored_vectors(angles, base_vector=[1.0, 0.0, 0.0]):

	vectors = vectors_from_angles(angles, base_vector)

	X = np.array(list(vectors.values()))
	x = X[:, 0]
	y = X[:, 1]

	c = np.array(list(kcolors.values())).reshape((-1, 3)) / 255.0

	plt.scatter(x, y, c=c.tolist())
	plt.show()


def cluster_center_vectors(mesh, centers, cluster_labels, base_vector=[1.0, 1.0, 0.0], length=1.0):

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

			a = vector_from_angle(sign * 45.0, vec)
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


def faces_clustered_field(mesh, cluster_labels, tags, base_vector=[1.0, 1.0, 0.0]):

	for fkey, label in cluster_labels.items():
		angle = label

		vec = vector_from_angle(angle, base_vector)
	
		for sign, tag in zip((-1, 1), tags):
			rotvec = vector_from_angle(sign * 45.0, vec)
			mesh.face_attribute(fkey, name=tag, value=rotvec)


def transformed_forces(fx, fy, fxy, angle):
	theta = angle

	tfx = (fx + fy) / 2 + (fx - fy) * math.cos(2 * theta) / 2 + fxy * math.sin(2 * theta)
	tfy = fx + fy - tfx
	tfxy = - (fx - fy) * math.sin(2 * theta) / 2 + fxy * math.cos(2 * theta)

	return tfx, tfy, tfxy


def principal_forces(fx, fy, fxy):

	pf_a = (fx + fy) / 2
	pf_b = math.sqrt(((fx - fy) / 2 )** 2 + fxy ** 2)

	return pf_a + pf_b, pf_a - pf_b


def principal_angles(fx, fy, fxy):

	a = 2 * fxy / (fx - fy)
	b = math.atan(a) / 2
	
	return b, b + math.radians(90.0)


def aligned_principal_angles(forces, tol=0.01):
	fx, fy, fxy = forces

	a1, a2 = principal_angles(fx, fy, fxy)
	pf1, pf2 = principal_forces(fx, fy, fxy)
	tf1, tf2, _ = transformed_forces(fx, fy, fxy, a1)

	if math.fabs(math.fabs(tf1) - math.fabs(pf1)) < tol:
		return a1, a2

	return a2, a1


def compute_design_moments(mesh, refmoments, angles):

	transformed = {}
	for fkey in mesh.faces():
		mx, my, mxy = refmoments[fkey]
		angle = angles[fkey]
		t_moments = transformed_forces(mx, my, mxy, angle)
		transformed[fkey] = t_moments

	return transformed


def bar_area(diameter):
	return (math.pi * bar_diameter ** 2) / 4.0


if __name__ == '__main__':

	# ==========================================================================
	# Constants
	# ==========================================================================

	# HERE = '../data/json_files/four_point_slab'  # interesting
	# HERE = '../data/json_files/perimeter_supported_slab' # interesting
	# HERE = '../data/json_files/topleft_supported_slab'  # interesting

	# HERE = '../data/json_files/leftright_supported_slab'  # interesting

	HERE = '../data/json_files/bottomleftright_supported_slab'  

	HERE = '../data/json_files/middle_supported_slab_cantilever'
	# HERE = '../data/json_files/triangle_supported_slab_cantilever'  # interesting

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

	forces = [
		'nx',
		'ny',
		'nxy',
		'mx',
		'my',
		'mxy',
		'vx',
		'vy'
		]


	vector_tag = 'ps_1_top'
	bisec_vector_tag = 'ps_12_top'
	cluster_vector_tags = ['ps_1_top_cluster', 'ps_2_top_cluster']

	vector_display_tags = ['ps_1_top', 'ps_2_top']
	vector_display_tags = cluster_vector_tags

	vector_display_colors = [(50, 50, 50), (50, 50, 50)]

	smooth_iters = 20
	n_clusters = 2

	data_to_color_tag = "clusters"  # angles, clusters, mass, spacings

	design_set = "ortho"  # ortho, ps, k

	steel_tag = "asyb"
	
	draw_contours = False
	draw_vector_field = False
	draw_kmeans_colors = False  # 2d representation
	draw_arrows = False

	plot_mesh = True

	export_json = True

	fck = 40.0
	fy = 500.0
	cover = 0.05  # m
	shell_thickness = 0.15  # m
	ftcd_factor = 10

	bar_diameter = 0.8   # cm
	max_spacing = 0.40  # m

	# ==========================================================================
	# Import mesh
	# ==========================================================================

	mesh = Mesh.from_json(HERE + ".json")
	mesh_unify_cycles(mesh)

	# ==========================================================================
	# 45 degrees field
	# ==========================================================================

	faces_bisector_field(mesh, vector_tag, bisec_vector_tag)

	# ==========================================================================
	# Process PS vectors
	# ==========================================================================

	angles = faces_angles(mesh, bisec_vector_tag, ref_vector=[1.0, 1.0, 0.0])

	# ==========================================================================
	# Compute principal forces
	# ==========================================================================

	moment_tags = ["mx", "my", "mxy"]
	ref_moment_tags = ["m_1", "m_2"]

	prin_angles = {}
	ref_moments = {}
	delta_angles = {}

	for fkey in mesh.faces():		
		forces = mesh.face_attributes(fkey, names=moment_tags)

		karamba_moments = mesh.face_attributes(fkey, names=ref_moment_tags)

		pforces = principal_forces(*forces)
		pangles = aligned_principal_angles(forces, tol=0.01)

		pangle = pangles[0]
		angle = angles[fkey]

		prin_angles[fkey] = pangle
		ref_moments[fkey] = forces
		delta_angles[fkey] = pangle - math.radians(angle)


	# ==========================================================================
	# Average smooth angles
	# ==========================================================================

	smoothed_angles(angles, smooth_iters)

	# ==========================================================================
	# Kmeans angles
	# ==========================================================================

	labels, centers = cluster(angles, n_clusters, reshape=(-1, 1))

	print('centers')
	print(centers)

	# ==========================================================================
	# Quantized Colors
	# ==========================================================================

	cluster_labels = faces_labels(labels, centers)

	for fkey, label in cluster_labels.items():
		mesh.face_attribute(fkey, name="k_label", value=label)

	# ==========================================================================
	# Register clustered field
	# ==========================================================================

	tags = cluster_vector_tags
	faces_clustered_field(mesh, cluster_labels, tags, base_vector=[1.0, 1.0, 0.0])

	# ==========================================================================
	# Transform forces for design
	# ==========================================================================

	design_angles = {k: v for k, v in cluster_labels.items()}
	threshold = math.radians(45.0)

	for fkey in angles.keys():
		temp = prin_angles[fkey]
		a = math.radians(design_angles[fkey])
		a += delta_angles[fkey]		
		design_angles[fkey] = a


	design_moments_set = {
			"ortho": ref_moments,
			"ps": compute_design_moments(mesh, ref_moments, prin_angles),
			"k": compute_design_moments(mesh, ref_moments, design_angles)
			}

	design_moments = design_moments_set[design_set]

	# ==========================================================================
	# Sandwich definition
	# ==========================================================================

	shells = {}

	for fkey in mesh.faces():
		concrete = Concrete(fck)
		rebar = Rebar(fy)
		shell = ReinforcedShell(fkey, concrete, rebar, shell_thickness, cover)
		shell.tcred = shell.concrete.fctd * ftcd_factor

	# ==========================================================================
	# Sandwich internal forces
	# ==========================================================================

		nx, ny, nxy = 0.0, 0.0, 0.0
		vx, vy = mesh.face_attributes(fkey, names=['vx', 'vy'])

		try:
			mx, my, mxy = design_moments[fkey]
		except ValueError:
			mx, my = design_moments[fkey]
			mxy = 0.0

		shell.applyForces(nx, ny, nxy, mx, my, mxy, vx, vy)
		shell.get_membrane_forces()

	# ==========================================================================
	# Sandwich reinforcement design
	# ==========================================================================

		shell.get_reinforcement_forces()
		shell.design_reinforcement()

		shells[fkey] = shell

	# ==========================================================================
	# Force statistics
	# ==========================================================================

	steel_tags = {"asxt": 0.0, "asxb": 0.0, "asxc": 0.0, "asyt": 0.0, "asyb": 0.0, "asyc": 0.0, "ast": 0.0}
	steel_massing = {tag: {} for tag in steel_tags.keys()}

	for fkey in mesh.faces():		
		shell = shells[fkey]

		for tag in steel_tags.keys():
			mass = getattr(shell, tag)
			steel_massing[tag][fkey] = mass
			steel_tags[tag] +=mass

	print()
	print("masses")
	for k, v in steel_tags.items():
		print(k, v)

	print()
	for k, v in steel_massing.items():
		massing = list(v.values())
		print(k, " ", "min mass", min(massing), "max mass", max(massing))

	# ==========================================================================
	# Mat spacings
	# ==========================================================================

	steel_spacing = {tag: {} for tag in steel_tags.keys()}

	area = bar_area(bar_diameter)
	for tag, steel_mass in steel_massing.items():
		for fkey, steel_req in steel_mass.items():

			tol = 1e-6
			if steel_req < tol:
				steel_spacing[tag][fkey] = max_spacing
				continue

			num_bars = steel_req / area
			spacing = 1 / num_bars

			if spacing > max_spacing:
				spacing = max_spacing

			steel_spacing[tag][fkey] = spacing


	print()
	print("spacings")
	for k, v in steel_spacing.items():
		spacings = np.array([v for v in v.values() if v != -1.0])
		try:
			print(k, " ", "min", np.amin(spacings), "max", np.amax(spacings), "mean", \
			      np.mean(spacings), "median", np.median(spacings))
		except ValueError:
			pass

	# ==========================================================================
	# data to plot
	# ==========================================================================

	data_to_color = {
		"clusters": rgb_colors(cluster_labels),
		"mass": rgb_colors({fkey: steel_massing[steel_tag][fkey] for fkey in mesh.faces()}),
		"spacings": rgb_colors({fkey: steel_spacing[steel_tag][fkey] for fkey in mesh.faces()}, invert=True),
		"angles": rgb_colors(angles)
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
	# Draw cluster arrows
	# ==========================================================================

	if draw_arrows:
		center_vecs = cluster_center_vectors(mesh, centers, cluster_labels, length=0.2)
		arrows = cluster_arrows(center_vecs, width=1.0, color=None)
		plotter.draw_arrows(arrows)

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
