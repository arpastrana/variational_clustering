import numpy as np

from math import fabs
from math import degrees

from variational_clustering.clustering import make_faces
from variational_clustering.clustering import furthest_init
from variational_clustering.clustering import k_means

from directional_clustering.plotters import ClusterPlotter
from directional_clustering.plotters import rgb_colors
from directional_clustering.plotters import plot_colored_vectors
from directional_clustering.geometry import laplacian_smoothed

from compas.datastructures import Mesh
from compas.datastructures import mesh_unify_cycles
from compas.utilities import geometric_key

from directional_clustering.plotters import ClusterPlotter

# ==========================================================================
# Constants
# ==========================================================================

tags = [
    "n_1",
    "n_2",
    "m_1",
    "m_2",
    "ps_1_top",
    "ps_1_bot",
    "ps_1_mid",
    "ps_2_top",
    "ps_2_bot",
    "ps_2_mid",
    "custom_1",
    "custom_2"
    ]

HERE = "../data/json_files/perimeter_supported_slab"

THERE = HERE.replace("json_files", "images")

base_vector_tag = "m_1"
transformable_vector_tags = ["m_1", "m_2"]
vector_cluster_tags = ["m_1_k", "m_2_k"]
vector_display_tags = [base_vector_tag]

vector_display_colors = [(0, 0, 0), (255, 0, 0)]  # blue and red
perp_flags = [False, True]
line_width = 1.0

n_clusters = 5  # int or auto
iters = 30
mergesplit = True
x_lim = -10.0

smooth_iters = 0  # currently fails because vectors are flipped everywhere
damping = 0.5

show_mesh = True
save_fig = False

data_to_color_tag = "clusters"  # angles, clusters, uncolored

draw_vector_field = False
uniform_length = True
vector_length = 0.05  # 0.005 if not uniform, 0.05 otherwise

export_json = False

# ==========================================================================
# Import mesh
# ==========================================================================

name = HERE.split("/").pop()
mesh = Mesh.from_json(HERE + ".json")
mesh_unify_cycles(mesh)

# ==========================================================================
# Store subset attributes
# ==========================================================================

centroids = {}
vectors = {}
for fkey in mesh.faces():
    centroids[geometric_key(mesh.face_centroid(fkey))] = fkey
    vectors[fkey] = mesh.face_attribute(fkey, base_vector_tag)

# ==========================================================================
# Rebuild mesh
# ==========================================================================

polygons = [mesh.face_coordinates(fkey) for fkey in mesh.faces() if mesh.face_centroid(fkey)[0] >= x_lim]
mesh = Mesh.from_polygons(polygons)
mesh_unify_cycles(mesh)

for fkey in mesh.faces():
    gkey = geometric_key(mesh.face_centroid(fkey))
    ofkey = centroids[gkey]
    vector = vectors[ofkey]
    mesh.face_attribute(fkey, base_vector_tag, vector)

# ==========================================================================
# Laplacian smoothing
# ==========================================================================

vectors = {fkey: mesh.face_attribute(fkey, base_vector_tag) for fkey in mesh.faces()}
if smooth_iters:
    vectors = laplacian_smoothed(mesh, vectors, smooth_iters, damping)

# ==========================================================================
# Extract faces
# ==========================================================================

faces = make_faces(mesh, vectors)

# ==========================================================================
# Cluster
# ==========================================================================

initial_clusters = furthest_init(n_clusters, faces, callback=None)[-1]
final_clusters = k_means(initial_clusters, faces, iters, mergesplit)[-1]

# ==========================================================================
# Cluster labels
# ==========================================================================

cluster_labels = {}
for i, cluster in final_clusters.items():
    label = cluster.id
    for fkey in cluster.faces_keys:
        cluster_labels[fkey] = label
        mesh.face_attribute(fkey, name="k_label", value=label)

# ==========================================================================
# Register clustered field
# ==========================================================================

# for ref_tag, target_tag, perp in zip(transformable_vector_tags, vector_cluster_tags, perp_flags):
#     faces_clustered_field(mesh, cluster_labels, ref_tag, target_tag, perp=perp, func=mode)

# =============================================================================
# data to plot
# =============================================================================

data_to_color = {
    "clusters": rgb_colors(cluster_labels),
    "uncolored": {}
    }

datacolors = data_to_color[data_to_color_tag]

# =============================================================================
# Set up Plotter
# =============================================================================

plotter = ClusterPlotter(mesh, figsize=(12, 9))
plotter.draw_edges(keys=list(mesh.edges_on_boundary()))
plotter.draw_faces(facecolor=datacolors)

# =============================================================================
# Create PS vector lines
# =============================================================================

if draw_vector_field:
    for tag, color in zip(vector_display_tags, vector_display_colors):
        plotter.draw_vector_field(tag, color, uniform_length, vector_length, line_width)

# =============================================================================
# Export json
# =============================================================================

if export_json:
    out = HERE + "_k_{}.json".format(n_clusters)
    mesh.to_json(out)
    print("Exported mesh to: {}".format(out))

if save_fig:
    out = THERE + "_field_{}.png".format(n_clusters)
    plotter.save(out, bbox_inches="tight")

# =============================================================================
# Show
# =============================================================================

if show_mesh:
    plotter.show()
