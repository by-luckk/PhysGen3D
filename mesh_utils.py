import open3d as o3d
import numpy as np
import trimesh
from scipy.spatial import KDTree
import copy
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import random

# def find_nearest_points()
#     pcd = o3d.io.read_point_cloud("path_to_your_pcd_file.pcd")
#     pcd_points = np.asarray(pcd.points)
#     mesh = trimesh.load("path_to_your_mesh_file.obj")
#     mesh_points = mesh.vertices
#     kdtree = KDTree(mesh_points)
#     distances, indices = kdtree.query(pcd_points)
#     constraint_vertex_indices = indices
#     constraint_vertex_positions = pcd_points
#     max_iter = 50 
#     smoothed_alpha = 0.01 
#     deformed_mesh = deform_as_rigid_as_possible(
#         self=mesh, 
#         constraint_vertex_indices=constraint_vertex_indices,
#         constraint_vertex_positions=constraint_vertex_positions,
#         max_iter=max_iter,
#     )
def sample_volume(trimesh_mesh, num_points, seed):
    np.random.seed(seed)
    return trimesh.sample.volume_mesh(trimesh_mesh, num_points)
    
def sample_and_fill_mesh(mesh_o3d, volume_sample_size, num_interior_points):
    object_points = np.asarray(mesh_o3d.vertices)
    faces = np.asarray(mesh_o3d.triangles)
    trimesh_mesh = trimesh.Trimesh(vertices=object_points, faces=faces)

    # sample points
    total_points = []
    num_iterations = int(num_interior_points / 1000)
    with ProcessPoolExecutor() as executor:
        futures = [
            executor.submit(sample_volume, trimesh_mesh, 1000, random.randint(0, 100000))
            for _ in range(num_iterations)
        ]
        for future in tqdm(as_completed(futures), total=num_iterations):
            sampled_points = future.result()
            total_points.append(sampled_points)
    interior_points = np.concatenate(total_points, axis=0)

    all_points = np.concatenate([interior_points, object_points], axis=0)
    # Do the volume sampling for the object points, prioritize the original object points, then surface points, then interior points
    min_bound = np.min(all_points, axis=0)
    index = []
    grid_flag = {}
    for i in range(object_points.shape[0]):
        grid_index = tuple(np.floor((object_points[i] - min_bound) / volume_sample_size).astype(int))
        if grid_index not in grid_flag:
            grid_flag[grid_index] = 1
            index.append(i)
    final_interior_points = []
    for i in range(interior_points.shape[0]):
        grid_index = tuple(np.floor((interior_points[i] - min_bound) / volume_sample_size).astype(int))
        if grid_index not in grid_flag:
            grid_flag[grid_index] = 1
            final_interior_points.append(interior_points[i])
    print('final_interior_points', len(final_interior_points))
    print(f'In {len(object_points)} object points, select {len(index)} points')
    if len(final_interior_points) == 0:
        all_points = object_points[index]
    else:
        all_points = np.concatenate([object_points[index], final_interior_points], axis=0)
    return all_points, index

def find_biggest_cluster(mesh, keep1=False, delete_vetices=True):
    print("Cluster connected triangles")
    with o3d.utility.VerbosityContextManager(
            o3d.utility.VerbosityLevel.Debug) as cm:
        triangle_clusters, cluster_n_triangles, cluster_area = (
            mesh.cluster_connected_triangles())
    triangle_clusters = np.asarray(triangle_clusters)
    cluster_n_triangles = np.asarray(cluster_n_triangles)
    cluster_area = np.asarray(cluster_area)

    mesh_0 = copy.deepcopy(mesh)
    if keep1:
        largest_cluster_idx = np.argmax(cluster_area)
        triangles_to_keep = (triangle_clusters == largest_cluster_idx)
        mesh_0.remove_triangles_by_mask(~triangles_to_keep)
    else:
        triangles_to_remove = cluster_n_triangles[triangle_clusters] < 100
        mesh_0.remove_triangles_by_mask(triangles_to_remove)
    if delete_vetices:
        mesh_0.remove_unreferenced_vertices()

    original_vertices = np.asarray(mesh.vertices)
    vertex_position_to_index = {tuple(pos): idx for idx, pos in enumerate(original_vertices)}
    mesh_0_vertices = np.asarray(mesh_0.vertices)
    vertex_indices_in_original = []

    cnt = 0
    for vertex in mesh_0_vertices:
        vertex_tuple = tuple(vertex)
        if vertex_tuple in vertex_position_to_index:
            vertex_indices_in_original.append(vertex_position_to_index[vertex_tuple])
        else:
            vertex_indices_in_original.append(None)
            cnt += 1
    print(f'{cnt} points not in original mesh')
    return mesh_0, vertex_indices_in_original
