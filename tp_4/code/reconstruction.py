#
#
#      0===========================================================0
#      |              TP4 Surface Reconstruction                   |
#      0===========================================================0
#
#
# ------------------------------------------------------------------------------------------
#
#      Jean-Emmanuel DESCHAUD - 15/01/2024
#


# Import numpy package and name it "np"
import numpy as np

# Import functions from scikit-learn
from sklearn.neighbors import KDTree

# Import functions to read and write ply files
from ply import write_ply, read_ply

# Import time package
import time

from skimage import measure

import trimesh


# Hoppe surface reconstruction
def compute_hoppe(points,normals,scalar_field,grid_resolution,min_grid,size_voxel):
    """ function to compute the hoppe function"""
    my_tree = KDTree(points, 10)

    d = points.shape[1]  # should be equal to 3
    coords = np.arange(grid_resolution)
    queries = np.stack(np.meshgrid(*d * [coords], indexing="ij"), axis=-1)
    queries = (queries * size_voxel + min_grid).astype(np.float32).reshape(-1, d)

    indices = my_tree.query(queries, return_distance=False).squeeze()

    # Hoppe function
    volume = np.sum(normals[indices] * (queries - points[indices]), axis=-1)

    scalar_field[:, :, :] = volume.reshape(*d * [grid_resolution])

    return scalar_field

# IMLS surface reconstruction
def compute_imls(points,normals,scalar_field,grid_resolution,min_grid,size_voxel,knn, h = 0.01):
    """ function to compute the IMLS function"""
    my_tree = KDTree(points, 10)
    d = points.shape[1]  # should be equal to 3

    coords = np.arange(grid_resolution)
    queries = np.stack(np.meshgrid(*d * [coords], indexing="ij"), axis=-1)
    queries = (queries * size_voxel + min_grid).astype(np.float32).reshape(-1, d)

    distances, indices = my_tree.query(queries, k=knn)

    # computing theta
    theta = np.exp(-((distances / h) ** 2))
    hoppe = np.sum(normals[indices] * (queries[:, np.newaxis] - points[indices]), axis=-1)
    volume = np.sum(hoppe * theta, axis=-1) / (np.sum(theta, axis=-1) + 1e-16)

    scalar_field[:, :, :] = volume.reshape(*d * [grid_resolution])

    return scalar_field



if __name__ == '__main__':

    t0 = time.time()

    # Path of the file
    file_path = '../data/bunny_normals.ply'

    # Load point cloud
    data = read_ply(file_path)

    # Concatenate data
    points = np.vstack((data['x'], data['y'], data['z'])).T
    normals = np.vstack((data['nx'], data['ny'], data['nz'])).T

	# Compute the min and max of the data points
    min_grid = np.amin(points, axis=0)
    max_grid = np.amax(points, axis=0)

	# Increase the bounding box of data points by decreasing min_grid and inscreasing max_grid
    min_grid = min_grid - 0.10*(max_grid-min_grid)
    max_grid = max_grid + 0.10*(max_grid-min_grid)

	# grid_resolution is the number of voxels in the grid in x, y, z axis
    grid_resolution = 128 #128
    size_voxel = max([(max_grid[0]-min_grid[0])/(grid_resolution-1),(max_grid[1]-min_grid[1])/(grid_resolution-1),(max_grid[2]-min_grid[2])/(grid_resolution-1)])
    print("size_voxel: ", size_voxel)

	# Create a volume grid to compute the scalar field for surface reconstruction
    scalar_field = np.zeros((grid_resolution,grid_resolution,grid_resolution),dtype = np.float32)

	# Compute the scalar field in the grid
    #scalar_field = compute_hoppe(points,normals,scalar_field,grid_resolution,min_grid,size_voxel)

    scalar_field = compute_imls(points,normals,scalar_field,grid_resolution,min_grid,size_voxel,30)

	# Compute the mesh from the scalar field based on marching cubes algorithm
    verts, faces, normals_tri, values_tri = measure.marching_cubes(scalar_field, level=0.0, spacing=(size_voxel,size_voxel,size_voxel))
    verts += min_grid

    # Export the mesh in ply using trimesh lib
    mesh = trimesh.Trimesh(vertices = verts, faces = faces)
    mesh.export(file_obj='../bunny_mesh_128_imls.ply', file_type='ply')

    print("Total time for surface reconstruction : ", time.time()-t0)
