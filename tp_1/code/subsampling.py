#
#
#      0===========================================================0
#      |    TP1 Basic structures and operations on point clouds    |
#      0===========================================================0
#
#
# ------------------------------------------------------------------------------------------
#
#      Second script of the practical session. Subsampling of a point cloud
#
# ------------------------------------------------------------------------------------------
#
#      Hugues THOMAS - 13/12/2017
#


# ------------------------------------------------------------------------------------------
#
#          Imports and global variables
#      \**********************************/
#


# Import numpy package and name it "np"
import numpy as np

# Import functions to read and write ply files
from ply import write_ply, read_ply

# Import time package
import time


# ------------------------------------------------------------------------------------------
#
#           Functions
#       \***************/
#
#
#   Here you can define useful functions to be used in the main
#


def cloud_decimation(points, colors, labels, factor):

    decimated_points = points[::factor]
    decimated_colors = colors[::factor]
    decimated_labels = labels[::factor]

    return decimated_points, decimated_colors, decimated_labels

def grid_subsampling(points, colors, labels, voxel_size: float):
    """
    Performs a voxel subsampling on the point cloud.
    """
    nb_vox = np.ceil((np.max(points - np.min(points, axis=0), axis=0)) / voxel_size)
    print(
        f"Number of voxels: {'; '.join(nb_vox.astype(int).astype(str))} (total: {int(nb_vox.prod()):,})".replace(
            ",", " "
        ).replace(
            ";", ","
        )
    )
    non_empty_voxel_keys, inverse, nb_pts_per_voxel = np.unique(
        ((points - np.min(points, axis=0)) // voxel_size).astype(int),
        axis=0,
        return_inverse=True,
        return_counts=True,
    )
    print(f"Number of nonempty voxels: {len(non_empty_voxel_keys):,}".replace(",", " "))
    idx_pts_vox_sorted = np.argsort(inverse)

    subsampled_points = np.zeros((len(non_empty_voxel_keys), 3), points.dtype)
    subsampled_colors = np.zeros((len(non_empty_voxel_keys), 3), colors.dtype)
    subsampled_labels = np.zeros(len(non_empty_voxel_keys), labels.dtype)

    last_seen = 0
    for idx in range(len(non_empty_voxel_keys)):
        indexes_in_voxel = idx_pts_vox_sorted[
            last_seen : last_seen + nb_pts_per_voxel[idx]
        ]
        # barycenter of points
        subsampled_points[idx] = points[indexes_in_voxel].mean(axis=0)
        # color of the point closer to the center
        subsampled_colors[idx] = colors[indexes_in_voxel][
            np.linalg.norm(
                points[indexes_in_voxel] - points[indexes_in_voxel].mean(axis=0),
                axis=1,
            ).argmin()
        ]
        # most frequent label
        subsampled_labels[idx] = np.bincount(labels[indexes_in_voxel]).argmax()

        last_seen += nb_pts_per_voxel[idx]

    return subsampled_points, subsampled_colors, subsampled_labels




# ------------------------------------------------------------------------------------------
#
#           Main
#       \**********/
#
#
#   Here you can define the instructions that are called when you execute this file
#

if __name__ == '__main__':

    # Path of the file
    file_path = '../data/indoor_scan.ply'

    # Load point cloud
    data = read_ply(file_path)

    method = 'grid_subsampling'

    # Concatenate data
    points = np.vstack((data['x'], data['y'], data['z'])).T
    colors = np.vstack((data['red'], data['green'], data['blue'])).T
    labels = data['label']

    # Decimate the point cloud

    # Define the decimation factor
    factor = 300
    if method == 'grid_subsampling':
        # Define the size of the voxels
        voxel_size = 0.1

        # Decimate
        t0 = time.time()
        decimated_points, decimated_colors, decimated_labels = grid_subsampling(points, colors, labels, voxel_size)
        t1 = time.time()
        print('decimation done in {:.3f} seconds'.format(t1 - t0))
        # Save
        write_ply('../data/decimated_grid.ply', [decimated_points, decimated_colors, decimated_labels], ['x', 'y', 'z', 'red', 'green', 'blue', 'label'])
    else:
        # Decimate
        t0 = time.time()
        decimated_points, decimated_colors, decimated_labels = cloud_decimation(points, colors, labels, factor)
        t1 = time.time()
        print('decimation done in {:.3f} seconds'.format(t1 - t0))

        # Save
        write_ply('../data/decimated.ply', [decimated_points, decimated_colors, decimated_labels], ['x', 'y', 'z', 'red', 'green', 'blue', 'label'])


    print('Done')
