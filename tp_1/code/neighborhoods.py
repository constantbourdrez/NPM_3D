#
#
#      0===========================================================0
#      |    TP1 Basic structures and operations on point clouds    |
#      0===========================================================0
#
#
# ------------------------------------------------------------------------------------------
#
#      Third script of the practical session. Neighborhoods in a point cloud
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

# Import functions from scikit-learn
from sklearn.neighbors import KDTree

# Import functions to read and write ply files
from ply import write_ply, read_ply

# Import time package
import time
import matplotlib.pyplot as plt


# ------------------------------------------------------------------------------------------
#
#           Functions
#       \***************/
#
#
#   Here you can define useful functions to be used in the main
#


def brute_force_spherical(queries, supports, radius):
    return [
        supports[np.linalg.norm(supports - query, axis=1) < radius] for query in queries
    ]


def brute_force_KNN(queries, supports, k):
    neighborhoods = []
    for query in queries:
        distances = np.linalg.norm(supports - query, axis=1)
        neighborhoods.append(supports[np.argpartition(distances, k)])

    return neighborhoods





# ------------------------------------------------------------------------------------------
#
#           Main
#       \**********/
#
#
#   Here you can define the instructions that are called when you execute this file
#

if __name__ == '__main__':

    # Load point cloud
    # ****************
    #
    #   Load the file '../data/indoor_scan.ply'
    #   (See read_ply function)
    #

    # Path of the file
    file_path = '../data/indoor_scan.ply'

    # Load point cloud
    data = read_ply(file_path)

    # Concatenate data
    points = np.vstack((data['x'], data['y'], data['z'])).T

    # Brute force neighborhoods
    # *************************
    #

    # If statement to skip this part if you want
    if False:

        # Define the search parameters
        neighbors_num = 100
        radius = 0.2
        num_queries = 10

        # Pick random queries
        random_indices = np.random.choice(points.shape[0], num_queries, replace=False)
        queries = points[random_indices, :]

        # Search spherical
        t0 = time.time()
        neighborhoods = brute_force_spherical(queries, points, radius)
        t1 = time.time()

        # Search KNN
        neighborhoods = brute_force_KNN(queries, points, neighbors_num)
        t2 = time.time()

        # Print timing results
        print('{:d} spherical neighborhoods computed in {:.3f} seconds'.format(num_queries, t1 - t0))
        print('{:d} KNN computed in {:.3f} seconds'.format(num_queries, t2 - t1))

        # Time to compute all neighborhoods in the cloud
        total_spherical_time = points.shape[0] * (t1 - t0) / num_queries
        total_KNN_time = points.shape[0] * (t2 - t1) / num_queries
        print('Computing spherical neighborhoods on whole cloud : {:.0f} hours'.format(total_spherical_time / 3600))
        print('Computing KNN on whole cloud : {:.0f} hours'.format(total_KNN_time / 3600))





    # KDTree neighborhoods
    # ********************
    #

    # If statement to skip this part if wanted
    if True:

        # Define the search parameters
        num_queries = 1000
        leaf_size = 40
        radius = 0.2

        # Pick random queries
        random_indices = np.random.choice(points.shape[0], num_queries, replace=False)
        queries = points[random_indices, :]
        print(queries.shape)
        supports = points
        # Implement spherical neigbourhood  using KDTree
        t0 = time.time()
        kdt = KDTree(supports, leaf_size=leaf_size)
        neighborhoods = kdt.query_radius(queries, radius)
        t1 = time.time()
        print('KDTree spherical neighborhoods computed in {:.3f} seconds'.format(t1 - t0))

        total_kd_tree_spherical_time = points.shape[0] * (t1 - t0) / num_queries
        print('Computing spherical neighborhoods on whole cloud : {:.0f} hours'.format(total_kd_tree_spherical_time / 3600))

        # Which leaf size allows the fastest spherical neighborhoods search?
        # Define the leaf sizes to be tested
        leaf_sizes = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000]
        for leaf_size in leaf_sizes:
            kdt = KDTree(supports, leaf_size=leaf_size)
            t0 = time.time()
            neighborhoods = kdt.query_radius(queries, radius)
            t1 = time.time()
            print('KDTree spherical neighborhoods computed in {:.3f} seconds with leaf size {:d}'.format(t1 - t0, leaf_size))


        optimal_leaf_size = 50
        radius_list = [0.1, 0.2, 0.5, 1, 2, 5, 10]
        time_list = []
        for radius in radius_list:
            kdt = KDTree(supports, leaf_size=optimal_leaf_size)
            t0 = time.time()
            neighborhoods = kdt.query_radius(queries, radius)
            t1 = time.time()
            time_list.append(t1 - t0)
        plt.figure()
        plt.plot(radius_list, time_list)
        plt.xlabel('Radius')
        plt.ylabel('Time (s)')
        plt.title('Time to compute spherical neighborhoods with KDTree')
        plt.show()

        # Time for 20cm radius in all cloud
        radius = 0.2
        kdt = KDTree(supports, leaf_size=optimal_leaf_size)
        t0 = time.time()
        neighborhoods = kdt.query_radius(queries, radius)
        t1 = time.time()
        total_kd_tree_spherical_time = points.shape[0] * (t1 - t0) / num_queries
        print('Computing spherical neighborhoods on whole cloud for 20cm radius: {:.0f} hours'.format(total_kd_tree_spherical_time / 3600))
