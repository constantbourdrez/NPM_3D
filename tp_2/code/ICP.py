#
#
#      0===================================0
#      |    TP2 Iterative Closest Point    |
#      0===================================0
#
#
#------------------------------------------------------------------------------------------
#
#      Script of the practical session
#
#------------------------------------------------------------------------------------------
#
#      Hugues THOMAS - 17/01/2018
#


#------------------------------------------------------------------------------------------
#
#          Imports and global variables
#      \**********************************/
#


# Import numpy package and name it "np"
import numpy as np

# Import library to plot in python
from matplotlib import pyplot as plt

# Import functions from scikit-learn
from sklearn.neighbors import KDTree

# Import functions to read and write ply files
from ply import write_ply, read_ply
from visu import show_ICP

import sys


#------------------------------------------------------------------------------------------
#
#           Functions
#       \***************/
#
#
#   Here you can define usefull functions to be used in the main
#

def RMS(data, ref):
    delta = ref - data
    delta_sq = np.sum(delta**2, axis=0)
    return np.mean(delta_sq)**0.5


def best_rigid_transform(data, ref):
    '''
    Computes the least-squares best-fit transform that maps corresponding points data to ref.
    Inputs :
        data = (d x N) matrix where "N" is the number of points and "d" the dimension
         ref = (d x N) matrix where "N" is the number of points and "d" the dimension
    Returns :
           R = (d x d) rotation matrix
           T = (d x 1) translation vector
           Such that R * data + T is aligned on ref
    '''

    # Calculate barycenters (mean points)
    p_m = np.mean(data, axis=1, keepdims=True)  # (d x 1)
    p_prime_m = np.mean(ref, axis=1, keepdims=True)  # (d x 1)

    # Compute centered clouds
    Q = data - p_m  # Centered data points (d x N)
    Q_prime = ref - p_prime_m  # Centered ref points (d x N)

    # Compute cross-covariance matrix
    H = Q @ Q_prime.T  # (d x d)

    # Perform Singular Value Decomposition
    U, _, Vt = np.linalg.svd(H)

    # Compute rotation matrix
    R = Vt.T @ U.T

    # Ensure a proper rotation (handle reflection)
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    # Compute translation vector
    T = p_prime_m - R @ p_m

    return R, T



def icp_point_to_point(data, ref, max_iter, RMS_threshold):
    '''
    Iterative closest point algorithm with a point to point strategy.
    Inputs :
        data = (d x N_data) matrix where "N_data" is the number of points and "d" the dimension
        ref = (d x N_ref) matrix where "N_ref" is the number of points and "d" the dimension
        max_iter = stop condition on the number of iterations
        RMS_threshold = stop condition on the distance
    Returns :
        data_aligned = data aligned on reference cloud
        R_list = list of the (d x d) rotation matrices found at each iteration
        T_list = list of the (d x 1) translation vectors found at each iteration
        neighbors_list = At each iteration, you search the nearest neighbors of each data point in
        the ref cloud and this obtain a (1 x N_data) array of indices. This is the list of those
        arrays at each iteration

    '''

    # Variable for aligned data
    # Variable for aligned data
    data_aligned = np.copy(data)

    # Initiate lists
    d, n = data.shape
    R_list = [np.eye(d)]
    T_list = [np.zeros((d,1))]
    neighbors_list = []
    RMS_list = []

    kdtree = KDTree(ref.T)
    for i in range(max_iter):
        _, neighbors = kdtree.query(data_aligned.T, k=1)
        neighbors = neighbors.squeeze()
        R, T = best_rigid_transform(data_aligned, ref[:,neighbors])

        T = R @ T_list[-1] + T
        R = R @ R_list[-1]

        T_list.append(T)
        R_list.append(R)
        neighbors_list.append(neighbors)

        data_aligned = R @ data + T

        rms = RMS(data_aligned, ref[:,neighbors].squeeze())
        RMS_list.append(rms)
        if rms < RMS_threshold:
            return data_aligned, R_list[1:], T_list[1:], neighbors_list, RMS_list
    return data_aligned, R_list[1:], T_list[1:], neighbors_list, RMS_list


def icp_point_to_point_fast(data, ref, max_iter, RMS_threshold, sampling_limit, final_overlap=1.):
    '''
    Iterative closest point algorithm with a point to point strategy.
    Inputs :
        data = (d x N_data) matrix where "N_data" is the number of points and "d" the dimension
        ref = (d x N_ref) matrix where "N_ref" is the number of points and "d" the dimension
        max_iter = stop condition on the number of iterations
        RMS_threshold = stop condition on the distance
        sampling_limit = maximum number of points to use to compute transformations
        final_overlap = overlap parameter
    Returns :
        data_aligned = data aligned on reference cloud
        R_list = list of the (d x d) rotation matrices found at each iteration
        T_list = list of the (d x 1) translation vectors found at each iteration
        neighbors_list = At each iteration, you search the nearest neighbors of each data point in
        the ref cloud and this obtain a (1 x N_data) array of indices. This is the list of those
        arrays at each iteration

    '''

    # Variable for aligned data
    data_aligned = np.copy(data)

    # Initiate lists
    d, n = data.shape
    R_list = [np.eye(d)]
    T_list = [np.zeros((d,1))]
    neighbors_list = []
    RMS_list = []

    kdtree = KDTree(ref.T)

    n_samples = min(n, sampling_limit)
    n_overlap = int(final_overlap * n_samples)

    for i in range(max_iter):
        # Sampling points
        data_idx = np.random.choice(n, n_samples, replace=False)

        # Matching points
        dist, neighbors = kdtree.query(data_aligned[:,data_idx].T, k=1)
        neighbors = neighbors.squeeze()
        if n_overlap != n_samples:
            dist = dist.squeeze()
            best_neighbors = np.argpartition(dist, n_overlap)[:n_overlap]
            neighbors = neighbors[best_neighbors]
            data_idx = data_idx[best_neighbors]

        # Estimating the best transform
        R, T = best_rigid_transform(data_aligned[:,data_idx], ref[:,neighbors])

        # Computing the full transform
        T = R @ T_list[-1] + T
        R = R @ R_list[-1]

        # Store everything
        T_list.append(T)
        R_list.append(R)
        neighbors_list.append(neighbors)

        # Aligne the data
        data_aligned = R @ data + T

        # Check the RMS threshold
        rms = RMS(data_aligned[:,data_idx], ref[:,neighbors].squeeze())
        RMS_list.append(rms)
        if rms < RMS_threshold:
            return data_aligned, R_list[1:], T_list[1:], neighbors_list, RMS_list
    return data_aligned, R_list[1:], T_list[1:], neighbors_list, RMS_list

#------------------------------------------------------------------------------------------
#
#           Main
#       \**********/
#
#
#   Here you can define the instructions that are called when you execute this file
#


if __name__ == '__main__':

    # Transformation estimation
    # *************************
    #

    # If statement to skip this part if wanted
    if False:

        # Cloud paths
        bunny_o_path = '../data/bunny_original.ply'
        bunny_r_path = '../data/bunny_returned.ply'

		# Load clouds
        bunny_o_ply = read_ply(bunny_o_path)
        bunny_r_ply = read_ply(bunny_r_path)
        bunny_o = np.vstack((bunny_o_ply['x'], bunny_o_ply['y'], bunny_o_ply['z']))
        bunny_r = np.vstack((bunny_r_ply['x'], bunny_r_ply['y'], bunny_r_ply['z']))

        # Find the best transformation
        R, T = best_rigid_transform(bunny_r, bunny_o)

        # Apply the tranformation
        bunny_r_opt = R.dot(bunny_r) + T

        # Save cloud
        write_ply('../bunny_r_opt', [bunny_r_opt.T], ['x', 'y', 'z'])

        # Compute RMS
        distances2_before = np.sum(np.power(bunny_r - bunny_o, 2), axis=0)
        RMS_before = np.sqrt(np.mean(distances2_before))
        distances2_after = np.sum(np.power(bunny_r_opt - bunny_o, 2), axis=0)
        RMS_after = np.sqrt(np.mean(distances2_after))

        print('Average RMS between points :')
        print('Before = {:.3f}'.format(RMS_before))
        print(' After = {:.3f}'.format(RMS_after))


    # Test ICP and visualize
    # **********************
    #

    # If statement to skip this part if wanted
    if False:

        # Cloud paths
        ref2D_path = '../data/ref2D.ply'
        data2D_path = '../data/data2D.ply'

        # Load clouds
        ref2D_ply = read_ply(ref2D_path)
        data2D_ply = read_ply(data2D_path)
        ref2D = np.vstack((ref2D_ply['x'], ref2D_ply['y']))
        data2D = np.vstack((data2D_ply['x'], data2D_ply['y']))

        # Apply ICP
        data2D_opt, R_list, T_list, neighbors_list, RMS_list = icp_point_to_point(data2D, ref2D, 20, 1e-4)

        # Show ICP
        show_ICP(data2D, ref2D, R_list, T_list, neighbors_list)

        # Plot RMS
        plt.plot(RMS_list)
        plt.show()


    # If statement to skip this part if wanted
    if False:

        # Cloud paths
        bunny_o_path = '../data/bunny_original.ply'
        bunny_p_path = '../data/bunny_perturbed.ply'

        # Load clouds
        bunny_o_ply = read_ply(bunny_o_path)
        bunny_p_ply = read_ply(bunny_p_path)
        bunny_o = np.vstack((bunny_o_ply['x'], bunny_o_ply['y'], bunny_o_ply['z']))
        bunny_p = np.vstack((bunny_p_ply['x'], bunny_p_ply['y'], bunny_p_ply['z']))

        # Apply ICP
        bunny_p_opt, R_list, T_list, neighbors_list, RMS_list = icp_point_to_point(bunny_p, bunny_o, 25, 1e-4)

        # Show ICP
        show_ICP(bunny_p, bunny_o, R_list, T_list, neighbors_list)

        # Plot RMS
        plt.plot(RMS_list)
        plt.xlabel('Iteration')
        plt.ylabel('RMS')
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    if True:

        # Cloud paths
        NDDC_1_path = '../data/Notre_Dame_Des_Champs_1.ply'
        NDDC_2_path = '../data/Notre_Dame_Des_Champs_2.ply'


        # Load point cloud
        ref = read_ply(NDDC_1_path)
        ref = np.vstack((ref['x'], ref['y'], ref['z']))
        data = read_ply(NDDC_2_path)
        data = np.vstack((data['x'], data['y'], data['z']))

        # Apply fast ICP for different values of the sampling_limit parameter
        RMS_list = []
        sampling_limit = [1000, 10000, 50000]
        for sl in sampling_limit:
            print("Number of sampling points {} ...".format(sl), end="")

            data_aligned,_,_,_, RMS_l = icp_point_to_point_fast(data, ref, 100, 1e-2, sl, final_overlap=0.4)
            RMS_list.append(RMS_l)
            print(" Done.")

        # Plot RMS

        for sl, rms in zip(sampling_limit, RMS_list):
            plt.plot(rms, label="{} samples".format(sl))
        plt.legend()
        plt.grid()
        plt.xlabel('Iteration')
        plt.ylabel('RMS')
        plt.show()


        write_ply('../NDDC_icp.ply', [data_aligned.T], ['x', 'y', 'z'])
