#
#
#      0===========================================================0
#      |                      TP6 Modelisation                     |
#      0===========================================================0
#
#
#------------------------------------------------------------------------------------------
#
#      Plane detection with RANSAC
#
#------------------------------------------------------------------------------------------
#
#      Xavier ROYNARD - 19/02/2018
#


#------------------------------------------------------------------------------------------
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
from descriptors import compute_local_PCA


#------------------------------------------------------------------------------------------
#
#           Functions
#       \***************/
#
#
#   Here you can define usefull functions to be used in the main
#


def compute_plane(points):

    point_plane = np.zeros((3,1))
    normal_plane = np.zeros((3,1))

    vec1 = points[1] - points[0]
    vec2 = points[2] - points[0]
    vec3 = np.cross(vec1, vec2)
    #print(vec1, vec2, vec3)
    normal_plane = (vec3 / np.linalg.norm(vec3)).reshape(3,1)
    point_plane = points[0].reshape(3,1)

    return point_plane, normal_plane





def in_plane(points, pt_plane, normal_plane, threshold_in=0.1):

    indexes = np.zeros(len(points), dtype=bool)

    dist = np.abs((points - pt_plane.T) @ normal_plane)
    indexes = dist < threshold_in

    return indexes

def in_plane_normal(points, pt_plane, normal_plane, normals, threshold_in=0.1, threshold_angle = np.pi/3):

    indexes = np.zeros(len(points), dtype=bool)

    dist = np.abs((points - pt_plane.T) @ normal_plane)
    angle = np.arccos(np.clip(np.abs(normals @ normal_plane), -1, 1))
    indexes = (dist < threshold_in) & (angle < threshold_angle)

    return indexes

def RANSAC(points, nb_draws=100, threshold_in=0.1, normal_set = False, threshold_angle = np.pi/3, normals = None):
    best_vote = 3
    best_pt_plane = np.zeros((3,1))
    best_normal_plane = np.zeros((3,1))

    for i in range(nb_draws):

            # Take randomly three points
            pts = points[np.random.randint(0, len(points), size=3)]

            # Computes the plane passing through the 3 points
            pt_plane, normal_plane = compute_plane(pts)

            # Find points in the plane and others
            if normal_set:
                points_in_plane = in_plane_normal(points, pt_plane, normal_plane, normals, threshold_in, threshold_angle= threshold_angle)
            else:
                points_in_plane = in_plane(points, pt_plane, normal_plane, threshold_in)
            #plane_inds = points_in_plane.nonzero()[0]
            #remaining_inds = (1-points_in_plane).nonzero()[0]

            # Count the number of points in the plane
            vote = np.sum(points_in_plane)

            # Update best plane
            if vote > best_vote:
                best_vote = vote
                best_pt_plane = pt_plane
                best_normal_plane = normal_plane

    return best_pt_plane, best_normal_plane, best_vote




def recursive_RANSAC(points, nb_draws=100, threshold_in=0.1, nb_planes=2):

    nb_points = len(points)
    plane_inds = np.arange(0,0)
    plane_labels = np.arange(0,0)
    remaining_inds = np.arange(0,nb_points)

    for i in range(nb_planes):

            # Find best plane by RANSAC
            best_pt_plane, best_normal_plane, best_vote = RANSAC(points[remaining_inds], nb_draws, threshold_in)

            # Find points in the plane and others
            points_in_plane = in_plane(points[remaining_inds], best_pt_plane, best_normal_plane, threshold_in).squeeze()
            plane_inds = np.concatenate((plane_inds, remaining_inds[points_in_plane]))
            plane_labels = np.concatenate((plane_labels, i*np.ones(np.sum(points_in_plane), dtype=int)))
            remaining_inds = remaining_inds[(1-points_in_plane).astype(bool)]

    return plane_inds, remaining_inds, plane_labels

def ortogonal_recursive_RANSAC(points, nb_draws=100, threshold_in=0.1, threshold_angle = np.pi/3, nb_planes=2):

    nb_points = len(points)
    plane_inds = np.arange(0,0)
    plane_labels = np.arange(0,0)
    remaining_inds = np.arange(0,nb_points)
    normals = compute_local_PCA(points, points, radius=0.5, nghbrd_search='knn', k=30)[1][:, :, 0]
    print('Normals computed')
    for i in range(nb_planes):

            # Find best plane by RANSAC
            best_pt_plane, best_normal_plane, best_vote = RANSAC(points[remaining_inds], nb_draws, threshold_in, normal_set=True, threshold_angle=threshold_angle, normals = normals[remaining_inds])

            # Find points in the plane and others
            points_in_plane = in_plane_normal(points[remaining_inds], best_pt_plane, best_normal_plane, normals[remaining_inds], threshold_in, threshold_angle).squeeze()


            plane_inds = np.concatenate((plane_inds, remaining_inds[points_in_plane]))
            plane_labels = np.concatenate((plane_labels, i*np.ones(np.sum(points_in_plane), dtype=int)))
            remaining_inds = remaining_inds[(1-points_in_plane).astype(bool)]

    return plane_inds, remaining_inds, plane_labels



#------------------------------------------------------------------------------------------
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
    #Load the file '../data/indoor_scan.ply'
    #   (See read_ply function)
    #

    # Path of the file
    file_path = '../data/Lille_street_small.ply'

    # Load point cloud
    data = read_ply(file_path)

    # Concatenate data
    points = np.vstack((data['x'], data['y'], data['z'])).T
    #colors = np.vstack((data['red'], data['green'], data['blue'])).T
    #labels = data['label']
    nb_points = len(points)


    # Computes the plane passing through 3 randomly chosen points
    # ************************
    #

    print('\n--- 1) and 2) ---\n')

    # Define parameter
    threshold_in = 0.1

    # Take randomly three points
    pts = points[np.random.randint(0, nb_points, size=3)]

    # Computes the plane passing through the 3 points
    t0 = time.time()
    pt_plane, normal_plane = compute_plane(pts)
    t1 = time.time()
    print('plane computation done in {:.3f} seconds'.format(t1 - t0))

    # Find points in the plane and others
    t0 = time.time()
    points_in_plane = in_plane(points, pt_plane, normal_plane, threshold_in)
    t1 = time.time()
    print('plane extraction done in {:.3f} seconds'.format(t1 - t0))
    plane_inds = points_in_plane.nonzero()[0]
    remaining_inds = (1-points_in_plane).nonzero()[0]

    # Save extracted plane and remaining points
    #write_ply('./plane.ply', [points[plane_inds], colors[plane_inds], labels[plane_inds]], ['x', 'y', 'z', 'red', 'green', 'blue', 'label'])
    #write_ply('./remaining_points_plane.ply', [points[remaining_inds], colors[remaining_inds], labels[remaining_inds]], ['x', 'y', 'z', 'red', 'green', 'blue', 'label'])


    # Computes the best plane fitting the point cloud
    # ***********************************
    #
    #

    print('\n--- 3) ---\n')

    # Define parameters of RANSAC
    nb_draws = 500
    threshold_in = 0.10

    # Find best plane by RANSAC
    t0 = time.time()
    best_pt_plane, best_normal_plane, best_vote = RANSAC(points, nb_draws, threshold_in)
    t1 = time.time()
    print('RANSAC done in {:.3f} seconds'.format(t1 - t0))

    # Find points in the plane and others
    points_in_plane = in_plane(points, best_pt_plane, best_normal_plane, threshold_in)
    plane_inds = points_in_plane.nonzero()[0]
    remaining_inds = (1-points_in_plane).nonzero()[0]

    # Save the best extracted plane and remaining points
    #write_ply('./best_plane.ply', [points[plane_inds], colors[plane_inds], labels[plane_inds]], ['x', 'y', 'z', 'red', 'green', 'blue', 'label'])
    #write_ply('./remaining_points_best_plane.ply', [points[remaining_inds], colors[remaining_inds], labels[remaining_inds]], ['x', 'y', 'z', 'red', 'green', 'blue', 'label'])


    # Find "all planes" in the cloud
    # ***********************************
    #
    #

    print('\n--- 4) ---\n')

    # Define parameters of recursive_RANSAC
    nb_draws = 500
    threshold_in = 0.10
    nb_planes = 5
    threshold_angle = 0.05

    # Recursively find best plane by RANSAC
    t0 = time.time()
    plane_inds, remaining_inds, plane_labels = ortogonal_recursive_RANSAC(points, nb_draws, threshold_in, threshold_angle, nb_planes)
    t1 = time.time()
    print('recursive RANSAC done in {:.3f} seconds'.format(t1 - t0))

    color_label = np.zeros((len(plane_inds), 3))
    s = 0
    color_label[:np.sum(plane_labels == 0)] = [255, 0, 0]  # Red
    s += np.sum(plane_labels == 0)
    color_label[s:s+np.sum(plane_labels == 1)] = [0, 255, 0]  # Green
    s += np.sum(plane_labels == 1)
    color_label[s:s+np.sum(plane_labels == 2)] = [0, 0, 255]  # Blue
    s += np.sum(plane_labels == 2)
    color_label[s:s+np.sum(plane_labels == 3)] = [255, 255, 0]  # Yellow
    s += np.sum(plane_labels == 3)
    color_label[s:s+np.sum(plane_labels == 4)] = [0, 255, 255]  # Cyan





    print("points shape:", points[plane_inds].shape)
    print("color_label shape:", color_label.shape)
    #print("labels shape:", labels[plane_inds].shape)
    print("plane_labels shape:", plane_labels.shape)

    # Save the best planes and remaining points
    write_ply('./best_planes_2.ply', [points[plane_inds], color_label, plane_labels.astype(np.int32)], ['x', 'y', 'z', 'red', 'green', 'blue', 'plane_label'])
    #write_ply('./best_planes_2.ply', [points[plane_inds], color_label, labels[plane_inds], plane_labels.astype(np.int32)], ['x', 'y', 'z', 'red', 'green', 'blue', 'label', 'plane_label'])
   # write_ply('./remaining_points_best_planes_2.ply', [points[remaining_inds], colors[remaining_inds], labels[remaining_inds]], ['x', 'y', 'z', 'red', 'green', 'blue', 'label'])

    print('Done')
