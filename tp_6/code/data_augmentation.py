import os
import math
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from ply import read_ply


class RandomRotation(object):
    """
    Applies a random rotation to a point cloud along a specified axis.
    """
    def __init__(self, axis: str):
        assert axis in ['x', 'y', 'z'], "Axis must be 'x', 'y', or 'z'."
        self.axis = axis

    def __call__(self, point_cloud):
        theta = random.uniform(0, 2 * math.pi)
        cos_theta, sin_theta = math.cos(theta), math.sin(theta)

        if self.axis == 'x':
            rot_matrix = np.array([
                [1, 0, 0],
                [0, cos_theta, -sin_theta],
                [0, sin_theta, cos_theta]
            ])
        elif self.axis == 'y':
            rot_matrix = np.array([
                [cos_theta, 0, -sin_theta],
                [0, 1, 0],
                [sin_theta, 0, cos_theta]
            ])
        else:  # self.axis == 'z'
            rot_matrix = np.array([
                [cos_theta, -sin_theta, 0],
                [sin_theta, cos_theta, 0],
                [0, 0, 1]
            ])

        return point_cloud @ rot_matrix.T


class RandomNoise(object):
    """
    Adds Gaussian noise to the point cloud.
    """
    def __init__(self, std_dev=0.02):
        self.std_dev = std_dev

    def __call__(self, point_cloud):
        noise = np.random.normal(0, self.std_dev, point_cloud.shape)
        return point_cloud + noise

class AnisotropicScale(object):
    """
    This applies slight random rescale in each axis

    """

    def __call__(self, pointcloud):
        scale_factor = 0.1
        scale = np.random.uniform(1-scale_factor, 1+scale_factor, (3))
        scale_matrix = np.array([[scale[0], 0, 0],
                                [0, scale[1], 0],
                                [0, 0, scale[2]]])
        scaled_pointcloud = scale_matrix.dot(pointcloud.T).T
        return scaled_pointcloud


class RandomRepeat(object):
    """
    This randomly removes a proportion of points, and duplicate
    others to keep the same number of points.


    """

    def __call__(self, pointcloud):
        ablation_rate = 0.1
        np.random.shuffle(pointcloud)
        repeat_size = np.random.randint(0, int(ablation_rate*1024))
        if repeat_size > 0:
            pointcloud[-repeat_size:] = pointcloud[:repeat_size]
        return pointcloud


class RandomSymmetry(object):
    """
    Flips the point cloud along a random axis with a given probability.
    """
    def __init__(self, probability=0.5):
        self.probability = probability

    def __call__(self, point_cloud):
        axis = random.choice([0, 1])  # Flip along x or y axis
        if random.random() < self.probability:
            point_cloud[:, axis] = np.max(point_cloud[:, axis]) - point_cloud[:, axis]
        return point_cloud


class ToTensor(object):
    """
    Converts a NumPy array to a PyTorch tensor.
    """
    def __call__(self, point_cloud):
        return torch.tensor(point_cloud, dtype=torch.float32)




def default_transforms():
    return transforms.Compose([
        RandomRotation('z'),
        RandomNoise(),
        ToTensor(),
    ])


def custom_transforms():
    return transforms.Compose([RandomRepeat(), RandomRotation("z"), RandomNoise(), AnisotropicScale(), ToTensor()])


def test_transforms():
    return transforms.Compose([ToTensor()])


class PointCloudDataAugmented(Dataset):
    """
    Loads point cloud data from .ply files into RAM.
    """
    def __init__(self, root_dir, folder="train", transform=None):
        self.root_dir = root_dir
        self.transform = transform or default_transforms()
        self.data = []

        # Gather class names from directory structure
        self.classes = {folder: i for i, folder in enumerate(sorted(os.listdir(root_dir)))}

        for category, class_idx in self.classes.items():
            category_dir = os.path.join(root_dir, category, folder)
            if not os.path.isdir(category_dir):
                continue

            for file in os.listdir(category_dir):
                if file.endswith(".ply"):
                    ply_path = os.path.join(category_dir, file)
                    data = read_ply(ply_path)
                    point_cloud = np.vstack((data["x"], data["y"], data["z"])).T
                    self.data.append({"pointcloud": point_cloud, "category": class_idx})

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        point_cloud = self.transform(sample["pointcloud"])
        return {"pointcloud": point_cloud, "category": sample["category"]}
