import cv2
import numpy as np
import math
import torch
import random

class Hog_descriptor():
    def __init__(self, img, cell_size=16, bin_size=8):
        self.img = img.astype(np.float32) / 255.0  # Normalize image to 0-1 range
        self.img = np.sqrt(self.img)
        self.cell_size = cell_size
        self.bin_size = bin_size
        self.angle_unit = 360 / self.bin_size
        assert type(self.bin_size) == int, "bin_size should be integer."
        assert type(self.cell_size) == int, "cell_size should be integer."
        assert self.angle_unit.is_integer(), "bin_size should be divisible by 360."

    def extract(self):
        height, width = self.img.shape
        gradient_magnitude, gradient_angle = self.global_gradient()
        gradient_magnitude = abs(gradient_magnitude)
        
        # Adjust cell_gradient_vector size according to the cell size
        cell_gradient_vector = np.zeros((int(height / self.cell_size), int(width / self.cell_size), self.bin_size))
        
        for i in range(cell_gradient_vector.shape[0]):
            for j in range(cell_gradient_vector.shape[1]):
                cell_magnitude = gradient_magnitude[i * self.cell_size:(i + 1) * self.cell_size,
                                                    j * self.cell_size:(j + 1) * self.cell_size]
                cell_angle = gradient_angle[i * self.cell_size:(i + 1) * self.cell_size,
                                            j * self.cell_size:(j + 1) * self.cell_size]
                cell_gradient_vector[i][j] = self.cell_gradient(cell_magnitude, cell_angle)

        # Flatten and concatenate block vectors to create a single feature vector
        hog_vector = self.flatten(cell_gradient_vector)
        hog_tensor = torch.from_numpy(hog_vector).float()
        hog_tensor = hog_tensor.unsqueeze(0)
        hog_tensor = hog_tensor.cpu()
        return hog_vector

    def global_gradient(self):
        gradient_values_x = cv2.Sobel(self.img, cv2.CV_64F, 1, 0, ksize=5)
        gradient_values_y = cv2.Sobel(self.img, cv2.CV_64F, 0, 1, ksize=5)
        gradient_magnitude = cv2.addWeighted(gradient_values_x, 0.5, gradient_values_y, 0.5, 0)
        gradient_angle = cv2.phase(gradient_values_x, gradient_values_y, angleInDegrees=True)
        return gradient_magnitude, gradient_angle

    def cell_gradient(self, cell_magnitude, cell_angle):
        orientation_centers = [0] * self.bin_size
        for i in range(cell_magnitude.shape[0]):
            for j in range(cell_magnitude.shape[1]):
                gradient_strength = cell_magnitude[i][j]
                gradient_angle = cell_angle[i][j]
                min_angle, max_angle, mod = self.get_closest_bins(gradient_angle)
                orientation_centers[min_angle] += (gradient_strength * (1 - (mod / self.angle_unit)))
                orientation_centers[max_angle] += (gradient_strength * (mod / self.angle_unit))
        return orientation_centers

    def get_closest_bins(self, gradient_angle):
        idx = int(gradient_angle / self.angle_unit)
        mod = gradient_angle % self.angle_unit
        return idx, (idx + 1) % self.bin_size, mod

    def flatten(self, cell_gradient_vector):
        # Flatten and concatenate all block vectors
        hog_vector = cell_gradient_vector.flatten()
        return hog_vector

# Example usage:
# img = cv2.imread('your_image_path.png', cv2.IMREAD_GRAYSCALE)
# hog_descriptor = Hog_descriptor(img, cell_size=16, bin_size=8)
# hog_vector = hog_descriptor.extract()
# print(hog_vector.shape)


class SIFT_descriptor:
    def __init__(self, img, descriptor_size=128):
        self.img = img
        self.sift = cv2.SIFT_create(nfeatures=500, nOctaveLayers=3, contrastThreshold=0.04, edgeThreshold=10, sigma=1.6)
        self.descriptor_size = descriptor_size  # expected descriptor size

    def extract(self):
        # Detect keypoints and descriptors
        keypoints, descriptors = self.sift.detectAndCompute(self.img, None)
        # If no keypoint is detected, create a dummy keypoint (center of image)
        if not keypoints:
            h, w = self.img.shape
            keypoint = cv2.KeyPoint(w / 2, h / 2, 10)  # Use the image center as the key point location, _size is the key point neighborhood diameter
            keypoints = [keypoint]
            _, descriptors = self.sift.compute(self.img, keypoints)

        # Normalize descriptors to between 0 and 1
        if descriptors is not None:
            descriptors = self.normalize_descriptors(descriptors)
        # Convert to PyTorch Tensor
        descriptors_tensor = torch.tensor(descriptors, dtype=torch.float32).view(1, -1)  # Adjust the shape to [1, n]

        return descriptors_tensor

    def normalize_descriptors(self, descriptors):
        # min-max normalization
        descriptors_min = np.min(descriptors, axis=1, keepdims=True)
        descriptors_max = np.max(descriptors, axis=1, keepdims=True)
        normalized_descriptors = (descriptors - descriptors_min) / (descriptors_max - descriptors_min + 1e-16)  # Prevent division by 0

        # Perform average pooling to get a fixed length vector
        pooled_descriptors = np.mean(normalized_descriptors, axis=0)
        return pooled_descriptors.flatten()  # Flatten for later processing

# # Example usage:
# # img = cv2.imread('path_to_your_image.png', cv2.IMREAD_GRAYSCALE)
# # sift_descriptor = SIFTDescriptor(img)
# # descriptors_tensor = sift_descriptor.extract()
# # print(descriptors_tensor.shape)


class SIFT_descriptor:
    def __init__(self, img, n_keypoints=20):
        self.img = img
        self.n_keypoints = n_keypoints
        self.sift = cv2.SIFT_create(nfeatures=500, nOctaveLayers=3, contrastThreshold=0.04, edgeThreshold=10, sigma=1.6)

    def extract(self):
        keypoints, descriptors = self.sift.detectAndCompute(self.img, None)
        if not isinstance(keypoints, list):
            keypoints = list(keypoints)
        # Supplement key points when the number of detected key points is insufficient
        if len(keypoints) < self.n_keypoints:
            additional_keypoints = self.generate_additional_keypoints(len(keypoints))
            # keypoints += additional_keypoints
            keypoints.extend(additional_keypoints)
            _, descriptors = self.sift.compute(self.img, keypoints)       
        # Keep only the descriptors of the first n keypoints
        descriptors = descriptors[:self.n_keypoints, :]
        # If the number of descriptors is less than n, pad it with zeros to n*128
        if descriptors.shape[0] < self.n_keypoints:
            zero_padding = np.zeros((self.n_keypoints - descriptors.shape[0], 128))
            descriptors = np.vstack([descriptors, zero_padding])

        descriptors_norm = self.normalize_descriptors(descriptors)
        # Flatten the descriptor into a 1*128n vector
        descriptors_flattened = descriptors_norm.flatten().reshape(1, -1)
        # Convert to PyTorch Tensor
        descriptors_tensor = torch.tensor(descriptors_flattened, dtype=torch.float32)
        return descriptors_tensor

    def normalize_descriptors(self, descriptors):
        # min-max normalization
        descriptors_min = np.min(descriptors, axis=1, keepdims=True)
        descriptors_max = np.max(descriptors, axis=1, keepdims=True)
        normalized_descriptors = (descriptors - descriptors_min) / (descriptors_max - descriptors_min + 1e-16)  # Prevent division by 0
        return normalized_descriptors  # Flatten for later processing

    def generate_additional_keypoints(self, existing_keypoints_count):
        additional_keypoints = []
        needed_keypoints = self.n_keypoints - existing_keypoints_count
        h, w = self.img.shape
        for _ in range(needed_keypoints):
            # Randomly select a position on the diagonal from the lower left corner to the upper right corner
            ratio = random.random()
            x = ratio * w
            y = (1 - ratio) * h
            keypoint = cv2.KeyPoint(x, y, 10)
            additional_keypoints.append(keypoint)
        return additional_keypoints

# 示例使用
# img = cv2.imread('path_to_your_image.png', cv2.IMREAD_GRAYSCALE)
# sift_descriptor = SIFTDescriptorWithFixedKeypoints(img, n_keypoints=20)
# descriptors_tensor = sift_descriptor.extract()
# print(descriptors_tensor.shape)
