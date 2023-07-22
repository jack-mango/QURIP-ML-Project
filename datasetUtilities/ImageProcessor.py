"""
Author: Jack Mango
Contact: jackmango@berkeley.edu
Date: 2023-07-21
Description: A class used to perform image processing tasks.
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
import warnings
from datasetUtilities.AutoGauss import Gaussian2D, GaussianMixture
from datasetUtilities.AnalysisWarning import AnalysisWarning

import logging
log = logging.getLogger(__name__)

class ImageProcessor():
    """
    A class for various image and tweezer array geometry manipulation tasks.

    Attributes:
        n_tweezers (int): The number of tweezers in the stack images.
        n_loops (int): Number of loops in the image stack.
        r_atom (float): approximate radius of atom in pixels.
        positions (numpy.ndarray): Coordinates of tweezers in the stack. Should have the shape [# of tweezers, 2].
        nn_dist (float): The nearest neighbor distance of two tweezers given by the positions.
        make_plots (bool): If True, a plot of the tweezer positions will be automatically generated when finding the positions of tweezers.
        info (dict): The relevant information gathered when executing this pipeline stage as a dictionary.
    """

    def __init__(self, n_tweezers, n_loops, r_atom=2.5, tweezer_positions=None, make_plots=True):
        """
        Initialize the ImageProcessor with the provided parameters.

        Parameters:
            n_tweezers (int): The number of tweezers in the stack images.
            n_loops (int): Number of loops in the image stack.
            r_atom (float, optional): approximate radius of atom in pixels.
            tweezer_positions (numpy.ndarray, optional): Coordinates of tweezers in the stack. Should have the shape [# of tweezers, 2]. Default is None.
            make_plots (bool, optional): If True, a plot of the tweezer positions will be automatically generated when finding the positions of tweezers. Default is True.
        """
        self.n_tweezers = n_tweezers
        self.n_loops = n_loops
        self.r_atom = r_atom
        self.positions = tweezer_positions
        self.nn_dist = None
        self.make_plots = make_plots
        if not tweezer_positions is None:
            self.find_nn_dist()
        self.info = {}

    def pixel(self, x):
        """
        Find the nearest pixel value(s) to x.

        Parameters:
            x (scalar or numpy.ndarray): Pixel value(s).

        Returns:
            numpy.ndarray: The closest pixel value(s) to x.
        """
        return np.rint(x).astype(int)
    
    def find_tweezer_positions(self, stack):
        """
        Given an array of 2D images, find the sorted locations of tweezer sites using the find_centroids method. Additionally finds the nearest
        neighbor distance, which is used for sorting the positions.

        Parameters:
            stack (numpy.ndarray): A numpy array of 2D images. This function expects the array to have shape [# of images, image width, image height].

        Returns:
            numpy.ndarray: An array containing coordinates of each tweezer position. The shape of the array is [# of positions, 2].
        """
        log.info("Finidng tweezer positions...")
        self.positions = self.find_centroids(stack)
        if self.positions.shape[0] != self.n_tweezers:
            warnings.warn(f"User specified {self.n_tweezers} tweezers; only found {self.positions.shape[0]}!", AnalysisWarning)     
        self.nn_dist = self.find_nn_dist()
        self.position_tile_sort(stack.shape[-1], self.nn_dist)
        if self.make_plots:
            self.plot(stack)
        return self.positions

    def find_centroids(self, stack):
        """
        Attempt to find the locations of tweezers in an image using image processing methods
        and connected component analysis.

        Parameters:
            stack (numpy.ndarray): A numpy array of 2D images. This function expects the array to have shape [# of images, image width, image height].

        Returns:
            numpy.ndarray: The centroids associated with each tweezer in a self.n_tweezers x 2 array, where the ith entry corresponds to the coordinates of the ith tweezer.
        """
        frac_stack = self.fractional_stack(stack, 0.1)
        frac_stack = self.to_uint8_img(frac_stack)
        final = np.zeros(frac_stack.shape[1:], dtype='uint8')
        model = GaussianMixture(frac_stack.flatten())
        params, r_sq = model.fit()
        dark_params, bright_params = params
        thresh = -4 * dark_params[1]
        block_size = int(4 * (self.r_atom // 2) + 5)
        for img in frac_stack:
            img = cv2.adaptiveThreshold(
                img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, block_size, thresh)
            n_sites, labeled_img, stats, centroids = cv2.connectedComponentsWithStats(img)
            for stat in stats:
                if stat[-1] <= 1:
                    img[stat[1]:stat[1] + stat[3], stat[0]:stat[0] +
                        stat[2]] = np.zeros((stat[3], stat[2]))
            final = np.maximum(img, final)
        n_sites, labeled_img, stats, centroids = cv2.connectedComponentsWithStats(
            final)
        centroids = centroids[np.argsort(stats[:, -1])]
        return centroids[-2:-self.n_tweezers - 2:-1, ::-1]
    
    def to_uint8_img(self, img):
        """
        Convert an image to the uint8 data type, mapping the lowest pixel value to zero and the highest to 255.

        Parameters:
            img (numpy.ndarray): The input image to be converted.

        Returns:
            numpy.ndarray: The converted image of uint8 data type.

        Note:
            The input image is expected to be a numpy ndarray. It can include multiple images.
        """
        img = 255 * (img - img.min()) / (img.max() - img.min())
        return img.astype('uint8')

    def fractional_stack(self, stack, fraction):
        """
        Separate the first fraction of images from the start of every loop, rounding up to the largest
        integer number of images.

        Parameters:
            stack (numpy.ndarray): A numpy array of 2D images. This function expects the array to have shape [# of images, image width, image height].
            fraction (float): The fraction of images in each loop to be returned. This should fall between zero and one

        Returns:
            numpy.ndarray: The first fraction of images from each loop in the stack.
        """
        per_loop = stack.shape[0] // self.n_loops
        img_width, img_height = stack.shape[-2:]
        slices = np.empty((self.n_loops, img_width, img_height))
        slice_size = np.ceil(fraction * per_loop).astype(int)
        for i in range(self.n_loops):
            slices[i] = np.mean(
                stack[i * per_loop: i * per_loop + slice_size], axis=0)
        return slices

    def find_nn_dist(self):
        """
        Find the "nearest neighbor distance" given a list of coordinates.

        Returns:
            float: The nearest neighbor distance based on the above criterion.
        """
        min_dist, closest_pair = self.closest_pair_distance(self.positions)
        nn_dist = np.max(np.absolute(np.diff(closest_pair, axis=0)))
        return nn_dist

    def closest_pair_bf(self, points):
        """
        Find the two closest points using the L2 distance metric (brute force approach).

        Parameters:
            points (numpy.ndarray): An m x 2 array of coordinates corresponding to each tweezer, where the ith entry corresponds to the coordinates of the ith tweezer.

        Returns:
            float: The distance between the closest two points.
            tuple: A tuple containing the coordinates of the two closest points.
        """
        min_distance = np.inf
        closest_pair = (None, None)
        for i in range(len(points) - 1):
            for j in range(i + 1, len(points)):
                dist = np.linalg.norm(
                    points[i] - points[j])
                if dist < min_distance:
                    min_distance = dist
                    closest_pair = (
                        points[i], points[j])
        return min_distance, closest_pair

    def closest_pair_distance(self, points):
        """
        Find the two closest points using the L2 distance metric (recursive divide and conquer approach).

        Parameters:
            points (numpy.ndarray): An m x 2 array of coordinates corresponding to each tweezer, where the ith entry corresponds to the coordinates of the ith tweezer.

        Returns:
            float: The distance between the closest two points.
            tuple: A tuple containing the coordinates of the two closest points.
        """
        if len(points) <= 3:
            return self.closest_pair_bf(points)
        points = points[points[:, 0].argsort(
        )]
        mid_point = len(points) // 2
        left_points = points[:mid_point]
        right_points = points[mid_point:]
        min_dist_left, closest_left = self.closest_pair_distance(left_points)
        min_dist_right, closest_right = self.closest_pair_distance(
            right_points)
        if min_dist_left < min_dist_right:
            min_distance = min_dist_left
            closest_pair = closest_left
        else:
            min_distance = min_dist_right
            closest_pair = closest_right
        strip_points = points[np.abs(
            points[:, 0] - points[mid_point, 0]) < min_distance]
        strip_points = strip_points[strip_points[:, 1].argsort()]
        min_dist_strip = min_distance
        closest_strip = (None, None)
        for i in range(len(strip_points)):
            j = i + 1
            while j < len(strip_points) and strip_points[j, 1] - strip_points[i, 1] < min_dist_strip:
                dist = np.linalg.norm(strip_points[i] - strip_points[j])
                if dist < min_dist_strip:
                    min_dist_strip = dist
                    closest_strip = (strip_points[i], strip_points[j])
                j += 1
        if min_dist_strip < min_distance:
            return min_dist_strip, closest_strip
        else:
            return min_distance, closest_pair

    def crop(self, stack, x, y, border):
        """
        Return the portion of the stack centered at the pixel corresponding to (x, y) with horizontal
        and vertical padding of pixel values corresponding to h_border and v_border respectively.

        Parameters:
            stack (numpy.ndarray): A numpy array of 2D images. This function expects the array to have shape [# of images, image width, image height].
            x (float or int): x-coordinate of crop center.
            y (float or int): y-coordinate of crop center.
            border (int): An amount of padding to add to either side of the central pixel specified by x and y.

        Returns:
            numpy.ndarray: An array of every crop from every image in the stack corresponding to the parameters given.
        """
        return stack[:, self.pixel(x - border): self.pixel(x + border) + 1,
                          self.pixel(y - border): self.pixel(y + border) + 1]

    def crop_tweezers(self, stack, side_length, separation=None):
        """
        Create square crops for every tweezer in the stack.

        Parameters:
            stack (numpy.ndarray): A numpy array of 2D images. This function expects the array to have shape [# of images, image width, image height].
            side_length (int): How many nearest neighbor distances each side of the square crop should have. If the tweezers are in a rectangular lattice, then each crop would contain at most side_length x side_length tweezers.
            separation (float, optional): Number of pixels per side_length unit. Default is None.

        Returns:
            numpy.ndarray: An array of every crop of every tweezer. Consecutive crops in this array correspond to those taken around the same center, and the same loop corresponding in the same center.
        """
        if separation is None:
            separation = self.nn_dist
        border = self.pixel(side_length * separation / 2)
        crop_size = 2 * border + 1
        stack = np.reshape(stack, (-1, *stack.shape[-2:]))
        crops = np.empty((self.n_tweezers, stack.shape[0], crop_size, crop_size))
        for i, pos in enumerate(self.positions):
            crops[i] = self.crop(stack, *pos, border)
        return crops

    def plot(self, stack):
        """
        Plot the given position coordinates on top of stack averaged over all images.

        Parameters:
            stack (numpy.ndarray): A numpy array of 2D images. This function expects the array to have shape [# of images, image width, image height].

        Returns:
            matplotlib.figure.Figure: A matplotlib figure of the plot generated by this function.
        """
        fig = plt.figure()
        img = stack.mean(axis=0)
        img = plt.imshow(img.T, cmap='viridis')
        plt.plot(*self.positions.T, 'ro', fillstyle='none')
        plt.colorbar()
        plt.title("Tweezer Positions")
        return fig

    def fit_gaussian_to_image(self, image_data):
        """
        Fit a two-dimensional Gaussian to a two-dimensional array representing pixel intensities for an image.

        Parameters:
            image_data (numpy.ndarray): A two-dimensional array with entries represented by pixel values.

        Returns:
            tuple: The parameters for the two-dimensional Gaussian fit.
            numpy.ndarray: A normalized array of the same size as the image data with values sampled calculated from the fitted function, considering the pixel indices as (x, y) coordinates.

        Note: Currently unused in this version.
        """
        model = AutoGauss.Gaussian2D(image_data)
        params = model.fit()
        weights = model.func(model.make_coordinates(), 1, *params, 0)
        return np.array(params[:2]), weights
    
    def position_tile_sort(self, img_size, tile_size):
        """
        Sort an array of position vectors based on which square tile they fall into.

        Parameters:
            img_size (int): The side length of the image the positions are to be sorted on.
            tile_size (int): The side length of the square tile used to divide the rectangular region into tiles.

        Returns:
            numpy.ndarray: An m x 2 array of the vectors sorted based on which tile they fall into. If two vectors fall in the same tile, then the order is ambiguous.
        """
        num_tiles_x = img_size // tile_size
        tile_indices = np.floor(self.positions / tile_size).astype(int)
        tile_numbers = tile_indices[:, 1] * num_tiles_x + tile_indices[:, 0]
        sorted_indices = np.argsort(tile_numbers)
        sorted_vectors = self.positions[sorted_indices]
        self.positions = sorted_vectors
        return sorted_vectors
    
    def transform_positions(self, src_stack, target_stack):
        """
        Calculate the corresponding position coordinates of tweezers in a target_stack given a src_stack using an affine transform.

        Parameters:
            src_stack (numpy.ndarray): An array of 2D images whose coordinates will be used to calculate the coordinates of tweezers in the target stack.
            target_stack (numpy.ndarray): An array of 2D images. The locations of tweezers in this stack will be calculated.

        Returns:
            numpy.ndarray: An array of coordinates for tweezers in the target stack.
        """
        src_positions = self.find_tweezer_positions(src_stack)
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        orb = cv2.ORB_create(edgeThreshold=10)
        src_pts = []
        target_pts = []
        for i in np.random.choice(src_stack.shape[0], 10, replace=False):
            src_img = self.to_uint8_img(src_stack[i])
            target_img = self.to_uint8_img(target_stack[i])
            block_size = int(2 * (self.r_atom // 2) + 1)
            src_img = cv2.bilateralFilter(src_img, block_size, 25, 25)
            target_img = cv2.bilateralFilter(target_img, block_size, 25, 25)
            src_kp, src_desc = orb.detectAndCompute(src_img.T, None)
            target_kp, target_desc = orb.detectAndCompute(target_img.T, None)
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            matches = bf.match(src_desc, target_desc)
            src_pts.extend([src_kp[m.queryIdx].pt for m in matches])
            target_pts.extend([target_kp[m.trainIdx].pt for m in matches])
        affine_mat, _ = cv2.estimateAffine2D(np.array(src_pts), np.array(target_pts))
        target_positions = np.reshape(cv2.transform(np.reshape(src_positions, (-1, 1, 2)), affine_mat), (-1, 2))
        self.positions = target_positions
        self.find_nn_dist()
        return self.positions
    
    
