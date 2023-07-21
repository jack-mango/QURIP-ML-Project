"""
Author: Jack Mango
Contact: jackmango@berkeley.edu
Date: 2023-07-21
Description: Class to create labels for images based on thresholding.
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from datasetUtilities.AutoGauss import GaussianMixture
from datasetUtilities.AnalysisWarning import AnalysisWarning
import warnings


import logging
log = logging.getLogger(__name__)

class Labeler():

    """
    Contains the methods used in the labeling tweezers as bright or dark.

    Attributes:
        n_tweezers (int): The number of tweezers in the stack images.
        n_loops (int): The number of loops in the image stack.
        per_loop (int): The number of images in each loop of the stack.
        img_vals (numpy.ndarray): A single value that is attributed to each crop to be labeled.
        make_plots (bool): If set to True, plots will be generated of the fits when make_labels is called.
        thresholds (numpy.ndarray): An array of upper and lower thresholds used for labeling. Should have shape [# of tweezers, 2]. The first entry is the lower threshold, the second the upper threshold.
        fits (numpy.ndarray): An array containing Gaussian fits for each tweezer's brightness data.
        labels (numpy.ndarray): An array of labels denoting whether a tweezer is bright, dark, or unknown. Should have labels for the same tweezer and loop next to each other.
        info (dict): The relevant information gathered when executing this pipeline stage as a dictionary.
    """

    def __init__(self, crops, n_tweezers, n_loops, make_plots=True):
        self.n_tweezers = n_tweezers
        self.n_loops = n_loops
        self.per_loop = crops.shape[1] // n_loops
        self.img_vals = self.find_img_vals(crops)
        self.make_plots = make_plots
        self.thresholds = None
        self.fits = None
        self.labels = None
    
    def bright_dark_fit(self):
        """
        Fit a bimodal Gaussian distribution to each self.img_vals for each individual tweezer. 

        Returns:
            numpy.ndarray: A self.n_tweezers x 2 x 3 array, where the ith entry in the first axis corresponds to the ith tweezer,
                           the first (second) entry on the second axis corresponds to Gaussian fit with the lower (higher) mean,
                           and the final axis contains the mean, standard deviation, and relative amplitude in that order.
            numpy.ndarray: An array of R^2 values for the fits contained in a self.n_tweezers long array. 
        """
        fits = np.empty((self.n_tweezers, 2, 3))
        r_sq = np.empty(self.n_tweezers)
        for i in range(self.n_tweezers):
            model = GaussianMixture(self.img_vals[i])
            fits[i], r_sq[i] = model.fit()
        self.fits = fits
        return self.fits, r_sq
    
    def find_img_vals(self, crops, r_atom=2.5):
        """
        Assign a brightness score to each image in crops. In the current implementation, this is done
        by taking the average pixel value across each image in crops.

        Parameters:
            crops (numpy.ndarray): A four-dimensional array of images where the first axis corresponds to tweezer number, 
                                   the second corresponds to image number, and the final two correspond to the row and column
                                   numbers of the pixels within the image.

        Returns:
            numpy.ndarray: Values assigned to each crop according to the function method used by this method.
        """
        return np.mean(crops, axis=(2, 3))
    
    def find_thresholds(self, fits, z=4.753424308822899):
        """
        Assuming img_vals follows a bimodal Gaussian distribution, calculates the image values that are
        z standard distributions above (below) the lower (upper) threshold. The particular value of z chosen
        corresponds to the number of standard deviations that 1 - CDF(x) = 1e-6 for a Gaussian distribution along
        whichever direction corresponds to the lower or upper thresholding.

        Parameters:
            fits (numpy.ndarray): Parameters for a Gaussian fit. Must be a numpy array with shape (n_tweezers, 2, 3), with the first axis
                                  corresponding to tweezer number, the second corresponding to which Gaussian mode (lower mean first, higher 
                                  mean second), and the third containing the parameters corresponding to that Gaussian mode, ordered
                                  [mean, standard deviation, relative amplitude].
            z (float): The number of standard deviations to be used in calculating the thresholds. 

        Returns:
            numpy.ndarray: An array of thresholds for each tweezer with shape (n_tweezers, 2).
            matplotlib.figure.Figure or None: A plot for each tweezer of its bimodal Gaussian fit, its image value histogram, and its thresholds.
                                              Returns None if self.make_plots is False.
        """
        thresholds = np.empty((self.n_tweezers, 2))
        if self.make_plots:
            fig, axs = plt.subplots(self.n_tweezers // 5 + (self.n_tweezers % 5 > 0), 5, figsize=(8.5, 11 * self.n_tweezers / 50))
            fig.tight_layout(h_pad=0.8)
            log.info("Making threshold plots...")
        for i, fit in enumerate(fits):
            lower_thresh = fit[1, 0] - fit[1, 1] * z
            upper_thresh = fit[0, 0] + fit[0, 1] * z
            thresholds[i] = np.array([lower_thresh, upper_thresh])
            if self.make_plots:
                counts, bins, _ = axs[i // 5][i % 5].hist(self.img_vals[i], bins=(self.per_loop // 4), density=True)
                x_vals = np.linspace(bins[0], bins[-1], self.per_loop)
                axs[i // 5][i % 5].plot(x_vals, GaussianMixture.func(x_vals, *fit[0], *fit[1]), 'k')
                axs[i // 5][i % 5].axvline(lower_thresh, color='r', linestyle='--')
                axs[i // 5][i % 5].axvline(upper_thresh, color='r', linestyle='--')
                axs[i // 5][i % 5].set_title(f"Tweezer {i}", fontsize=8)
                axs[i // 5][i % 5].tick_params(axis='both', labelsize=8) 
        self.thresholds = thresholds
        if self.make_plots:
            return self.thresholds, fig
        else:
            return self.thresholds, None
    
    def make_labels(self):
        """
        Create a label for each crop, corresponding to an image value. If the crop's image value falls below the tweezer's lower
        threshold or in a segment of image values that fall below the lower threshold and don't rise above the upper threshold, then 
        it is labeled as dark. Similarly, if a crop's image value is above the tweezer's upper threshold or in a segment of
        image values above the upper theshold and don't fall below the lower threshold, then it is labeled as bright. If an image
        value is between the two thresholds and isn't in a segment of values all above or below one of the thresholds, it is labeled
        as unknown. Dark crops are labeled by 0, bright crops by 1, and unknown crops by NaNs. 

        Returns:
            numpy.ndarray: A one-dimensional array of labels for each image value. 
        """
        labels = np.empty((self.n_tweezers, self.n_loops, self.per_loop))
        for i, tweezer_vals in enumerate(self.img_vals):
            for loop_num in range(self.n_loops):
                loop = tweezer_vals[loop_num * self.per_loop: (loop_num + 1) * self.per_loop]
                labels[i, loop_num] = self.slicer(loop, *self.thresholds[i])
        self.labels = labels.ravel()
        return self.labels
    
    def slicer(self, arr, lower_thresh, upper_thresh):
        """
        Finds where values of arr that are above (below) upper_thresh (lower_thresh). Indices of arr that lie between two
        indices that are both above or below the same threshold are labeled as bright (1) or dark (0). Indices of arr that
        are between the two thresholds are labeled as unknown (NaN).

        Parameters:
            arr (numpy.ndarray): A one-dimensional array of image values.
            lower_thresh (float): A single number to be used as the lower threshold.
            upper_thresh (float): A single number to be used as the upper threshold.
    
        Returns:
            numpy.ndarray: A one-dimensional array of labels for each image value. 
        """
        labels = np.empty(arr.size)
        head = tail = 0
        bright = True
        for i, val in enumerate(arr):
            if val >= upper_thresh and bright:
                head = i + 1
            elif val >= upper_thresh and not bright:
                labels[head:i] = np.full(i - head, np.NaN)
                labels[tail:head] = np.zeros(head - tail)
                tail = i
                head = i
                bright = True
            elif val <= lower_thresh and not bright:
                head = i + 1
            elif val <= lower_thresh and bright:
                labels[head:i] = np.full(i - head, np.NaN)
                labels[tail:head] = np.ones(head - tail)
                head = i
                tail = i
                bright = False
        if bright:
            labels[tail:head] = np.ones(head - tail)
            labels[head:] = np.full(labels.size - head, np.NaN)
        else:
            labels[tail:] = np.zeros(labels.size - tail)
        return labels
    
    def threshold_misfits(self):
        """
        Note if any tweezer's thresholds are so high (low) such that all that tweezer's image values lie below (above) that
        threshold, leading to a lack of bright (dark) images for that tweezer.
    
        Returns:
            list: A list containing the indices for which all image values are below the upper threshold.
            list: A list containing the indices for which all image values are above the lower threshold.
        """
        all_below_upper = []
        all_above_lower = []
        for i, thresh in enumerate(self.thresholds):
            if np.all(thresh[0] < self.img_vals):
                all_above_lower.append(i)
                warnings.warn(f"Tweezer {i} threshold too low! No dark images!", AnalysisWarning)
            if np.all(thresh[1] > self.img_vals):
                all_below_upper.append(i)
                warnings.warn(f"Tweezer {i} threshold too high! No bright images!", AnalysisWarning)
        return all_below_upper, all_above_lower
    
    def threshold_plot(self, tweezer_num, show_unknowns=False):
        """
        Create a plot for an individual tweezer displaying its brightness values color-coded according to their label. Also plotted
        are thresholds and loop number dividers. 

        Parameters:
            tweezer_num (int): An integer indicating which tweezer's image values to plot.
            show_unknowns (bool): If True, the unknown image values will also be plotted.

        Returns:
            matplotlib.figure.Figure: A matplotlib figure displaying the color-coded image values, thresholds, and loop dividers.  
        """

        tweezer_vals = self.img_vals[tweezer_num]
        tweezer_labels = self.labels[tweezer_num * self.per_loop * self.n_loops:(tweezer_num + 1)* self.per_loop * self.n_loops]

        bright_mask = tweezer_labels == 1
        dark_mask = tweezer_labels == 0
        unknown_mask = np.isnan(tweezer_labels)

        bright_indices = np.where(bright_mask)[0]
        bright_vals = tweezer_vals[bright_mask]

        dark_indices = np.where(dark_mask)[0]
        dark_vals = tweezer_vals[dark_mask]

        unknown_indices = np.where(unknown_mask)[0]
        unknown_vals = tweezer_vals[unknown_mask]

        fig = plt.figure(figsize=(20, 10))
        plt.plot(bright_indices, bright_vals, '.', label='bright')
        plt.plot(dark_indices, dark_vals, '.', label='dark')
        if show_unknowns:
            plt.plot(unknown_indices, unknown_vals, 'o', label='?')
        plt.axhline(self.thresholds[tweezer_num, 1], color='r', linestyle='--', label=f"Upper Threshold = {self.thresholds[tweezer_num, 1]:.3f}")
        plt.axhline(self.thresholds[tweezer_num, 0], color='g', linestyle='--', label=f"Lower Threshold = {self.thresholds[tweezer_num, 0]:.3f}")
        plt.legend(loc='upper right', fontsize=16)
        plt.title(f"Tweezer Number = {tweezer_num}", fontsize=20)
        plt.xlabel("Image Number", fontsize=16)
        plt.ylabel("Average Pixel Value", fontsize=16)
        plt.tick_params(axis='both', labelsize=14)
        for i in range(self.n_loops):
            plt.axvline(i * self.per_loop, color='k', linestyle='--', label="Loop Separation")
        return fig 
    
def gaussian_kernel(k_size, std):
    """
    Unused
    """
    kernel = cv2.getGaussianKernel(k_size, std)
    return np.matmul(kernel, kernel.T)