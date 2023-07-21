"""
Author: Jack Mango
Contact: jackmango@berkeley.edu
Date: 2023-07-20
Description: This class contains utilities for automatically fitting Gaussian distributions to data and plotting them.
"""

import numpy as np
from scipy.optimize import curve_fit
from sklearn.mixture import GaussianMixture as GMM

import logging
log = logging.getLogger(__name__)

class AutoGauss_v1():

    """ DEPRECATED """

    GAUSSIAN_BOUNDS = (
        [-np.inf, 0, -np.inf],
        [np.inf, np.inf, np.inf]
    )

    def __init__(self, x_data, y_data):
        self.x_data = x_data
        self.y_data = y_data

    def fwhm(self, x, y):
        """ Estimate the full-width a half maximum of the data. """
        half_max = x.max() / 2
        over_half_max = np.where(y < half_max)
        return x[over_half_max[0][1]] - x[over_half_max[0][0]]
    
    def fit(self):
        """ Try to fit two Gaussians to the data given in data. To do so first we try to fit the Gaussian to 
        each peak separately, then try to do a fit of both together. If the data is found to only contain
        one Gaussian, then the corresponding fit parameters are set to np.inf. """
        first_guess, second_guess = self.double_gaussians_guess()
        if second_guess[0] == np.inf:
            return first_guess, second_guess
        else:
            params, cov = curve_fit(double_gaussian, self.x_data, self.y_data,
                                     p0=[*first_guess, *second_guess])
            return params[:3], params[3:]
                   
    def double_gaussians_guess(self):
        """ Give a reasonable estimate for what the possible fit parameters could be for both Gaussians.
        Returns two arrays where the first is a guess for the taller Gaussian and the second the shorter.
        You can read the documentation for a detailed overview of how the algorithm works. """
        first_peak = self.y_data.argmax()
        intersection = self.n_consecutive_increases(self.y_data[first_peak:], 3)
        if not intersection:
            intersection = self.n_consecutive_increases(self.y_data[first_peak::-1], 3)
            if intersection:
                intersection *= -1
        if intersection:
            split = np.absolute(first_peak + intersection)
            x_first, x_second = self.x_data[:split], self.x_data[split:]
            y_first, y_second = self.y_data[:split], self.y_data[split:]
            first_std_guess = self.fwhm(x_first, y_first) / (2 * np.sqrt(2 * np.log(2)))
            second_std_guess = self.fwhm(x_second, y_second) / (2 * np.sqrt(2 * np.log(2)))
            first_params, first_cov = curve_fit(gaussian, x_first, y_first,
                                               p0=[x_first[y_first.argmax()], first_std_guess, y_first.mean()],
                                               bounds=self.GAUSSIAN_BOUNDS)
            second_params, second_cov = curve_fit(gaussian, x_second, y_second,
                                               p0=[x_second[y_second.argmax()], second_std_guess, y_second.mean()],
                                               bounds=self.GAUSSIAN_BOUNDS)
        else:
            first_params = np.full(3, np.inf)
            second_params = np.full(3, -np.inf)
        return first_params, second_params
    
    def find_local_min(self, array):
        """ Given an array, return the index of a local minimum. Does so by splitting array at maximal index,
        then searching for a minimum on either side using n_consecutive_increases. """

    def n_consecutive_increases(self, array, n):
        """ Return the index of the first occurence of n consecutive increases if one exists. If none exist
        then this method returns None."""
        for i in range(array.size - n):
            if all(array[i: i + n] == np.sort(array[i:i + n])) and np.size(np.unique(array[i: i + n])) == n:
                return i
        return
    
class GaussianMixture():

    def __init__(self, data):
        self.data = data

    def fit(self):
        model = GMM(2)
        model.fit(np.reshape(self.data, (-1, 1)))
        means = model.means_.flatten()
        stds = np.sqrt(model.covariances_.flatten())
        amplitudes = model.weights_.flatten() / (stds * np.sqrt(2 * np.pi))
        if means[0] < means[1]:
            lower = [means[0], stds[0], amplitudes[0]]
            upper = [means[1], stds[1], amplitudes[1]]
        else:
            upper = [means[0], stds[0], amplitudes[0]]
            lower = [means[1], stds[1], amplitudes[1]]
        r_sq = self.r_squared(lower + upper)
        return np.array([lower, upper]), r_sq

    def func(x, mean_1, std_1, a_1, mean_2, std_2, a_2):
        return a_1 * np.exp(- (x - mean_1) ** 2 / (2 * std_1 ** 2)) + a_2* np.exp(- (x - mean_2) ** 2 / (2 * std_2 ** 2))

    def r_squared(self, params):
        vals, bins = np.histogram(self.data, bins=int(1 + self.data.max() - self.data.min()), density=True)
        centers = (bins[1:] + bins[:-1]) / 2
        estimated_vals = GaussianMixture.func(centers, *params)
        sst = np.sum((vals - vals.mean()) ** 2)
        sse = np.sum((vals - estimated_vals) ** 2)
        r_sq = 1 - (sse / sst)
        return r_sq
        
class MultivariateGaussian():

    """
    Used to fit two dimensional Gaussian distribution to image data. Pixel coordinates are generated at
    the center of each square pixel for the corresponding value in that pixel.
    """

    def __init__(self, data):
        self.data = data

    def make_coordinates(self):
        height, width = self.data.shape
        x_coords, y_coords = np.meshgrid(np.arange(width), np.arange(height))
        return x_coords.flatten(), y_coords.flatten()
    
    def fit(self):
        x, y = self.make_coordinates()
        width, height = self.data.shape

        bounds = (
            [0, 0, 0, 0, 0, -np.pi/4, -np.inf],   # Lower bounds
            [np.inf, width, height, np.inf, np.inf, np.pi/4, np.inf]  # Upper bounds
        )

        guess = (
        np.max(self.data),
        *np.unravel_index(self.data.argmax(), self.data.shape),
        width / 10,
        height / 10,
        0,
        np.min(self.data)
        )
        
        params, cov = curve_fit(self.func, (x, y), self.data.flatten(), p0=guess, bounds=bounds)
        return params[1:-1]
    
    def func(self, xy, amplitude, xo, yo, sigma_x, sigma_y, angle, offset):
        x, y = xy
        a = np.cos(angle)**2 / (2 * sigma_x**2) + np.sin(angle)**2 / (2 * sigma_y**2)
        b = -np.sin(2 * angle) / (4 * sigma_x**2) + np.sin(2 * angle) / (4 * sigma_y**2)
        c = np.sin(angle)**2 / (2 * sigma_x**2) + np.cos(angle)**2 / (2 * sigma_y**2)
        return amplitude * np.exp(- (a * (x - xo)**2) - (2 * b * (x - xo) * (y - yo)) - (c * (y - yo)**2)) + offset

class Gaussian2D():
    def __init__(self, data):
        self.data = data

    def make_coordinates(self):
        height, width = self.data.shape
        x_coords, y_coords = np.meshgrid(np.arange(width), np.arange(height))
        return x_coords.flatten(), y_coords.flatten()
    
    def fit(self):
        x, y = self.make_coordinates()
        width, height = self.data.shape

        bounds = (
            [0, 0, 0, 0, 0, -np.inf],   # Lower bounds
            [np.inf, width, height, np.inf, np.inf, np.inf]  # Upper bounds
        )

        guess = (
        np.max(self.data),
        *np.unravel_index(self.data.argmax(), self.data.shape),
        width / 10,
        height / 10,
        np.min(self.data)
        )
        
        params, cov = curve_fit(self.func, (x, y), self.data.flatten(), p0=guess, bounds=bounds)
        return params[1:-1]
    
    def func(self, xy, amplitude, xo, yo, sigma_x, sigma_y, offset):
        x, y = xy
        return amplitude * np.exp(
            -((x - xo)**2 / (2 * sigma_x**2) + (y - yo)**2 / (2 * sigma_y**2))
        ) + offset

def gaussian(x, mean, std, a):
    return a * np.exp(- (x - mean) ** 2 / (2 * std ** 2))

def double_gaussian(x, mean_1, std_1, a_1, mean_2, std_2, a_2):
    return gaussian(x, mean_1, std_1, a_1) + gaussian(x, mean_2, std_2, a_2)
