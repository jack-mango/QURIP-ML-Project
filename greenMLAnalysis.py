"""
Author: Jack Mango
Contact: jackmango@berkeley.edu
Date: 2023-07-21
Description: Class for training machine learning models and determining tweezer occupancy from images.
"""

from datasetUtilities import ImageProcessor, Labeler
import matplotlib.pyplot as plt
from tensorflow.keras import models
import logging
import numpy as np
import os
from scipy.io import loadmat


class greenMLAnalysis():

    """
    A class used to train machine learning models that can be used to find the occupancy
    of tweezer sites in 556 nm (green) imaging. The class is designed to work with data from
    a continuous imaging experiment.

    Attributes:
        n_tweezers (int): The number of tweezers in the stack images.
        n_loops (int): Number of loops in the image stack.
        img_processor (ImageProcessor): an instance of the image processor class. A single instance is intended to be used to help with finding tweezer positions and creating image crops.
        model_path (string): The absolute path to a directory of a tensorflow neural network file, typically saved in the .h5 format.
        model (tensorflow.keras.Model): An instance of a tensorflow model loaded from the model path.
        separation (int, optional): The number of pixels each tweezer is separated by. Three times this number will be used to determine the side length of crops output by this function. If left as None the nearest neighbor distance will be used instead.
        make_plots (bool): If True, a plot of the tweezer positions will be automatically generated when finding the positions of tweezers.
    """

    DEFAULT = {
        'validation_split': 0.1,
        'epochs': 8
    }

    def __init__(self, n_tweezers, n_loops, model_path, r_atom=2.5, separation=None, tweezer_positions=None, make_plots=True):
        """
        Initialize the greenMLAnalysis with the provided parameters.

        Parameters:
            n_tweezers (int): The number of tweezers in the stack images.
            n_loops (int): Number of loops in the image stack.
            model_path (string): The absolute path to a directory of a tensorflow neural network file, typically saved in the .h5 format.
            r_atom (float, optional): approximate radius of atom in pixels.
            separation (int, optional): The number of pixels each tweezer is separated by. Three times this number will be used to determine the side length of crops output by this function.
            tweezer_positions (numpy.ndarray, optional): Coordinates of tweezers in the stack. Should have the shape [# of tweezers, 2]. Default is None.
            make_plots (bool, optional): If True, plots will be generated to verify positions and labeling.
        """
        self.n_tweezers = n_tweezers
        self.n_loops = n_loops
        self.img_processor = ImageProcessor.ImageProcessor(n_tweezers, n_loops,
                                                            r_atom=r_atom,  tweezer_positions=tweezer_positions, make_plots=make_plots)
        self.separation = separation
        self.make_plots = make_plots
        self.model_path = model_path
        self.model = models.load_model(model_path)

    def find_tweezers(self, stack):
        """
        Given an array of 2D images, find the locations of tweezer sites and the nearest neighbor distance using the
        image processor's find_tweezer_positions method.

        Parameters:
            stack (numpy.ndarray): A numpy array of 2D images. This function expects the array to have shape [# of images, image width, image height].

        Returns:
            numpy.ndarray: An array containing coordinates of each tweezer position. The shape of the array is [# of positions, 2].
        """
        positions = self.img_processor.find_tweezer_positions(stack[0])
        return positions

    def train_model(self, data_path, model_path=None, training_kwargs=DEFAULT):
        """
        Load data from a file or directory to train a model on, crop the images and make labels for them, then train the model 
        and save.

        Parameters:
            data_path (string): Either the absolute path to a directory containing .mat files or the path to a single .mat file to be used for training.
            model_path (string): The absolute path to a directory where the model will be saved to after training. If left as None the model will be saved to the same directory specified in self.model during initialization.
            training_kwargs (dict, optional): A dictionary containing parameters used when calling the model's .fit() method. Keys and values should match those that would be passed as kwargs to the .fit() method.

        Returns:
            tensorflow.keras.callbacks.History: Information about the metrics collected for the model during training. 
        """
        stack = self.load_data(data_path)
        self.find_tweezers(stack)
        labels = self.make_labels(stack)
        crops = self.make_crops(stack)
        crops, labels = self.filter_unlabeled(crops, labels)
        history = self.model.fit(crops, labels, **training_kwargs)
        if model_path is None:
            self.model.save(self.model_path)
        else:
            self.model.save(model_path)
        return history

    def get_occupancies(self, data_path):
        """
        Determine the occupancies of each tweezer in each image using a convolutional neural network.

        Parameters:
            data_path (string): Either the absolute path to a directory containing .mat files or the path to a single .mat file.

        Returns:
            positions (numpy.ndarray, optional): Coordinates of tweezers in the stack. Should have the shape [# of tweezers, 2]. Default is None.
            numpy.ndarray: positions of the tweezers sites.
            numpy.ndarray: occupancies of each tweezer site, contained in an array taking the shape [# of files, # of loops, # of images per loop, tweezer occupancies].
                           The ordering of the tweezers is the same as that of the positions.
        """
        stack = self.load_data(data_path)
        n_files = stack.shape[0]
        stack = np.array([np.concatenate(stack)])
        crops = self.make_crops(stack)
        occupancies = self.model.predict(crops)
        occupancies = np.argmax(occupancies, axis=1)
        return self.img_processor.positions, np.moveaxis(np.reshape(occupancies, (self.n_tweezers, n_files, self.n_loops, -1)), 0, -1)

    def fidelity_analysis(self, data_path):
        """
        Perform fidelity analysis on imaging data, returning the bright to dark and dark to bright probability for each tweezer. Also create
        a graph displaying this information.

        Parameters:
            data_path (string): Either the absolute path to a directory containing .mat files or the path to a single .mat file.

        Returns:
            prob_db (numpy.array): Dark to bright probability for each tweezer.
            prob_bd (numpy.array): Bright to dark probability for each tweezer.
            matplotlib.figure.Figure: Graph displaying bright to dark and dark to bright probability for each tweezer.
        """
        positions, occupancies = self.get_occupancies(data_path)
        first_diff = np.diff(occupancies, axis=-2)
        n_dark_to_bright = np.sum(first_diff == -1, axis=(0, 1, 2))
        n_dark = np.sum(occupancies[:, :, :-1] == 0, axis=(0, 1, 2))
        n_bright_to_dark = np.sum(first_diff == 1, axis=(0, 1, 2))
        n_bright = np.sum(occupancies[:, :, :-1] == 1, axis=(0, 1, 2))
        prob_db, prob_bd = n_dark_to_bright / n_dark, n_bright_to_dark / n_bright
        fig = plt.figure(figsize=(12.8, 4.8))
        plt.bar(np.arange(self.n_tweezers), prob_bd,
                label=f'Bright to Dark Probability', color='orange', alpha=0.5)
        plt.bar(np.arange(self.n_tweezers), prob_db,
                label=f'Dark to Bright Probability', color='steelblue', alpha=0.5)
        plt.axhline(prob_bd.mean(
        ), label=f"Bright to Dark Average={prob_bd.mean():.3}", color='darkorange', linestyle='--')
        plt.axhline(prob_db.mean(
        ), label=f"Dark to Bright Average={prob_db.mean():.3}", color='dodgerblue', linestyle='--')
        plt.xlabel('Tweezer Number')
        plt.ylabel('Probability')
        plt.legend(loc='upper left')
        plt.title('Fidelity')
        plt.show()
        return positions, prob_db, prob_bd, fig

    def load_data(self, path):
        """
        Load .mat files from self.data_path if that directory exists.

        Parameters:
            path (string): Either the absolute path to a directory containing .mat files or the path to a single .mat file.

        Returns:
            stack (numpy.ndarray): A numpy array of 2D images. The shape of the array should be [# of files, # of images per file, image width, image height].
        """
        if os.path.isdir(path):
            stack = []
            for file in os.listdir(path):
                if file.endswith('.mat'):
                    data = loadmat(path + '/' + file)
                    stack.append(data['stack'])
            stack = np.array(stack)
        else:
            stack = np.array([loadmat(path)['stack']])
        return stack

    def make_labels(self, stack):
        """
        Generate labels for each tweezer in each image to be used for training. The ordering of the labels corresponds to the ordering of the crops output by the make_crops() method. You can read more about the full algorithm in the documentation.

        Parameters:
            stack (numpy.ndarray): A numpy array of 2D images. The shape of the array should be [# of files, # of images per file, image width, image height].

        Returns:
            numpy.ndarray: A numpy array of labels with the shape [# of samples, 2]. Bright labels are encoded as [0, 1], dark as [1, 0] and unknowns as [NaN, NaN].
        """
        per_tweezer_file = stack.shape[1]
        n_files = stack.shape[0]
        labels = np.empty((self.n_tweezers * n_files * per_tweezer_file))
        for i, file_stack in enumerate(stack):
            crops_1x1 = self.img_processor.crop_tweezers(file_stack, 1)
            labeler = Labeler.Labeler(
                crops_1x1, self.n_tweezers, self.n_loops, make_plots=self.make_plots)
            bright_dark_fits, r_sq = labeler.bright_dark_fit()
            thresholds, plots = labeler.find_thresholds(bright_dark_fits)
            all_below_upper, all_above_lower = labeler.threshold_misfits()
            labels[i * self.n_tweezers * per_tweezer_file:
                   (i + 1) * self.n_tweezers * per_tweezer_file] = labeler.make_labels()
        return np.transpose(np.array([np.absolute(labels - 1), labels]))

    def make_crops(self, stack):
        """
        Crop each tweezer, including possible nearest neighbors.

        Parameters:
            stack (numpy.ndarray): A numpy array of 2D images. The shape of the array should be [# of files, # of images per file, image width, image height].
            separation (int, optional): The number of pixels each tweezer is separated by. Three times this number will be used to determine the side length of crops output by this function. If left as None the nearest neighbor distance will be used instead.

        Returns:
            numpy.ndarray: A numpy array of images corresponding to each tweezer site. The ordering matches that of the labels output by the make_labels() method. The shape should be [# of samples, image height, image_width].
        """
        if self.separation == None:
            separation = self.img_processor.nn_dist
        else:
            separation = self.separation
        crop_size = 2 * np.rint(1.5 * separation).astype(int) + 1
        per_tweezer_file = stack.shape[1]
        n_files = stack.shape[0]
        crops_3x3 = np.empty(
            (self.n_tweezers * n_files * per_tweezer_file, crop_size, crop_size))
        for i, file_stack in enumerate(stack):
            crops = self.img_processor.crop_tweezers(file_stack, 3, separation=separation)
            crops_3x3[i * self.n_tweezers * per_tweezer_file:
                      (i + 1) * self.n_tweezers * per_tweezer_file] = np.reshape(crops, (-1, crop_size, crop_size))
        return np.reshape(crops_3x3, (-1, *crops_3x3.shape[-2:]))

    def filter_unlabeled(self, crops, labels):
        """
        Remove all entries of crops and labels where the array at the corresponding index of labels contains a NaN

        Parameters:
            crops (numpy.ndarray): A numpy array of 2D images. The shape of the array should be [# of files, # of images per file, image width, image height].
            labels (numpy.ndarray): A numpy array of labels with the shape [# of samples, 2]. Bright labels are encoded as [0, 1], dark as [1, 0], and unknown as [NaN, NaN].
            
        Returns:
            numpy.ndarray: The crops array but with any entry that was labeled unknown removed.
            numpy.ndarray: The labels array but with any entry that was labeled unknown removed.
        """
        unlabeled_images = np.isnan(labels[:, 1])
        mask = ~ unlabeled_images
        return crops[mask], labels[mask]
