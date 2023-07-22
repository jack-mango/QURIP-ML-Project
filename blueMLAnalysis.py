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

class blueMLAnalysis():
    """
    A class used to train machine learning models that can be used to find the occupancy
    of tweezer sites in 399 nm (blue) imaging. The class is designed to work with data from green imaging as well to create a training dataset.

    Attributes:
        n_tweezers (int): The number of tweezers in the stack images.
        img_processor (ImageProcessor): an instance of the image processor class. A single instance is intended to be used to help with finding tweezer positions and creating image crops.
        model_path (string): The absolute path to a directory of a tensorflow neural network file, typically saved in the .h5 format.
        model (tensorflow.keras.Model): An instance of a tensorflow model loaded from the model path.
        separation (int, optional): The number of pixels each tweezer is separated by. Three times this number will be used to determine the side length of crops output by this function. If left as None the nearest neighbor distance will be used instead.
        make_plots (bool): If True, a plot of the tweezer positions will be automatically generated when finding the positions of tweezers.
    """

    DEFAULT = {
        'validation_split': 0.2,
        'epochs': 16
    }

    def __init__(self, n_tweezers, model_path, r_atom=2.5, separation=None, tweezer_positions=None, make_plots=True):
        """
        Initialize the greenMLAnalysis with the provided parameters.

        Parameters:
            n_tweezers (int): The number of tweezers in the stack images.
            model_path (string): The absolute path to a directory of a tensorflow neural network file, typically saved in the .h5 format.
            r_atom (float, optional): approximate radius of atom in pixels.
            separation (int, optional): The number of pixels each tweezer is separated by. Three times this number will be used to determine the side length of crops output by this function. If left as None the nearest neighbor distance will be used instead.
            tweezer_positions (numpy.ndarray, optional): Coordinates of tweezers in the stack. Should have the shape [# of tweezers, 2]. Default is None.
            make_plots (bool, optional): If True, plots will be generated to verify positions.
        """
        self.n_tweezers = n_tweezers
        self.img_processor = ImageProcessor.ImageProcessor(n_tweezers, 1,
                                                             r_atom=r_atom, tweezer_positions=tweezer_positions, make_plots=make_plots)
        self.separation = separation
        self.make_plots = make_plots
        self.model_path = model_path
        self.model = models.load_model(model_path)

    def find_tweezers(self, pv_stack, nuvu_stack):
        """
        Using the positions of the tweezers in the pv_stack, calculate the corresponding positions of the tweezers in the nuvu_stack.

        Parameters:
            pv_stack (numpy.ndarray): A numpy array of 2D images corresponding to 556 nm imaging data. Expects the array to have shape [# of files, # of images per file, image width, image height].
            nuvu_stack (numpy.ndarray): A numpy array of 2D images corresponding to 399 nm imaging data. Expects the array to have shape [# of files, # of images per file, image width, image height].

        Returns:
            numpy.ndarray: An array containing coordinates of each tweezer position from the nuvu_stack. The shape of the array is [# of positions, 2].
        """
        nuvu_positions = self.img_processor.transform_positions(pv_stack, nuvu_stack)
        self.img_processor.find_nn_dist()
        self.img_processor.plot(nuvu_stack)
        return nuvu_positions

    def train_model(self, data_path, green_model_path, blue_model_path=None, green_separation=None, training_kwargs=DEFAULT):
        """
        Load data from a file or directory to train a model on, crop the images and make labels for them, then train the model 
        and save.

        Parameters:
            data_path (string): Either the absolute path to a directory containing .mat files or the path to a single .mat file to be used for training.
            green_model_path (string): The absolute path to a pretrained green neural network which will be used to label matching images from the 556 nm dataset.
            blue_model_path (string): The absolute path to a directory where the model will be saved to after training. If left as None the model will be saved to the same directory specified in self.model during initialization.
            green_separation (int, optional): The number of pixels separating each tweezer in green images. If left as None then the nearest neighbor distance will be used.
            training_kwargs (dict, optional): A dictionary containing parameters used when calling the model's .fit() method. Keys and values should match those that would be passed as kwargs to the .fit() method.

        Returns:
            tensorflow.keras.callbacks.History: Information about the metrics collected for the model during training. 
        """
        pv_stack, nuvu_stack = self.load_training_data(data_path)
        green_model = models.load_model(green_model_path)
        self.find_tweezers(pv_stack, nuvu_stack)
        crops = self.make_crops(nuvu_stack, separation=self.separation)
        labels = self.make_labels(pv_stack, green_model, green_separation=green_separation)
        history = self.model.fit(crops, labels, **training_kwargs)
        if blue_model_path is None:
            self.model.save(self.model_path)
        else:
            self.model.save(blue_model_path)
        return history
    
    def get_occupancies(self, data_path):
        """
        Determine the occupancies of each tweezer in each image using a convolutional neural network.

        Parameters:
            data_path (string): Either the absolute path to a directory containing .mat files or the path to a single .mat file.

        Returns:
            numpy.ndarray: positions of the tweezers sites.
            numpy.ndarray: occupancies of each tweezer site, contained in an array taking the shape [# of images, tweezer occupancies].
                           The ordering of the tweezers is the same as that of the positions.
        """
        stack = self.load_data(data_path)
        crops = self.make_crops(stack, separation=self.separation)
        occupancies = self.model.predict(crops)
        occupancies = np.argmax(occupancies, axis=1)
        return self.img_processor.positions, np.swapaxes(np.reshape(occupancies, (self.n_tweezers, -1)))

    def load_training_data(self, path):
        """
        Loads data from path. Path should be a directory containing a folder called 'pvcam' and one called 'nuvu' containing .mat files of image data for each camera. 
        Only files with matching fIdx are loaded from the two folders. If the number of images in each stack doesn't match, it is assumed that there is an integer multiple of
        images corresponding to each loop. For example if there are three images in the pvcam stack for every two in the nuvu stack, then it is assumed there are 
        three images in each pvcam loop and two images in each nuvu loop. The first image of each loop is kept in the stack as it is assumed these images correspond to each other.

        Parameters:
            path (string): Either the absolute path to a directory containing .mat files or the path to a single .mat file.

        Returns:
            pv_stack (numpy.ndarray): A numpy array of 2D images from the pvcam folder. The shape of the array should be [# images, image width, image height].
            nuvu_stack (numpy.ndarray): A numpy array of 2D images from the nuvu folder. The shape of the array should be [# images, image width, image height].
            """
        pv_stack = []
        nuvu_stack = []
        for file in os.listdir(os.path.join(path, 'nuvu')):
            if file.endswith('.mat'):
                name = file.split(',', 1)
                if os.path.exists(os.path.join(path, 'pvcam', f"camera,{name[1]}")):
                    nuvu_stack.append(loadmat(os.path.join(path, 'nuvu', file))['stack'])
                    pv_stack.append(loadmat(os.path.join(path, 'pvcam', f"camera,{name[1]}"))['stack'])
        nuvu_stack = np.concatenate(nuvu_stack, axis=0)
        pv_stack = np.concatenate(pv_stack, axis=0)
        gcd = np.gcd(pv_stack.shape[0], nuvu_stack.shape[0])
        pv_per_nuvu = pv_stack.shape[0] // gcd
        nuvu_per_pv = nuvu_stack.shape[0] // gcd
        return pv_stack[::pv_per_nuvu], nuvu_stack[::nuvu_per_pv]

    def make_labels(self, pv_stack, green_model, green_separation=None):
        """
        Generate labels for each tweezer in each image to be used for training. The ordering of the labels corresponds to the ordering of the crops output by the make_crops() method.
        Labels are generated from the green_model's classifications of the pv_stack.

        Parameters:
            pv_stack (numpy.ndarray): A numpy array of 2D images from the pvcam folder. The shape of the array should be [# of images, image width, image height].
            green_model (tensorflow.keras.Model): An instance of a trained neural network to be used for labeling.
            green_separation (int, optional): The number of pixels separating each tweezer in green images. If left as None then the nearest neighbor distance will be used.

        Returns:
            numpy.ndarray: A numpy array of labels with the shape [# of samples, 2]. Bright labels are encoded as [0, 1] and dark as [1, 0].
        """
        self.img_processor.find_tweezer_positions(pv_stack)
        pv_crops = self.make_crops(pv_stack, separation=green_separation)
        labels = green_model.predict(pv_crops)
        labels = np.argmax(labels, axis=1)
        return np.transpose(np.array([np.absolute(labels - 1), labels]))
    
    def make_crops(self, stack, separation=None):
        """
        Crop each tweezer, including possible nearest neighbors.

        Parameters:
            stack (numpy.ndarray): A numpy array of 2D images. The shape of the array should be [# of files, # of images per file, image width, image height].
            separation (int, optional): The number of pixels each tweezer is separated by. Three times this number will be used to determine the side length of crops output by this function.

        Returns:
            numpy.ndarray: A numpy array of images corresponding to each tweezer site. The ordering matches that of the labels output by the make_labels() method. The shape should be [# of samples, image height, image_width].
        """
        crops_3x3 = self.img_processor.crop_tweezers(stack, 3, separation=separation)
        return np.reshape(crops_3x3, (-1, *crops_3x3.shape[-2:]))