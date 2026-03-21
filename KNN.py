__authors__ = ['Hugo Aguilar', 'Aida Peix', '']
__group__ = '73'

import numpy as np
import math
import operator
import scipy.spatial.distance as distance
from scipy.spatial.distance import cdist

class KNN:

    def __init__(self, train_data, labels):

        self._init_train(train_data)
        self.labels = np.array(labels)

    #############################################################
    # THIS FUNCTION CAN BE MODIFIED FROM THIS POINT, if needed
    #############################################################

    def _init_train(self, train_data):
        """
        initializes the train data
        :param train_data: PxMxNx3 matrix corresponding to P color images
        :return: assigns the train set to the matrix self.train_data shaped as PxD (P points in a D dimensional space)
        """
        train_data = train_data.astype(float)
        num_samples = train_data.shape[0]

        self.train_data = train_data.reshape(num_samples, -1)

    #############################################################

    def get_k_neighbours(self, test_data, k):
        """
        given a test_data matrix calculates de k nearest neighbours at each point (row) of test_data on self.neighbors
        :param test_data: array that has to be shaped to a NxD matrix (N points in a D dimensional space)
        :param k: the number of neighbors to look at
        :return: the matrix self.neighbors is created (NxK)
                 the ij-th entry is the j-th nearest train point to the i-th test point
        """

        test_data = test_data.astype(float)

        num_test = test_data.shape[0]

        test_data = test_data.reshape(num_test, -1)

        # Esto es equivalente a cdist(test_data, self.train_data) en scipy.spatial.distance
        # test_sq = np.sum(test_data**2, axis=1).reshape(-1, 1)
        # train_sq = np.sum(self.train_data**2, axis=1).reshape(1, -1)
        # cross = np.dot(test_data, self.train_data.T)

        dist_matrix = cdist(test_data, self.train_data, 'euclidean')

        # Obtenemos los índices de los k vecinos más cercanos para cada punto de prueba
        nearest_idx = np.argsort(dist_matrix, axis=1)[:, :k]

        self.neighbors = self.labels[nearest_idx]

    #############################################################


    def get_class(self):
        """
        Get the class by maximum voting
        :return: 1 array of Nx1 elements. For each of the rows in self.neighbors gets the most voted value
                (i.e. the class at which that row belongs)
        """

        classes_predict = []

        for neighbor_row in self.neighbors:

            votes = {}

            for label in neighbor_row:
                votes[label] = votes.get(label, 0) + 1

            predicted = max(votes, key=votes.get)

            classes_predict.append(predicted)

        return np.array(classes_predict)

    #############################################################

    def predict(self, test_data, k):
        """
        predicts the class at which each element in test_data belongs to
        :param test_data: array that has to be shaped to a NxD matrix (N points in a D dimensional space)
        :param k: the number of neighbors to look at
        :return: the output form get_class a Nx1 vector with the predicted shape for each test image
        """

        self.get_k_neighbours(test_data, k)

        return self.get_class()
