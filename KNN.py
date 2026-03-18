__authors__ = ['', '', '']
__group__ = ''

import numpy as np
import math
import operator


class KNN:

    def __init__(self, train_data, labels):

        self._init_train(train_data)
        self.labels = np.array(labels)

    #############################################################
    # THIS FUNCTION CAN BE MODIFIED FROM THIS POINT, if needed
    #############################################################

    def _init_train(self, train_data):
        """
        Initialize training data as PxD matrix
        """
        train_data = train_data.astype(float)
        num_samples = train_data.shape[0]

        self.train_data = train_data.reshape(num_samples, -1)

    #############################################################

    def get_k_neighbours(self, test_data, k):
        """
        Compute k nearest neighbours
        """

        test_data = test_data.astype(float)

        num_test = test_data.shape[0]

        test_data = test_data.reshape(num_test, -1)

        # Esto es equivalente a cdist(test_data, self.train_data) en scipy.spatial.distance
        test_sq = np.sum(test_data**2, axis=1).reshape(-1, 1)
        train_sq = np.sum(self.train_data**2, axis=1).reshape(1, -1)
        cross = np.dot(test_data, self.train_data.T)

        dist_matrix = np.sqrt(test_sq + train_sq - 2*cross)

        # Obtenemos los índices de los k vecinos más cercanos para cada punto de prueba
        nearest_idx = np.argsort(dist_matrix, axis=1)[:, :k]

        self.neighbors = self.labels[nearest_idx]

    #############################################################


    def get_class(self):
        """
        Get the class by maximum voting
        """

        classes_predict = []

        for neighbor_row in self.neighbors:

            votes = {}

            for label in neighbor_row:
                votes[label] = votes.get(label, 0) + 1

            predicted = max(votes.items(), key=lambda x: x[1])[0]

            classes_predict.append(predicted)

        return np.array(classes_predict)

    #############################################################

    def predict(self, test_data, k):
        """
        Predict classes for test data
        """

        self.get_k_neighbours(test_data, k)

        return self.get_class()
