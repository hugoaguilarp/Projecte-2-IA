__authors__ = ['', '', '']
__group__ = ''

import numpy as np
import utils


class KMeans:

    def __init__(self, X, K=1, options=None):
        """
        Constructor of KMeans class
        Args:
            K (int): Number of clusters
            options (dict): dictionary with options
        """
        self.num_iter = 0
        self.K = K
        self._init_X(X)
        self._init_options(options)

    #############################################################
    ##  THIS FUNCTION CAN BE MODIFIED FROM THIS POINT, if needed
    #############################################################

    def _init_X(self, X):
        """Initialization of all pixels, sets X as an array of data in vector form (PxD)
            Args:
                X (list or np.array): list(matrix) of all pixel values
                    if matrix has more than 2 dimensions, the dimensionality of the sample space is the length of
                    the last dimension
        """

        X = np.asarray(X, dtype=float)

        if X.ndim > 2:
            X = X.reshape(-1, X.shape[-1])

        self.X = X

    def _init_options(self, options=None):
        """
        Initialization of options in case some fields are left undefined
        Args:
            options (dict): dictionary with options
        """

        default_options = {
            'km_init': 'first',
            'verbose': False,
            'tolerance': 0,
            'max_iter': np.inf,
            'fitting': 'WCD'
        }

        if options is None:
            options = {}

        default_options.update(options)
        self.options = default_options

    #############################################################

    def _init_centroids(self):
        """
        Initialization of centroids
        """

        K = self.K

        if self.options['km_init'].lower() == 'first':

            unique_points = []
            for point in self.X:
                if not any(np.array_equal(point, c) for c in unique_points):
                    unique_points.append(point)

                if len(unique_points) == K:
                    break

            self.centroids = np.array(unique_points)

        elif self.options['km_init'].lower() == 'random':

            random_idx = np.random.choice(self.X.shape[0], K, replace=False)
            self.centroids = self.X[random_idx]

        elif self.options['km_init'].lower() == 'custom':

            min_vals = np.min(self.X, axis=0)
            max_vals = np.max(self.X, axis=0)

            self.centroids = np.linspace(min_vals, max_vals, K)

        self.old_centroids = np.zeros_like(self.centroids)

    #############################################################

    def get_labels(self):
        """
        Calculates the closest centroid of all points in X and assigns each point to the closest centroid
        """

        dist_matrix = distance(self.X, self.centroids)
        self.labels = np.argmin(dist_matrix, axis=1)

    #############################################################

    def get_centroids(self):
        """
        Calculates coordinates of centroids based on the coordinates of all the points assigned to the centroid
        """

        self.old_centroids = self.centroids.copy()

        for cluster_id in range(self.K):

            cluster_points = self.X[self.labels == cluster_id]

            if cluster_points.size > 0:
                self.centroids[cluster_id] = np.mean(cluster_points, axis=0)

    #############################################################

    def converges(self):
        """
        Checks if there is a difference between current and old centroids
        """

        centroid_shift = np.linalg.norm(
            self.centroids - self.old_centroids,
            axis=1
        )

        return np.all(centroid_shift <= self.options['tolerance'])

    #############################################################

    def fit(self):
        """
        Runs K-Means algorithm until it converges or until the number of iterations is smaller
        than the maximum number of iterations.
        """

        self._init_centroids()
        self.num_iter = 0

        while self.num_iter < self.options['max_iter']:

            self.get_labels()
            self.get_centroids()

            self.num_iter += 1

            if self.converges():
                break

    #############################################################

    def withinClassDistance(self):
        """
         returns the within class distance of the current clustering
        """

        N = self.X.shape[0]

        if N == 0:
            return 0

        total_distance = 0

        for cluster_id in range(self.K):

            cluster_points = self.X[self.labels == cluster_id]

            if cluster_points.size > 0:
                dist = np.sum((cluster_points - self.centroids[cluster_id])**2, axis=1)
                total_distance += np.sum(dist)

        return total_distance / N

    #############################################################

    def betweenClassDistance(self):
        """
        Compute Between Class Distance (BCD)
        """

        if self.K <= 1:
            return 0

        total_dist = 0
        pairs = 0

        for i in range(self.K):
            for j in range(i + 1, self.K):

                dist = np.sum((self.centroids[i] - self.centroids[j])**2)

                total_dist += dist
                pairs += 1

        return total_dist / pairs

    #############################################################

    def find_bestK(self, max_K, option):
        """
         sets the best k analysing the results up to 'max_K' clusters
        """

        max_K = max(2, max_K)

        original_K = self.K

        scores = []

        for k in range(1, max_K + 1):

            self.K = k
            self.fit()

            wcd = self.withinClassDistance()
            bcd = self.betweenClassDistance()

            if option == "w":
                scores.append(wcd)

            elif option == "b":
                scores.append(bcd)

            elif option == "f":
                scores.append(bcd / wcd if wcd > 0 else 0)

        if option == "w":

            threshold = 20
            best_k = max_K

            for k in range(1, len(scores)):

                decrease = 100 * (scores[k] / scores[k-1])

                if (100 - decrease) < threshold:
                    best_k = k + 1
                    break

        else:

            best_k = np.argmax(scores) + 1

        self.K = best_k
        self.fit()

        self.K = original_K


#############################################################

def distance(X, C):
    """
    Calculates the distance between each pixel and each centroid
    Args:
        X (numpy array): PxD 1st set of data points (usually data points)
        C (numpy array): KxD 2nd set of data points (usually cluster centroids points)

    Returns:
        dist: PxK numpy array position ij is the distance between the
        i-th point of the first set an the j-th point of the second set
    """

    X_sq = np.sum(X**2, axis=1).reshape(-1, 1)
    C_sq = np.sum(C**2, axis=1).reshape(1, -1)

    XC = X @ C.T

    dist = np.sqrt(X_sq + C_sq - 2 * XC)

    return dist


#############################################################


def get_colors(centroids):
    """
    for each row of the numpy matrix 'centroids' returns the color label following the 11 basic colors as a LIST
    Args:
        centroids (numpy array): KxD 1st set of data points (usually centroid points)

    Returns:
        labels: list of K labels corresponding to one of the 11 basic colors
    """

    color_probs = utils.get_color_prob(centroids)

    indices = np.argmax(color_probs, axis=1)

    return [utils.colors[i] for i in indices]