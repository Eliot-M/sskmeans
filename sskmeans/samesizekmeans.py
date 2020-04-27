# samesizekmeans.py

import sskmeans.support_functions as sf
import pandas as pd
import numpy as np
import numbers


class SameSizeKmeans:

    def __init__(self, n_clusters=4, max_iter=100, flexibility_size=0,
                 random_state=None, distance='euclidean'):

        # Define arguments from parameters
        self.k_clusters = n_clusters
        self.max_iter = max_iter
        self.flexibility_size = flexibility_size
        self.random_state = random_state
        self.distance = distance

        # Define in class arguments used/needed in further functions
        self._n_samples = 0
        self._n_features = 0
        self._k_min = 0
        self._k_max = 0
        self._k_centers = {}
        self._n_assign = {}

        # Check correct format of main parameters
        if (not(isinstance(self.k_clusters, int)) or
                not(isinstance(self.max_iter, int)) or
                not(isinstance(self.flexibility_size, numbers.Number))):
            raise TypeError("'n_cluster' and 'max_iter' must be positive intergers. "
                            "'flexibility_size' must be a positive number.")

        # Check number of cluster is strictly positive
        if self.k_clusters <= 0:
            raise ValueError('Number of iterations should be a positive number, got %d instead' % self.max_iter)

        # Check number of iterations is strictly positive
        if self.max_iter <= 0:
            raise ValueError('Number of iterations should be a positive number, got %d instead' % self.max_iter)

        # Check flexibility of clusters is positive or null
        if self.flexibility_size < 0:  # function check flexibility_size pour du 0-1 ou integer
            raise ValueError('Number of iterations should be a positive number '
                             '(float between 0 and 1 or a positive interger), got %d instead' % self.max_iter)

    def fit(self, X, y=None):
        """
        Compute a same size k-means clustering.
            Parameters
            ----------
            X : array-like or sparse matrix, shape=(n_samples, n_features)
                Training instances to cluster.
            y : Ignored
                Not used, present here for API consistency by convention.

            Returns
            -------
            self
                Fitted estimator.
        """

        # Get correct format for random_state
        self.random_state = sf.check_random_state(self.random_state)

        # Get correct format for input data (numpy array)
        X = sf.ensure_array_format(X)

        # Check format validity of input data
        sf.check_numeric_type(X)

        # Store dimensions
        self._n_samples, self._n_features = sf.get_shape(X)

        # Check there is more observations than clusters to make
        if self._n_samples < self.k_clusters:
            raise ValueError("n_samples=%d should be >= n_clusters=%d" % (self._n_samples, self.k_clusters))

        # Get size for clusters
        k_max, k_min, k_max_flex, k_min_flex = sf.cluster_size(self._n_samples, self.k_clusters, self.flexibility_size)

        # Get random centers to initialize the algorithms (sampled for observations)
        self._k_centers = sf.get_random_centers(array=X, n_clusters=self.k_clusters, random_state=self.random_state)

        # Create the id column of reference for observations
        X = sf.create_id_column_from_rows(X)

        # Compute the first assignment of observations to clusters
        self._n_assign = sf.cluster_init(X, self._k_centers, k_min)

        # Update the input data with cluster assignment by mapping the dictionary on ids
        X = sf.add_column_from_list(X, list(np.vectorize(self._n_assign.get)(X[:, -1])))

        # So far, initialization is over. The 2nd dimension of X has 2 new columns: observations ids and clusters ids.
        previous_dict_positive = {}

        for i in range(self.max_iter):
            # Get the current cluster's assignment of observations
            self._n_assign = sf.create_dict_from_columns(array=X, idx_key=-2, idx_value=-1)

            # Compute new centroids for clusters
            dict_new_cl_centers = sf.cal_cluster_center(array=X)

            # Compute distance between items and all clusters centroids
            dict_distances = sf.get_distance2centers(array=X, k_centers=dict_new_cl_centers, distance=self.distance)

            # After first iteration, save the previous dictionary with potential changes/moving/swapping
            if i > 0:
                previous_dict_positive = dict_positive_swap_deltas.copy()

            # Compute delta distance for all items between their current assignation and all other clusters
            deltas, dict_positive_swap_deltas = sf.delta_with_current_cluster(dict_dist_to_centers=dict_distances,
                                                                              dict_current_centers=self._n_assign)

            # Move and/or swap items if its lead to a better global assignment. Update the dictionary of assignments.
            self._n_assign = sf.optimize_items_assignment(dict_candidates_to_swap=dict_positive_swap_deltas,
                                                          current_clusters=self._n_assign, deltas=deltas,
                                                          n_min=k_min_flex, n_max=k_max_flex)

            # Update the input data with cluster assignment by mapping the freshly updated dictionary on ids
            X = sf.add_column_from_list(X[:, :-1], list(np.vectorize(self._n_assign.get)(X[:, -2])))
            # X[:,:-1] is the array without cluster assignment. X[:, -2] is the column with observations ids

            # Check if the loop can be broke. It can if there is no new candidate for moving or swapping
            if previous_dict_positive == dict_positive_swap_deltas:
                # If list of candidate is the same as in the previous iteration, break the loop
                break

        self._k_max = k_max_flex
        self._k_min = k_min_flex

        print("Same size kmeans fitting done")

    def transform(self, X, y=None, cluster_column="cluster_id"):
        """
        Apply results of the same size k-means clustering.

            Parameters
            ----------
            X : array-like or sparse matrix, shape=(n_samples, n_features)
                Training instances to cluster.
            y : Ignored
                Not used, present here for API consistency by convention.
            cluster_column: string
                Represents the name of the column containing the cluster assignment of observations.
                Default is 'cluster_id'

            Returns
            -------
            array-like or sparse matrix, shape=(n_samples, n_features with new columns)
        """

        # Apply the fitted clustering to input data
        if isinstance(X, pd.DataFrame):

            # Check column names are not 'index' or cluster_column
            if len(set(X.columns).intersection({'index', cluster_column})) != 0:
                raise KeyError("Error: Input data cannot have a column named 'index' nor %s" % cluster_column)

            X.reset_index(inplace=True)  # force the index order
            X.drop(['index'], axis=1, inplace=True)  # remove the initial index of the data
            columns_coordinates = list(X.columns)  # get columns used as input for the clustering

            X[cluster_column] = X.index.map(self._n_assign)  # map the clustering

            # Create a column 'index' from the index of the DataFrame (to keep consistency with array format)
            X.reset_index(inplace=True)

            # Force the order of columns (to keep consistency with array format)
            columns_order = columns_coordinates + ['index', 'cluster_id']
            X = X[columns_order]

            return X

        elif isinstance(X, np.ndarray):
            X = sf.create_id_column_from_rows(X)  # create the id column for observations

            # map the clustering and add it to the input array
            X = sf.add_column_from_list(X, list(np.vectorize(self._n_assign.get)(X[:, -1])))

            return X

    #
