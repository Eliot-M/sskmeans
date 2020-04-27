# support_functions.py

import pandas as pd
import math
import numpy as np
import numbers


# - Core
def get_shape(array):
    """
    Return the shape of a array-like matrix.

    Args:
        array: array-like or sparse matrix, shape=(n_samples, n_features)

    Returns:
        n_samples: int, number of rows
        n_features: int, number of columns

    """

    try:
        if len(array.shape) > 2:
            raise ValueError("Error: Array is of dimension %d but must be of dimension 2" % len(array.shape))

        return array.shape[0], array.shape[1]

    except AttributeError:
        print("Error: The object has no attribute 'shape'. Please use a pandas.DataFrame or numpy.ndarray object")

    except IndexError:
        print("Error: The object has only one dimension. "
              "Please use a pandas.DataFrame or numpy.ndarray object with 2 dimensions")


# - Core
def check_random_state(seed):
    """
    Turn seed into a np.random.RandomState instance

    Args:
        seed: None | int | instance of RandomState
            If seed is None, return the RandomState singleton used by np.random.
            If seed is an int, return a new RandomState instance seeded with seed.
            If seed is already a RandomState instance, return it.
            Otherwise raise ValueError.

    Returns:
        RandomState instance

    """

    if seed is None or seed is np.random:
        return np.random.mtrand._rand

    if isinstance(seed, numbers.Integral):
        return np.random.RandomState(seed)

    if isinstance(seed, np.random.RandomState):
        return seed

    raise ValueError('%r cannot be used to seed a numpy.random.RandomState instance' % seed)


# - Core
def check_numeric_type(array):
    """
    Check if an array contains only numeric values. Accepted formats: int64, int32, float64, float32

    Args:
        array: array-like, shape=(n_samples, n_features)

    Returns:
        boolean, True if the array is one of accepted types.
    """

    is_numeric = array.dtype == 'float64' or \
                 array.dtype == 'int64' or \
                 array.dtype == 'float32' or \
                 array.dtype == 'int32'

    if not is_numeric:
        raise TypeError("Error: Array is of type %s but expected a 'float' or 'int' format" % array.dtype)

    return is_numeric


# - Initialization part
def cluster_size(n_obs, n_cluster, size_flexibility=0):
    """
    Compute the number of items allowed in each cluster. Allow a flexibility to create almost same size clusters.

    Args:
        n_obs: int, n_samples value.
        n_cluster: int, The number of clusters to form as well as the number of centroids to generate.
        size_flexibility: int | float (default 0), flexibility allowed for the size of clusters.

            float numbers under 1 represent the percentage of variation allowed for cluster's size.
            Example: 0.2 for ±20% variation per cluster.

            integer numbers over 1 represent the number of observation variation allowed for cluster's size.
            Example: 5 for ±5 items per cluster.

            0 represents a fixed sized without any variation allowed.

    Returns:
        n_max: int, maximum size of a cluster. Needed in initialization part. (Unused atm)
        n_min: int, minimum size of a cluster. Needed in initialization part.
        n_min_flex: int, minimum size of a cluster (including the flexibility). Needed in iteration part.
        n_max_flex: int, maximum size of a cluster (including the flexibility). Needed in iteration part.

    """

    # Get the estimated size of clusters (total number of items / number of clusters)
    if (n_obs % n_cluster == 0) and (size_flexibility == 0):
        # Case of equal sized clusters
        n_min = n_min_flex = n_obs / n_cluster
        n_max = n_max_flex = n_obs / n_cluster

    else:
        # Case of unequal sized clusters
        n_min = math.floor(n_obs / n_cluster)
        n_max = math.ceil(n_obs / n_cluster)

        if size_flexibility >= 1:
            n_min_flex = max(n_min - int(abs(size_flexibility)), 1)
            n_max_flex = min(n_max + int(abs(size_flexibility)), n_obs - 1)
        else:
            n_min_flex = max(n_min - math.floor(abs(size_flexibility) * math.floor(n_obs / n_cluster)), 1)
            n_max_flex = min(n_max + math.floor(abs(size_flexibility) * math.floor(n_obs / n_cluster)), n_obs - 1)

    return int(n_max), int(n_min), int(n_max_flex), int(n_min_flex)


# - Initialization part
def get_random_centers(array, n_clusters, random_state=None):
    """
    Initialise random centers for k clusters based on item's features.

    Args:
        array: numpy ndarray, shape=(n_samples, n_features). Only features used to compute centroids.
        n_clusters: int, The number of clusters to form as well as the number of centroids to generate.
        random_state: RandomState instance, Determines random number generation for
            centroid initialization. Use an int to make the randomness deterministic.

    Returns:
        dict_cluster_center: dict, cluster's id (as key) and an array of features as the centroid of clusters (as value)

    """

    # Get k random centers - pandas version
    centers = array[random_state.choice(array.shape[0], replace=False, size=n_clusters), :]

    # Store centers in a dictionary, key is the id of the cluster
    dict_cluster_center = {k: centers[k] for k in range(n_clusters)}

    return dict_cluster_center


# - Core
def get_distance(x, y, distance='euclidean'):
    """
        Compute the distance between 2 elements (array-like).

    Args:
        x: array-like, input array.
        y: array-like, input array.
        distance: str (default is 'euclidean'), type of distance to compute. Also available: 'cosine'

    Returns:
        float, the distance between vectors x and y.

    To do:
        Add other measures
        Raise errors for non-numeric array

    """

    if distance == 'euclidean':
        return np.linalg.norm(x - y)

    elif distance == 'cosine':
        return np.dot(x, y)/(np.linalg.norm(x) * np.linalg.norm(y))

    else:
        # Add other distances later
        return np.linalg.norm(x - y)


# - Initialization part
def ensure_array_format(array):
    """
    Check if the type of the input a numpy ndarray.
    Pandas DataFrame format will be transformed to numpy ndarray format.

    Args:
        array: Any kind of object

    Returns:
        Raise an error if the input is not a pandas DataFrame or a numpy ndarray.
        Else it return the object as a ndarray object.

    """

    # For pandas Dataframes, return the numpy array
    if isinstance(array, pd.DataFrame):
        return array.values

    # For numpy arrays, keep it as is
    elif isinstance(array, np.ndarray):
        return array

    raise TypeError('input data must be a pandas DataFrame or a numpy array')


# - Core
def list2array(flat_list):
    """
    Transform a list into an array of single sub elements

    Args:
        flat_list: list or numpy array of  1 dimension

    Returns:
        numpy array of single sub elements

    Example : list2array([0, 1, 2]) returns array([[0],
                                                   [1],
                                                   [2]])

    """

    if isinstance(flat_list, list) or isinstance(flat_list, np.ndarray):
        return np.array([[elt] for elt in list(flat_list)])

    raise TypeError('flat_list argument is not a list')


# - Core
def add_column_from_list(array, flat_list):
    """
    Add a column at the end of a numpy ndarray with 2 dimensions.

    Args:
        array: numpy ndarray, ndarray with 2 dimensions
        flat_list: list or numpy array, numpy array of  1 dimension or flat list

    Returns:
        numpy ndarray, with values of the second argument as the last column

    """

    # Convert the list/array
    array_to_add = list2array(flat_list=flat_list)

    # Check dimension of the hosting array
    if len(array.shape) == 2:
        extended_array = np.append(array, array_to_add, axis=1)
        return extended_array

    # Raise error for any dimension issue
    else:
        raise ValueError("Array dimensions between 'array' and 'array_to_add' are not the same,"
                         "'array' is of dimension %d and 'array_to_add' is computed from "
                         "list2array(flat_list=flat_list) and is of dimension %d"
                         % (len(array.shape), len(array_to_add.shape)))


# - Initialization part
def create_id_column_from_rows(array):
    """
    Create a column in last position containing an id based on the row index.

    Args:
        array: numpy ndarray, shape=(n_samples, n_features)

    Returns:
        numpy ndarray, with an id column as the last column

    """

    # Create an id list
    idx = list(range(array.shape[0]))

    # Add it to the matrix
    array_and_idx = add_column_from_list(array, idx)

    return array_and_idx


# - Initialization part
def cluster_init(array, k_centers, k_min, distance='euclidean'):
    """

    Args:
        array: numpy ndarray, shape=(n_samples, n_features including an "id" column)
        k_centers: dict, cluster's id (as key) and an array of features as the centroid of clusters (as value)
        k_min: minimum size of a cluster
        distance: str (default 'euclidean'), distance to use to compute the distance between centroids and observations.

    Returns:
        dict, observations ids (as keys) and assignation clusters ids (as values)

    """

    # Based on the function "create_id_column_from_rows" the id column is at the end,
    # all previous columns are coordinates used for position
    coordinates_array = array[:, :-1]  # just remove the last column which is the id column, unnecessary

    dict_assigned_items2cluster = {}

    # Iter over all centroids
    for k_id, k_center in k_centers.items():

        # Lambda function easier to use in apply_along_axis
        distance2cluster = lambda x: get_distance(x, k_center, distance=distance)

        # Apply distance computation on each row
        distances = np.apply_along_axis(distance2cluster, axis=1, arr=array[:, :-1])  # coordinates_array

        # Get up to k_min closest items for each cluster
        rows_idx = np.argpartition(distances, min(k_min, len(distances) - 1))

        # Extract value in id column from the filtered array of closest items
        sample_idx = [x.item() for x in array[rows_idx[:k_min], -1:]]

        # Store item id and associated cluster (from main loop)
        for idx in sample_idx:
            dict_assigned_items2cluster[int(idx)] = int(k_id)

        # Remove already assigned items from array and coordinates_array
        array = np.delete(array, rows_idx[:k_min], axis=0)
        # coordinates_array = np.delete(coordinates_array, rows_idx[:k_min], axis=0)

    # For last items (if total number of items is not divisible by the number of clusters)
    if array.shape[0] != 0:
        # Find a cluster for last item. Temporary option: random dispatch
        for i in range(array.shape[0]):
            dict_assigned_items2cluster[int(array[i, -1:].item())] = int(list(k_centers.keys())[i])

    return dict_assigned_items2cluster


# - Iteration part
def cal_cluster_center(array, method='mean'):
    """
    Compute clusters centers from items positions assigned in it.

    Args:
        array: numpy ndarray, shape=(n_samples, n_features)
        method: str, method used to compute the center (default is 'mean').

    Returns:

    To do: Split function in 2 parts:
        1st one to compute the mean of a ndarray,
        2nd to adapt to the current issues (index -1 and -2)

    """
    dict_cluster_center = {}

    for i in np.unique(array[:, -1]):  # list of unique clusters (use self.k_clusters ?)
        # create sub table
        tmp = array[np.where(array[:, -1] == i)]
        # map means to cluster id
        dict_cluster_center[int(i)] = np.mean(tmp[:, :-2], axis=0)

    return dict_cluster_center


# - Iteration part
def get_distance2centers(array, k_centers, distance='euclidean'):
    """
    Compute the distance between all items and clusters. XXXXXXXX

    Args:
        array: numpy ndarray, shape=(n_samples, n_features including id and cluster)
        k_centers: dict, cluster's id (as key) and an array of features as the centroid of clusters (as value)
        distance: str (default 'euclidean'), distance to use to compute the distance between centroids and observations.

    Returns:
        dist_to_centers_dict: nested dict,
            First level: items (as key) and a dict of distance to clusters
            Second level: clusters id (as key) and the distance (between 1st level key and 2nd level key) (as value).

    To do:
        Consistency with args of the previous function 'compute_cluster_center'
        Check if another format could be better than a nested dict

    """

    # Initialize 1st level dictionary
    dist_to_centers_dict = {}

    # Iter over all item
    for row in array:
        index = int(row[-2])  # get the id
        # Iter (dict comprehension) over all clusters and compute each distance to centroids.
        # Create the 2nd level dictionary
        distance2k = {k_id: get_distance(row[:-2], k_center, distance) for k_id, k_center in k_centers.items()}

        # Complete the 1st level dictionary with item (as key) and the sub-dictionary (2nd level) previously computed.
        dist_to_centers_dict[index] = distance2k

    return dist_to_centers_dict


# - Core
def create_dict_from_columns(array, idx_key, idx_value):
    """
    Create a dictionary to assign item to their cluster_id based on a DataFrame

    Args:
        array: numpy array, shape=(n_samples, n_features).
        idx_key: int, index of the column used as a key in the dictionary.
        idx_value: int, index of the column used as a value in the dictionary.

    Returns:
        dict, item index (as key) and cluster assignation (as value)

    To do: check type idx_X -> must be integer
    """

    return dict(zip(array[:, idx_key].astype(int), array[:, idx_value].astype(int)))


# - Iteration part
def delta_with_current_cluster(dict_dist_to_centers, dict_current_centers):
    """
    Compute the difference between the distance to current assignation and the distance to other centers

    Args:
        dict_dist_to_centers: nested dict, items (as main-key) and a dict (as main-value) of
            clusters id (as sub-key) and the distance (between 1st level key and 2nd level key) (as sub-value).

        dict_current_centers: dict, item index (as key) and cluster assignation (as value)

    Returns:
        list_all_other_differences: list,
            contain all combination tuples (item, cluster, distance difference with current assignation)
        dict_candidates_to_switch: nested dict,
            First level: items (as key) and a dict of gain to a cluster
            Second level: clusters id (as key) and the distance gain (compare to current item assignation) (as value).

    To do:
        Find a less heavy way of thinking

    """
    dict_candidates_to_swap = {}
    all_other_differences = []

    for item, dict_distances_cluster in dict_dist_to_centers.items():  # key = item_id value: sub dict of possibilities
        current_cluster = dict_current_centers[item]  # get the current cluster of the item
        dict_gain_to_new_cluster = {}
        all_distances = []

        for cl_id, _ in dict_distances_cluster.items():  # value = cluster_id
            if cl_id != current_cluster:
                # For all other cluster, compute the difference between the distance of current assignation and an other
                distance_difference = dict_distances_cluster[current_cluster] - dict_distances_cluster[cl_id]

                # Keep track of all potential gain/loss to each cluster
                all_distances.append((cl_id, distance_difference))

                # Keep gain of distance as item to swap to improve overall clustering
                if distance_difference > 0:  # a positive difference means the item is closest to another centroid
                    dict_gain_to_new_cluster[cl_id] = distance_difference

        # Reshape sublist (cluster_id,  distance) by adding item in each tuple to get (item, cluster_id, distance)
        all_distances = [(item, x[0], x[1]) for x in all_distances]

        # Add it to overall list, will have to be flattened
        all_other_differences.append(all_distances)

        # From the item to swap (improving the overall clustering),
        # keep only non-empty ones into a new dictionary with item as kayΩ
        if bool(dict_gain_to_new_cluster):
            dict_candidates_to_swap[item] = dict_gain_to_new_cluster

    # Flatten the list of list containing all tuples (item, cluster_id, distance)
    list_all_other_differences = [item for sublist in all_other_differences for item in sublist]

    return list_all_other_differences, dict_candidates_to_swap  # dict(index: dict(cluster: gain))


# - Iteration part
def move_items(dict_candidates_to_swap, current_clusters, k_min, k_max):
    """
    Move items from one cluster to another if there is a benefit and if the size constraint allows it.

    Args:
        dict_candidates_to_swap: nested dict, item index (as key) and a dict (as value) of
            clusters id (as sub-key) and the distance gain (as sub-value).
        current_clusters:  dict, item index (as key) and cluster assignation (as value)
        k_min: int, minimum size of a cluster (including a flexibility)
        k_max: int, maximum size of a cluster (including a flexibility)

    Returns:
        dict_candidates_to_swap: nested dict, input dictionary imputed from moved without swap items
        current_assign: dict, input dictionary updated with new assignment with moved items
        candidates_key_remove: list, list of items moved to a new cluster

    """
    candidates_key_remove = []

    # Loop over items that could improve the clustering
    for item, swap_cluster in dict_candidates_to_swap.items():

        # Get the current size of clusters. Allocation of item will be updated in this loop and need to be re-computed
        dict_cl_count = {v: sum(x == v for x in current_clusters.values()) for v in current_clusters.values()}
        # Previously: Counter(list(current_clusters.values()))

        # Get the current cluster of the candidate item
        item_current_cluster = current_clusters[item]

        # Check if the candidate item can leave its cluster
        if dict_cl_count[item_current_cluster] > k_min:

            # Transform dictionary with potential new clusters to a sorted list of tuples
            tuple_cl_dist = list(swap_cluster.items())

            # This sorted list allows to find the best swap (first element)
            sorted_tuple_cl_dist = sorted(tuple_cl_dist, key=lambda x: x[1], reverse=True)

            # Iter over solution, and stop at the first solution
            for i in range(len(sorted_tuple_cl_dist)):

                # Check if the potential new cluster can receive the item (size constraint)
                if dict_cl_count[sorted_tuple_cl_dist[i][0]] < k_max:

                    # If it is possible, update the assignment to the new cluster
                    current_clusters[item] = sorted_tuple_cl_dist[i][0]

                    # If it is possible, update candidates list to be removed (no need to swap them further)
                    candidates_key_remove.append(item)

                    break  # Break the 2nd level loop, item has been moved

    # Update the list of potential swaps by removing already moved items (keys of the dict)
    for x in candidates_key_remove:
        dict_candidates_to_swap.pop(x, None)

    return dict_candidates_to_swap, current_clusters, candidates_key_remove


# - Iteration part
def swap_items(deltas, dict_candidates_to_swap, current_clusters, treated_items):
    """
    Swap items between 2 clusters if there is a global benefit. There is a benefit if the distance gain by the first
    item is bigger than the distance lost by the second. If there is a mutual gain, it's even better.

    Args:
        deltas: list, contain all combination tuples (item, cluster, distance difference with current assignation)
        dict_candidates_to_swap: nested dict, item index (as key) and a dict (as value) of
            clusters id (as sub-key) and the distance gain (as sub-value).
        current_clusters: dict, item index (as key) and cluster assignation (as value)
        treated_items: list, list of item already treated in the move function

    Returns:
        swapped_items: dict, item index (as key) and new cluster assignation (as value)

    To do:
        Optimize this mess

    """

    swapped_items = current_clusters

    # Loop over items that could improve the clustering
    for item, dict_swap_cluster in dict_candidates_to_swap.items():

        # Check if item hasn't been treated before
        if item not in treated_items:
            # Transform sub-dict to a sorted list of tuples
            tuple_cluster_dist = list(dict_swap_cluster.items())
            sorted_tuple_cl_dist = sorted(tuple_cluster_dist, key=lambda x: x[1], reverse=True)

            # Loop over possible cluster (for swap)
            for i in range(len(sorted_tuple_cl_dist)):

                # Get loop elements: Potential cluster & distance to it
                destination_cluster = sorted_tuple_cl_dist[i][0]
                dist_improve = sorted_tuple_cl_dist[i][1]

                # Get item current cluster
                item_current_cluster = current_clusters[item]

                # Get list of items of the destination cluster
                items_in_destination = [k for k, v in current_clusters.items() if v == destination_cluster]

                # Find potential items from list of deltas for not already treated items in destination cluster
                # having a distance to the current cluster of item to swap
                deltas_cluster = [x for x in deltas if
                                  x[0] in items_in_destination and x[1] == item_current_cluster and
                                  x[0] not in treated_items]

                # Sort according to distance to find the best candidate for the swap
                deltas_cluster = sorted(deltas_cluster, key=lambda x: x[2], reverse=True)

                # Get distance for the swapped candidate
                d = deltas_cluster[0][2]

                # If there is a benefit pure or if cost on one side is bellow benefit on the other side
                if ((d < 0) and (abs(d) < dist_improve)) or (d > 0):  # distance of 1st elt (which is min)
                    swapped_items[item] = destination_cluster  # update assignation of the item
                    swapped_items[deltas_cluster[0][0]] = item_current_cluster  # update assignation of the swapped

                    # Add item to the list of item already treated
                    treated_items.append(item)
                    treated_items.append(deltas_cluster[0][0])

                    break  # break the loop, item already swapped

    # At the end of the loop, all untouched item have to remain in their cluster
    keys_remaining = [x for x in list(current_clusters.keys()) if x not in treated_items]  # get untouched items

    for x in keys_remaining:
        swapped_items[x] = current_clusters[x]

    return swapped_items


def optimize_items_assignment(dict_candidates_to_swap, current_clusters, deltas, n_min, n_max):
    """
    Function running 'move_items' function and then 'swap_items' function to get a better global assignation

    Args:
        dict_candidates_to_swap: nested dict, item index (as key) and a dict (as value) of
            clusters id (as sub-key) and the distance gain (as sub-value).
        current_clusters: dict, item index (as key) and cluster assignation (as value)
        deltas: list, contain all combination tuples (item, cluster, distance difference with current assignation)
        n_min: int, minimum size of a cluster (including the flexibility)
        n_max: int, maximum size of a cluster (including the flexibility)

    Returns:
        new_clusters: dict, item index (as key) and new cluster assignation (as value)

    """

    # Step one: move candidates (if possible) to improve global clustering
    dict_candidates_to_swap, current_clusters, treated_items = move_items(dict_candidates_to_swap,
                                                                          current_clusters, n_min, n_max)

    # Step two: swap candidates to improve global clustering
    new_clusters = swap_items(deltas, dict_candidates_to_swap, current_clusters, treated_items)

    return new_clusters

#
