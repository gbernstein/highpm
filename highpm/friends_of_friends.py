"""
This module provides functions to find friends and groups of
friends-of-friends in a a set of points, typically used in the context of
spatial data analysis or social network analysis. The main functions include
`find_friend` for finding neighbors within a specified distance and
`friends_of_friends` for identifying connected groups based on direct and
indirect friendships.
"""

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components


def find_friend(data, length, cores=None):
    """Finds friends within a specified distance using either a BallTree or
    parallel query.

    Parameters
    ----------
    data : BallTree or similar
        The data structure containing the points to search for neighbors. Must
        support `query_ball_tree` and `query_ball_point` methods.
    length : float
        The maximum distance between points to be considered friends.
    cores : int or None, optional
        The number of worker threads to use for parallel computation. If None,
        a single-threaded BallTree query is used.
    Returns
    -------
    list of lists of int
        Indices of neighbors for each point within the specified distance.
    Notes
    -----
    If `cores` is None, uses `query_ball_tree` for neighbor search. Otherwise,
    uses `query_ball_point` with the specified number of workers.
    """

    if cores is None:
        return data.query_ball_tree(data, length)
    else:
        return data.query_ball_point(data.data, length, workers=cores)


def friends_of_friends(list_friends):
    """Finds groups of friends-of-friends in a friendship network. Given a list
    where each element contains the indices of direct friends for each person,
    this function identifies all connected groups (including indirect
    connections via friends-of-friends).

    Parameters
    ----------
    list_friends : list of list of int
        A list where the element at index `i` is a list of indices representing
        the direct friends of person `i`. Each person is identified by their
        index in the list.
    Returns
    -------
    result : list of list of int
        A list of groups, where each group is a list of indices representing
        people who are all connected directly or indirectly through
        friendships.
    Examples
    --------
    >>> friends = [[1, 2], [0, 2], [0, 1, 3], [2]]
    >>> friends_of_friends(friends)
    [[0, 1, 2, 3]]
    """

    n = len(list_friends)
    rows = np.repeat(np.arange(n), [len(f) for f in list_friends])
    cols = np.concatenate(list_friends).astype(int) if n else np.empty(0, int)
    graph = coo_matrix((np.ones(cols.size), (rows, cols)), shape=(n, n))

    _, labels = connected_components(graph, directed=False)
    order = np.argsort(labels, kind="stable")
    groups = np.split(order, np.flatnonzero(np.diff(labels[order])) + 1)
    return [g.tolist() for g in groups]


def query_pairs_groups(tree, length):
    """Friends-of-friends groups from a KDTree, without materializing
    find_friend's list-of-neighbor-arrays intermediate.

    Equivalent to friends_of_friends(find_friend(tree, length, cores=...))
    but built directly from tree.query_pairs(), which returns unique i<j
    pairs as one flat array instead of one Python list per point -- avoids
    materializing/boxing hundreds of millions of neighbor-index entries for
    dense catalogs (~19M detections on Sculptor-core healpixels), which was
    observed to exhaust memory before reaching any per-group fitting.

    Parameters
    ----------
    tree : scipy.spatial.KDTree
        Tree built over the points to group (e.g. via highpm.utils.arborist).
    length : float
        Maximum distance between points to be considered friends, same units
        as the tree's coordinates.

    Returns
    -------
    list of list of int
        Same format and semantics as friends_of_friends()'s return value.
    """

    n = tree.n
    pairs = tree.query_pairs(length, output_type="ndarray")
    if pairs.shape[0] == 0:
        return [[i] for i in range(n)]

    rows = pairs[:, 0]
    cols = pairs[:, 1]
    graph = coo_matrix((np.ones(rows.size), (rows, cols)), shape=(n, n))

    _, labels = connected_components(graph, directed=False)
    order = np.argsort(labels, kind="stable")
    groups = np.split(order, np.flatnonzero(np.diff(labels[order])) + 1)
    return [g.tolist() for g in groups]


if __name__ == "__main__":
    assert friends_of_friends([[1, 2], [0, 2], [0, 1, 3], [2]]) == [[0, 1, 2, 3]]
    assert sorted(friends_of_friends([[0, 1], [0, 1], [2], [3, 4], [3, 4]])) == [
        [0, 1],
        [2],
        [3, 4],
    ]

    import scipy.spatial as spspace

    pts = np.array(
        [[0.0, 0.0], [0.01, 0.0], [1.0, 1.0], [1.01, 1.0], [1.0, 1.01], [10.0, 10.0]]
    )
    tree = spspace.KDTree(pts)
    length = 0.1
    expected = friends_of_friends(find_friend(tree, length, cores=None))
    got = query_pairs_groups(tree, length)
    assert {frozenset(g) for g in got} == {frozenset(g) for g in expected}

    far_pts = np.array([[0.0, 0.0], [100.0, 0.0], [0.0, 100.0]])
    far_tree = spspace.KDTree(far_pts)
    isolated = query_pairs_groups(far_tree, 0.1)
    assert sorted(isolated) == [[0], [1], [2]]

    print("ok")
