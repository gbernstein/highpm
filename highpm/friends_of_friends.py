"""
This module provides functions to find friends and groups of
friends-of-friends in a a set of points, typically used in the context of
spatial data analysis or social network analysis. The main functions include
`find_friend` for finding neighbors within a specified distance and
`friends_of_friends` for identifying connected groups based on direct and
indirect friendships.
"""


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

    todo = set(range(len(list_friends)))
    result = []
    while len(todo) > 0:
        i = todo.pop()
        new_set = set([i])
        fresh_friends = set(list_friends[i])
        fresh_friends.remove(i)
        while len(fresh_friends) > 0:
            next_friend = fresh_friends.pop()
            new_set.add(next_friend)
            if next_friend in todo:
                fof = set(list_friends[next_friend])
                todo.remove(next_friend)
                fof = fof - new_set
                fresh_friends = fresh_friends | fof
        result.append(list(new_set))

    return result
