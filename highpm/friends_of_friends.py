import numpy as np
from scipy import spatial


def find_friend(data, length, cores=None):
    '''
    Uses scipy's kDTree functionalities to find all friends 
    (points within a given distance of each other)
    '''
    if cores == None:
        return data.query_ball_tree(data, length)
    else:
        return data.query_ball_point(data.data, length, workers=cores)


def friends_of_friends(list_friends):
    '''
    Main function of the code. Using a mutating list of friends, 
    finds all sets that overlap with each other and joins them together. 
    New version, for loop changed for set operations, significantly faster
    '''
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