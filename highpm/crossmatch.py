import numpy as np
from scipy.spatial import KDTree


def mutual_nearest_neighbor(pts1, pts2, tolerance):
    """Symmetric best match: pts1[i] <-> pts2[idx[i]] counts only if each is the
    other's nearest neighbor (mutual NN), which is automatically one-to-one.

    Returns (matched_mask, idx_in_pts2, dist).
    """
    dist, idx = KDTree(pts2).query(pts1)
    dist_back, idx_back = KDTree(pts1).query(pts2)

    mutual = idx_back[idx] == np.arange(len(pts1))
    matched = mutual & (dist < tolerance)

    return matched, idx, dist


if __name__ == "__main__":
    pts1 = np.array([[0, 0], [0.3, 0], [10, 10]])
    pts2 = np.array([[0, 0], [5, 5]])
    matched, idx, _ = mutual_nearest_neighbor(pts1, pts2, tolerance=1.0)
    assert matched.tolist() == [True, False, False]
    assert idx[0] == 0
    print("self-test ok")
