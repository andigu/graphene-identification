import numpy as np
from sklearn.decomposition import PCA
from skimage.measure import perimeter


def farthest_coords(labelled, label):
    pca = PCA(n_components=2)
    coords = np.argwhere(labelled == label)
    com = np.mean(coords)
    coords = coords - com
    pca.fit(coords)
    projected = np.round(coords.dot(pca.components_.T)).astype(int)
    major_range, minor_range = set(projected[:, 0]), set(projected[:, 1])
    best_across = 0
    min_coords = None, None
    for x in major_range:
        matching = projected[projected[:, 0] == x]
        peak, valley = np.max(matching, axis=0), np.min(matching, axis=0)
        if peak[1] - valley[1] > best_across:
            min_coords = valley, peak
            best_across = peak[1] - valley[1]
    best_along = 0
    max_coords = None, None
    for y in minor_range:
        matching = projected[projected[:, 1] == y]
        peak, valley = np.max(matching, axis=0), np.min(matching, axis=0)
        if peak[0] - valley[0] > best_along:
            max_coords = valley, peak
            best_along = peak[0] - valley[0]

    m = np.vstack([*min_coords, *max_coords])
    ret = m.dot(np.linalg.inv(pca.components_.T))
    return np.round(ret + com).astype(int)


def axial_ratio(labelled, label):
    p0, p1, p2, p3 = farthest_coords(labelled, label)
    axis1 = np.linalg.norm(p0 - p1)
    axis2 = np.linalg.norm(p2 - p3)
    return axis1 / axis2


def compactness(labelled, label):
    return 4 * np.pi * np.sum(labelled == label) / (perimeter(labelled == label) ** 2)
