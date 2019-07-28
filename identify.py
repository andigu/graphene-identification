import numpy as np
from scipy.misc import derivative
from scipy.ndimage import distance_transform_edt, filters, sobel
from scipy.optimize import least_squares
from scipy.signal import find_peaks
from skimage.color import rgb2gray
from skimage.feature import canny
from skimage.filters import gaussian, hessian
from skimage.future import graph
from skimage.measure import find_contours, approximate_polygon, grid_points_in_poly
from skimage.measure import label as im_label
from skimage.morphology import disk, binary_dilation, remove_small_objects, binary_closing
from skimage.segmentation import watershed, relabel_sequential
from skimage.transform import rescale


def smooth_with_mask(image, mask, sigma=1):
    """
    Taken from scikit-image to apply a Gaussian to an image while ignoring certain regions
    """
    smooth = lambda d: gaussian(d, sigma=sigma)
    bleed_over = smooth(mask.astype(float))
    masked_image = np.zeros(image.shape, image.dtype)
    masked_image[mask] = image[mask]
    smoothed_image = smooth(masked_image)
    output_image = smoothed_image / (bleed_over + np.finfo(float).eps)
    return output_image


def get_cdf(d, step=1e-3, sigma=2):
    _cdf = np.vectorize(lambda x: np.sum(d < x) / d.size)
    rng = np.arange(d.min() - step, d.max() + step, step)
    return rng, filters.gaussian_filter1d(_cdf(rng), sigma=sigma)


def get_histogram(d, step=1e-3, sigma=2):
    """
    Finds histogram as indicated by the PDF of the data
    :param d: Data to find histogram
    :param step: Step size in histogram range
    :param sigma: Smoothing factor
    """
    rng, cdf = get_cdf(d, step=step, sigma=sigma)
    pdf = derivative(lambda x: np.interp(x, rng, cdf), rng, dx=1e-3)
    return rng, pdf / pdf.max()


def find_vignetting(im, groups):
    """

    :param im: Image M to find vignetting for
    :param groups: Labelled image of same shape as im, representing groups of similar color
    :return:
    """
    height, width = im.shape[:2]
    labels = set(groups[groups > 0])
    in_group = {i: groups == i for i in labels}
    aspect = width / height
    grid = np.meshgrid(np.linspace(-0.5, 0.5, height, endpoint=False), np.linspace(-0.5, 0.5, width, endpoint=False),
                       indexing='ij')

    def evaluate(params, u, v, subtract_max=False):
        (f, theta, phi, u0, v0, const, *alpha) = params
        original_shape = u.shape
        u, v = (u - u0).flatten(), (v - v0).flatten() * aspect
        R_x, R_y, R_z = np.cos(theta), np.sin(theta), 0
        W = np.array([
            [0, -R_z, R_y],
            [R_z, 0, -R_x],
            [-R_y, R_x, 0]
        ])
        T = np.identity(3) + np.sin(phi) * W + (1 - np.cos(phi)) * W.dot(W)
        normal = T[:, 2]
        C0 = np.array([0, 0, f])
        im_r = np.vstack([u, v, np.zeros(len(u))])  # 2D Position vector in G coordinates (constant height=0)
        cam_r = T.dot(im_r)  # 3D position vector in space coordinates
        C = C0[np.newaxis].T - cam_r
        radial_distance = np.linalg.norm(cam_r[:2, :], axis=0)  # Pixel distance from center of lens
        alpha = np.hstack((1, alpha))
        geometric = np.vander(radial_distance, N=p + 1, increasing=True).dot(alpha).clip(0, 1)
        # Normal dot C0 is equivalent to normal dot C and is significantly faster
        vignette = ((f * (normal.dot(C0)) * (C0.dot(C))) / (np.linalg.norm(C, axis=0) ** 4)) + const
        vignette = vignette * geometric
        vignette = vignette.reshape(original_shape)
        largest = np.max(vignette)
        for group in labels:
            group_mask = in_group[group]
            I0 = vignette[group_mask].dot(im[group_mask]) / (np.linalg.norm(vignette[group_mask]) ** 2)
            if subtract_max:
                vignette[group_mask] = (vignette[group_mask] - largest) * I0
            else:
                vignette[group_mask] = vignette[group_mask] * I0
        return vignette

    def residual(params):
        return (im - evaluate(params, *grid))[groups > 0]

    p = 2
    init_params = np.hstack(([1, 0, 0, 0, 0, 0], np.repeat(0, p)))
    optimize_result = least_squares(residual, init_params, method="lm", loss="linear")
    return evaluate(optimize_result["x"], *grid, subtract_max=True)


def make_poly(labelled, label, tol=2):
    contour = find_contours(labelled == label, 0)[0]
    return approximate_polygon(contour, tolerance=tol)


def fill_shape(labelled, label=1, tol=0):
    in_shape = grid_points_in_poly(labelled.shape[:2], make_poly(labelled, label, tol=tol))
    return np.where(in_shape, label, labelled)


def peaks_and_valleys(x, find_peaks_kwargs=None):
    if find_peaks_kwargs is None:
        find_peaks_kwargs = {}
    peaks, _ = find_peaks(x, **find_peaks_kwargs)
    valleys = []
    peaks = [0, *peaks, -1]
    for i in range(len(peaks) - 1):
        valleys.append(np.argmin(x[peaks[i]:peaks[i + 1]]) + peaks[i])
    return np.array(peaks[1:-1]), np.array(valleys)


def prune(labelled, rel_threshold=0.0, abs_threshold=0, default=0):
    """
    Prunes area of a labelled image by thresholding area
    :param labelled: Labelled image
    :param rel_threshold: Threshold for area of a labelled group relative to the largest label area
    :param abs_threshold: Absolute threshold for area of a labelled group
    :param default: Label to send areas below threshold to
    :return:
    """
    labels = np.array(list(set(labelled[labelled > 0])))
    label_freq = np.array([np.sum(labelled == l) for l in labels])
    valid_labels = labels[(label_freq >= abs_threshold) & (label_freq >= rel_threshold * label_freq.max())]
    kept = np.logical_or.reduce([labelled == label for label in valid_labels])
    offset = -default if default < 0 else 0
    labelled = np.where(kept, labelled, default) + offset
    return relabel_sequential(labelled)[0] - offset


def downscale_to(im, area_limit, multichannel=True):
    downscale_fac = 1 if np.prod(im.shape[:2]) < area_limit else np.sqrt(area_limit / np.prod(im.shape[:2]))
    return rescale(im, downscale_fac, multichannel=multichannel)


def get_groups(original):
    """
    Finds a segmentation of image by taking an oversegmentation produced by the Priority-Flood watershed and
    progressively reducing with a boundary region adjacency graph

    :param original: Original RGB image to segment
    :return: Segmented image. label = 0 represents an edge, label = -1 represents a pruned area
    """
    original = gaussian(original, sigma=1.5, multichannel=True)
    original = downscale_to(original, area_limit=2e5)
    g = original[:, :, 1]

    def weight_boundary(RAG, src, dst, n):
        default = {'weight': 0.0, 'count': 0}
        count_src = RAG[src].get(n, default)['count']
        count_dst = RAG[dst].get(n, default)['count']
        weight_src = RAG[src].get(n, default)['weight']
        weight_dst = RAG[dst].get(n, default)['weight']
        count = count_src + count_dst
        return {
            'count': count,
            'weight': (count_src * weight_src + count_dst * weight_dst) / count
        }

    greyscale = rgb2gray(original)
    gradient = np.hypot(sobel(greyscale, axis=0), sobel(greyscale, axis=1))
    segmentation1 = watershed(gradient, markers=400, mask=greyscale > 0.3)
    RAG = graph.rag_boundary(segmentation1, gradient)
    segmentation2 = graph.merge_hierarchical(segmentation1, RAG, thresh=5e-3, rag_copy=False,
                                             in_place_merge=True,
                                             merge_func=lambda *args: None,
                                             weight_func=weight_boundary)
    segmentation2[greyscale < 0.3] = -1
    segmentation2 = prune(segmentation2, abs_threshold=g.size / 1000, default=-1)
    counts, lo, hi = [], [], []
    for label in set(segmentation2[segmentation2 >= 0]):
        interior = distance_transform_edt(segmentation2 == label) >= 1.5
        if np.sum(interior) >= 0:
            counts.append(np.sum(interior))
            lo.append(np.percentile(gradient[interior], q=70))
            hi.append(np.percentile(gradient[interior], q=90))

    edges = canny(greyscale, low_threshold=np.average(lo, weights=counts),
                  high_threshold=np.average(hi, weights=counts))
    edges = binary_dilation(edges, disk(2))
    edges = binary_closing(edges, disk(5))
    edges = remove_small_objects(edges, g.size / 1000)
    edges = edges[1:-1, 1:-1]
    edges = np.pad(edges, pad_width=1, mode='constant', constant_values=1)
    groups = im_label(edges, background=1, connectivity=1)
    groups = prune(groups, abs_threshold=g.size / 1000)
    groups[greyscale < 0.15] = -2  # Ignore black areas due to mechanical vignetting
    return groups


def monolayers(original, logger):
    logger.info("Finding edges and groups")
    groups = get_groups(original)

    logger.info("Finding vignetting")
    original = downscale_to(original, area_limit=2e5)
    g = original[:, :, 1]
    vignetting = find_vignetting(g, groups)
    vignetting = np.where(groups <= 0, 0, vignetting)
    g = smooth_with_mask(g - vignetting, mask=groups > 0, sigma=2)

    logger.info("Finding substrate and monolayer colors")
    if np.sum(groups == -2) != 0:
        distance_from_mechanical_vignette = distance_transform_edt(groups != -2)
    else:
        distance_from_mechanical_vignette = np.full(groups.shape, np.inf)

    far_from_mechanical_vignette = distance_from_mechanical_vignette > 50
    rng, histogram = get_histogram(g[(groups > 0) & far_from_mechanical_vignette])
    peaks, valleys = peaks_and_valleys(histogram, {"distance": 10, "prominence": 1e-3})
    substrate_color = rng[peaks[np.argmax(histogram[peaks])]]
    best_monolayer_color = (-0.06 + 1) * substrate_color
    monolayer_peak_index = np.argmin(np.abs(rng[peaks] - best_monolayer_color))
    min_monolayer, max_monolayer = rng[valleys[monolayer_peak_index]], rng[valleys[monolayer_peak_index + 1]]
    min_monolayer = 2 * rng[peaks[monolayer_peak_index]] - max_monolayer
    monolayer = (g >= min_monolayer) & (g <= max_monolayer) & (groups > 0) & far_from_mechanical_vignette
    monolayer = remove_small_objects(monolayer, np.sum(groups > 0) // 200)
    monolayer = binary_closing(monolayer, disk(5))
    monolayer = binary_dilation(monolayer, disk(1))
    monolayer = fill_shape(monolayer)

    logger.info("Separating overlapping monolayers")
    distance = distance_transform_edt(monolayer)
    ridges = hessian(distance, black_ridges=True) * monolayer
    markers = im_label(ridges == 1)
    monolayer_group = watershed(-distance, markers, mask=monolayer)
    monolayer_group = prune(monolayer_group, rel_threshold=0.05)
    for group in range(1, np.max(monolayer_group) + 1):
        monolayer_group = fill_shape(monolayer_group, group, tol=2)
    return {"original": original, "monolayers": monolayer_group}
