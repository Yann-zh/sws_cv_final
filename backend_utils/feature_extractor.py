from backend_utils.utils import img_map
from skimage import feature
from skimage.feature import hog, local_binary_pattern as lbp, canny as c
from backend_utils.preprocessor import cvt_2_gray
import numpy as np


def vectorize(src_imgs):
    X_features = []
    for img in src_imgs:
        X_features.append(img.ravel())
    return X_features


def histogram_of_gradient(src_imgs):
    try:
        return vectorize(img_map(src_imgs, hog, orientations=8, pixels_per_cell=(10, 10),
                                 cells_per_block=(1, 1), visualize=False, multichannel=True))
    except ValueError:
        return vectorize(img_map(src_imgs, hog, orientations=8, pixels_per_cell=(10, 10),
                                 cells_per_block=(1, 1), visualize=False, multichannel=False))


def local_binary_pattern(src_imgs, P=8, R=1.0):
    """
    Source images must be converted to gray scale image first.
    """
    return vectorize(img_map(cvt_2_gray(src_imgs), lbp, P=P, R=R))


def canny(src_imgs, sigma=3):
    return vectorize(img_map(cvt_2_gray(src_imgs), c, sigma=sigma))