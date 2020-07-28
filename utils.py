import cv2
import matplotlib.pyplot as plt
import copy
import numpy as np

from pytorch_openpose import model
from pytorch_openpose import util
from pytorch_openpose.body import Body

def get_keypoints(subset, candidate):
    """
    Returns keypoints for the first person in subset

    Parameters
    ----------
    subset : np.ndarray
        indices of candidates in each subset, total score, and number of parts

    candidate : np.ndarray
        all of the candidates

    Returns
    -------
    keypoints : np.ndarray
        coordinates of keypoints (-1 if it was not found)
    """
    parts = np.asarray(subset[0][:-2], dtype=np.int)
    keypoints = candidate[:, :2][parts]
    keypoints[parts == -1] = -1 * np.ones(2)

    return keypoints

