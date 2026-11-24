import cv2 as cv
import numpy as np


def dictionary():
    return cv.aruco.getPredefinedDictionary(cv.aruco.DICT_4X4_50)


def marker_image(marker_id, pixels=400):
    marker = cv.aruco.generateImageMarker(dictionary(), marker_id, pixels)
    margin = pixels // 8
    return cv.copyMakeBorder(marker, margin, margin, margin, margin,
                             cv.BORDER_CONSTANT, value=255)


class MarkerDetector:
    def __init__(self, marker_id):
        parameters = cv.aruco.DetectorParameters()
        parameters.cornerRefinementMethod = cv.aruco.CORNER_REFINE_SUBPIX
        self.detector = cv.aruco.ArucoDetector(dictionary(), parameters)
        self.marker_id = marker_id

    def detect(self, frame):
        corners, ids, _ = self.detector.detectMarkers(frame)
        if ids is None:
            return None
        matches = np.flatnonzero(ids.ravel() == self.marker_id)
        if len(matches) != 1:
            return None
        return corners[matches[0]].reshape(4, 2).astype(np.float64)
