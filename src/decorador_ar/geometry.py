from dataclasses import dataclass

import cv2 as cv
import numpy as np


@dataclass(frozen=True)
class Camera:
    matrix: np.ndarray
    distortion: np.ndarray
    size: tuple[int, int]

    def save(self, path, rms=None):
        np.savez(path, matrix=self.matrix, distortion=self.distortion,
                 size=self.size, rms=np.nan if rms is None else rms)

    @classmethod
    def load(cls, path):
        with np.load(path) as data:
            camera = cls(data["matrix"], data["distortion"], tuple(data["size"]))
        if (camera.matrix.shape != (3, 3) or not np.isfinite(camera.matrix).all()
                or min(camera.matrix[0, 0], camera.matrix[1, 1]) <= 0
                or not np.isfinite(camera.distortion).all()
                or len(camera.size) != 2 or min(camera.size) <= 0):
            raise ValueError("Calibración no válida.")
        return camera


@dataclass(frozen=True)
class Pose:
    rotation: np.ndarray
    translation: np.ndarray

    def camera_points(self, points):
        matrix, _ = cv.Rodrigues(self.rotation)
        return points @ matrix.T + self.translation.ravel()

    def project(self, points, camera):
        pixels, _ = cv.projectPoints(points, self.rotation, self.translation,
                                     camera.matrix, camera.distortion)
        return pixels.reshape(-1, 2)

    def blend(self, previous, alpha):
        old, _ = cv.Rodrigues(previous.rotation)
        new, _ = cv.Rodrigues(self.rotation)
        delta, _ = cv.Rodrigues(new @ old.T)
        step, _ = cv.Rodrigues(alpha * delta)
        rotation, _ = cv.Rodrigues(step @ old)
        translation = previous.translation + alpha * (self.translation - previous.translation)
        return Pose(rotation, translation)


def marker_points(side):
    s = side / 2
    return np.array([[-s, s, 0], [s, s, 0], [s, -s, 0], [-s, -s, 0]])


def reprojection_error(points, pixels, pose, camera):
    residual = pose.project(points, camera) - pixels
    return float(np.sqrt(np.mean(np.sum(residual ** 2, axis=1))))


def estimate_pose(points, pixels, camera):
    _, rotations, translations, _ = cv.solvePnPGeneric(
        points, pixels, camera.matrix, camera.distortion, flags=cv.SOLVEPNP_IPPE_SQUARE
    )
    candidates = []
    for rotation, translation in zip(rotations, translations):
        if not (np.isfinite(rotation).all() and np.isfinite(translation).all()):
            continue
        pose = Pose(rotation, translation)
        if np.all(pose.camera_points(points)[:, 2] > 0):
            candidates.append((pose, reprojection_error(points, pixels, pose, camera)))
    return min(candidates, key=lambda candidate: candidate[1], default=(None, None))
