"""Read short clips from TUM RGB-D without extracting the full archive."""
from bisect import bisect_left
from dataclasses import dataclass
from pathlib import PurePosixPath
import tarfile

import cv2 as cv
import numpy as np

from .geometry import Camera, Pose
from .image_io import decode_image

FR1_ROOM_URL = ("https://webshare.cvg.cit.tum.de/g/rgbd/dataset/freiburg1/"
                "rgbd_dataset_freiburg1_room.tgz")

# ROS defaults recommended by TUM for the registered RGB-D images.
CAMERA = Camera(np.array([[525., 0, 319.5], [0, 525., 239.5], [0, 0, 1]]),
                np.zeros(5), (640, 480))


@dataclass(frozen=True)
class Frame:
    timestamp: float
    image: str
    depth: str | None
    pose: Pose | None


def _rows(data):
    return [line.split() for line in data.decode("utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")]


def _nearest(times, stamp):
    index = bisect_left(times, stamp)
    candidates = range(max(0, index - 1), min(len(times), index + 1))
    return min(candidates, key=lambda i: abs(times[i] - stamp), default=None)


def _pose(values):
    translation = np.asarray(values[:3], dtype=float)
    x, y, z, w = np.asarray(values[3:], dtype=float)
    scale = x*x + y*y + z*z + w*w
    s = 2 / scale
    camera_to_world = np.array([
        [1-s*(y*y+z*z), s*(x*y-z*w), s*(x*z+y*w)],
        [s*(x*y+z*w), 1-s*(x*x+z*z), s*(y*z-x*w)],
        [s*(x*z-y*w), s*(y*z+x*w), 1-s*(x*x+y*y)],
    ])
    world_to_camera = camera_to_world.T
    return Pose(cv.Rodrigues(world_to_camera)[0], (-world_to_camera @ translation)[:, None])


def read_index(archive):
    names = {"rgb.txt", "depth.txt", "groundtruth.txt"}
    tables = {}
    with tarfile.open(archive, "r|gz") as source:
        for member in source:
            path = PurePosixPath(member.name)
            if member.isfile() and path.name in names:
                tables[path.name] = _rows(source.extractfile(member).read())
                if tables.keys() == names:
                    break
    if tables.keys() != names:
        raise ValueError("El archivo TUM debe incluir rgb.txt, depth.txt y groundtruth.txt.")

    depth = sorted((float(t), path) for t, path in tables["depth.txt"])
    poses = sorted((float(row[0]), row[1:]) for row in tables["groundtruth.txt"])
    depth_times = [t for t, _ in depth]
    pose_times = [t for t, _ in poses]
    frames = []
    for stamp, path in sorted((float(t), path) for t, path in tables["rgb.txt"]):
        d = _nearest(depth_times, stamp)
        p = _nearest(pose_times, stamp)
        frames.append(Frame(
            stamp, path,
            depth[d][1] if d is not None and abs(depth_times[d] - stamp) <= .03 else None,
            _pose(poses[p][1]) if p is not None and abs(pose_times[p] - stamp) <= .025 else None,
        ))
    return frames


def read_images(archive, paths):
    """Decode selected RGB and depth files in one sequential archive pass."""
    pending = set(paths)
    result = {}
    with tarfile.open(archive, "r|gz") as source:
        for member in source:
            if not pending:
                break
            parts = PurePosixPath(member.name).parts
            key = "/".join(parts[-2:])
            if member.isfile() and key in pending:
                flags = cv.IMREAD_UNCHANGED if key.startswith("depth/") else cv.IMREAD_COLOR
                result[key] = decode_image(source.extractfile(member).read(), key, flags)
                pending.remove(key)
    if pending:
        raise ValueError(f"Faltan {len(pending)} imágenes dentro del archivo TUM.")
    return result
