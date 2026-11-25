import unittest

import cv2 as cv
import numpy as np

from decorador_ar.geometry import Camera, Pose
from decorador_ar.photo_furniture import Support, rectangle
from decorador_ar.render import box, draw_furniture_depth
from decorador_ar.supported_objects import prepare_accessory, support_points


class FurnitureDepthTest(unittest.TestCase):
    def test_front_face_wins_independently_of_mesh_order(self):
        camera = Camera(np.array([[80., 0, 48], [0, 80., 48], [0, 0, 1]]),
                        np.zeros(5), (96, 96))
        pose = Pose(np.zeros((3, 1)), np.zeros((3, 1)))
        front = box(-.2, -.2, 2., .4, .4, .2, (0, 0, 255))
        back = box(-.4, -.4, 3., .8, .8, .2, (255, 0, 0))
        first = np.zeros((96, 96, 3), np.uint8)
        second = first.copy()
        draw_furniture_depth(first, (front, back), pose, camera)
        shared_depth = np.zeros(second.shape[:2], np.float32)
        draw_furniture_depth(second, (back,), pose, camera, inverse_depth=shared_depth)
        draw_furniture_depth(second, (front,), pose, camera, inverse_depth=shared_depth)
        np.testing.assert_array_equal(first, second)
        self.assertGreater(first[48, 48, 2], 0)
        self.assertEqual(first[48, 48, 0], 0)

    def test_accessories_stay_inside_support_and_below_next_shelf(self):
        camera = Camera(np.array([[80., 0, 48], [0, 80., 48], [0, 0, 1]]),
                        np.zeros(5), (96, 96))
        pose = Pose(cv.Rodrigues(np.diag([1., -1., -1.]))[0], np.array([[0.], [0.], [2.]]))
        support = Support("Balda", pose, camera, rectangle(.4, .3), .12)
        np.testing.assert_allclose(support_points(pose, camera, [(48, 48)]), [[0, 0]])
        plate = prepare_accessory("plato", (48, 48), support)
        self.assertIs(plate.attached_to, support)
        with self.assertRaisesRegex(ValueError, "sale del apoyo"):
            prepare_accessory("plato", (54, 48), support)
        with self.assertRaisesRegex(ValueError, "demasiado alto"):
            prepare_accessory("jarron", (48, 48), support)
