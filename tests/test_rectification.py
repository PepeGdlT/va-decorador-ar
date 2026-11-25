import unittest

import cv2 as cv
import numpy as np

from decorador_ar.rectification import Plane, place_rectified, project


class RectifiedPlacementTest(unittest.TestCase):
    def test_picture_uses_top_right_anchor_and_rotates_in_the_wall_plane(self):
        corners = np.float32([[35, 15], [285, 31], [280, 192], [28, 217]])
        plane = Plane.from_reference(corners, 3, 2)
        image = np.zeros((240, 320, 3), np.uint8)
        mask = np.ones(image.shape[:2], bool)
        art = np.full((20, 40, 3), 220, np.uint8)
        anchor = project(plane.to_image, [[2, .5]])[0]
        result, flat = place_rectified(image, mask, plane, "wall", anchor, 1,
                                       angle=12, texture=art)
        np.testing.assert_allclose(result.quad[1], anchor, atol=1e-5)
        right = flat[1] - flat[0]
        self.assertAlmostEqual(np.linalg.norm(right), 1.)
        self.assertAlmostEqual(np.degrees(np.arctan2(right[1], right[0])), 12.)
        self.assertAlmostEqual(np.linalg.norm(flat[3] - flat[0]), .5)

    def test_rotating_texture_changes_rug_orientation_in_floor_plane(self):
        corners = np.float32([[210, 150], [490, 150], [680, 490], [20, 490]])
        plane = Plane.from_reference(corners, 4, 5)
        region = cv.fillConvexPoly(np.zeros((500, 700), np.uint8), corners.astype(np.int32), 1)
        image = np.zeros((*region.shape, 3), np.uint8)
        rug = np.full((20, 40, 3), 220, np.uint8)
        anchor = project(plane.to_image, [[2, 1.]])[0]
        for vertical, width, depth in ((False, 1., .5), (True, .5, 1.)):
            with self.subTest(vertical=vertical):
                result, flat = place_rectified(image, region.astype(bool), plane, "floor", anchor, 1.,
                                               vertical=vertical, texture=rug)
                self.assertGreater(result.coverage, .98)
                self.assertAlmostEqual(np.linalg.norm(flat[1] - flat[0]), width)
                self.assertAlmostEqual(np.linalg.norm(flat[3] - flat[0]), depth)
