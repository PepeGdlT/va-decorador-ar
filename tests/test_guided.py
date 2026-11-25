import unittest

import cv2 as cv
import numpy as np

from decorador_ar.guided import (floor_plane, floor_quad, polygon_coverage,
                                 wall_quad)


class GuidedPlacementTest(unittest.TestCase):
    def test_wall_uses_click_as_top_right_and_preserves_angle(self):
        quad = wall_quad((400, 120), width_px=200, aspect=2, angle_deg=12)
        np.testing.assert_allclose(quad[1], [400, 120], atol=1e-5)
        top = quad[1] - quad[0]
        self.assertAlmostEqual(np.linalg.norm(top), 200, places=4)
        self.assertAlmostEqual(np.degrees(np.arctan2(top[1], top[0])), 12, places=4)
        self.assertAlmostEqual(np.linalg.norm(quad[3] - quad[0]), 100, places=4)

    def test_floor_rectangle_is_projected_inside_trapezoid(self):
        mask = np.zeros((500, 700), np.uint8)
        visible = np.int32([[210, 150], [490, 150], [680, 490], [20, 490]])
        cv.fillConvexPoly(mask, visible, 1)
        plane = floor_plane(mask)
        quad = floor_quad((430, 225), mask, long_side=240,
                          orientation="horizontal", angle_deg=0, aspect=1.6)
        self.assertTrue(cv.isContourConvex(np.rint(plane).astype(np.int32)))
        self.assertTrue(cv.isContourConvex(np.rint(quad).astype(np.int32)))
        self.assertGreater(polygon_coverage(quad, mask.astype(bool)), .98)


if __name__ == "__main__":
    unittest.main()
