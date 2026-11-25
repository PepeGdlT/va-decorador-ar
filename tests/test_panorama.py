import unittest

import numpy as np

from decorador_ar.panorama import perspective_view


class PanoramaTest(unittest.TestCase):
    def test_zero_yaw_looks_at_equirectangular_center(self):
        panorama = np.zeros((180, 360, 3), np.uint8)
        panorama[:, :, 0] = np.arange(360, dtype=np.uint16) % 256
        view = perspective_view(panorama, 0, size=(101, 81))
        self.assertLess(abs(int(view[40, 50, 0]) - 180), 2)

    def test_yaw_rotates_center_ray(self):
        panorama = np.zeros((180, 360, 3), np.uint8)
        panorama[:, :, 0] = np.arange(360, dtype=np.uint16) % 256
        positive = perspective_view(panorama, 90, size=(101, 81))
        negative = perspective_view(panorama, -90, size=(101, 81))
        self.assertLess(abs(int(positive[40, 50, 0]) - 14), 2)
        self.assertLess(abs(int(negative[40, 50, 0]) - 90), 2)


if __name__ == "__main__":
    unittest.main()
