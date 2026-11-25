import unittest
from types import SimpleNamespace

import numpy as np

from decorador_ar.course_experiments import evaluate_translation_pair, ransac_sensitivity


class RegistrationExperimentTest(unittest.TestCase):
    def test_sensitivity_rejects_invalid_thresholds_before_loading_scenes(self):
        for thresholds in ((), (0,), (-1,), (float("nan"),)):
            with self.subTest(thresholds=thresholds):
                with self.assertRaisesRegex(ValueError, "umbrales de RANSAC"):
                    ransac_sensitivity({}, thresholds=thresholds)

    def test_sensitivity_reports_missing_features_instead_of_crashing(self):
        image = np.zeros((96, 96, 3), dtype=np.uint8)
        intrinsics = np.array([[80., 0, 48], [0, 80., 48], [0, 0, 1]])
        scenes = {
            name: SimpleNamespace(image=image, intrinsics=intrinsics)
            for name in ("bonus_01", "bonus_02")
        }
        with self.assertRaisesRegex(ValueError, "No hay suficientes puntos ORB"):
            ransac_sensitivity(scenes)

    def test_translation_experiment_requires_a_manual_wall_quad(self):
        with self.assertRaisesRegex(ValueError, "Marca wall_quad manualmente"):
            evaluate_translation_pair("missing-a.jpg", "missing-b.jpg", None)


if __name__ == "__main__":
    unittest.main()
