import unittest
import numpy as np
from canny_lab import manual_canny, opencv_canny, skimage_canny, edge_metrics, synthetic_scene


class CannyTests(unittest.TestCase):
    def test_constant_has_no_edges(self):
        gray = np.full((50, 50), 80, dtype=np.uint8)
        for function in (opencv_canny, skimage_canny):
            self.assertFalse(function(gray).any())
        self.assertFalse(manual_canny(gray)['edges'].any())

    def test_geometry_detected(self):
        gray, truth = synthetic_scene()
        for function in (opencv_canny, skimage_canny, lambda im: manual_canny(im)['edges']):
            self.assertGreater(edge_metrics(function(gray), truth)['f1'], 0.8)

    def test_metrics_and_parameters(self):
        truth = np.zeros((20,20), bool)
        truth[5:15,10] = True
        self.assertEqual(edge_metrics(truth,truth)['f1'], 1)
        self.assertEqual(edge_metrics(np.zeros_like(truth),truth)['f1'], 0)
        with self.assertRaises(ValueError):
            opencv_canny(np.zeros((20,20),np.uint8), low=100, high=50)


if __name__ == '__main__':
    unittest.main()
