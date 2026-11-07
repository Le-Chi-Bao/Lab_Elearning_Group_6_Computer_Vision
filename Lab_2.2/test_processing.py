import unittest
import numpy as np
from image_processing import brightness, contrast, negative, threshold, mean_filter, gaussian_filter, sharpen


class ProcessingTests(unittest.TestCase):
    def test_point_operations(self):
        image = np.array([[0, 100, 200, 255]], dtype=np.uint8)
        np.testing.assert_array_equal(brightness(image, 40), [[40, 140, 240, 255]])
        np.testing.assert_array_equal(brightness(image, -40), [[0, 60, 160, 215]])
        np.testing.assert_array_equal(contrast(image, 1.5), [[0, 150, 255, 255]])
        np.testing.assert_array_equal(negative(negative(image)), image)
        np.testing.assert_array_equal(threshold(np.array([[127, 128, 129]], dtype=np.uint8)), [[0, 255, 255]])

    def test_filters_preserve_constant_image_and_channels(self):
        image = np.full((15, 15, 3), [30, 100, 200], dtype=np.uint8)
        for operation in (mean_filter, gaussian_filter, sharpen):
            np.testing.assert_array_equal(operation(image), image)

    def test_impulse_response(self):
        image = np.zeros((11, 11), dtype=np.uint8)
        image[5, 5] = 225
        self.assertEqual(int(mean_filter(image, 3)[5, 5]), 25)
        result = gaussian_filter(image)
        self.assertGreater(int(result[5, 5]), int(result[5, 4]))
        np.testing.assert_array_equal(result, result[::-1, ::-1])


if __name__ == '__main__':
    unittest.main()
