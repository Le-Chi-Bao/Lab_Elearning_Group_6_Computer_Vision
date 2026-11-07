import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from wavelet_lab import hamming_distance, wavelet_hash


class WaveletTests(unittest.TestCase):
    def test_hamming_known_vectors(self):
        self.assertEqual(hamming_distance([0, 1, 1, 0], [1, 1, 0, 0]), 0.5)
        with self.assertRaises(ValueError):
            hamming_distance([0], [0, 1])

    def test_hashes_and_brightness_invariance(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.png"
            bright = Path(directory) / "bright.png"
            pixels = np.random.default_rng(7).integers(20, 180, (128, 128), dtype=np.uint8)
            Image.fromarray(pixels).save(source)
            Image.fromarray(pixels + 30).save(bright)
            for wavelet in ("haar", "db2", "db4"):
                original = wavelet_hash(source, wavelet)
                self.assertEqual(original.shape, (64,))
                self.assertEqual(original.dtype, np.dtype(bool))
                self.assertEqual(hamming_distance(original, wavelet_hash(source, wavelet)), 0)
                self.assertEqual(hamming_distance(original, wavelet_hash(bright, wavelet)), 0)


if __name__ == "__main__":
    unittest.main()
