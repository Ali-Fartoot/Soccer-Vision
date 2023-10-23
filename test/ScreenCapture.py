import unittest
from PIL import Image
import numpy

""" Will split this tests up into other files"""


class CaptureVisionTest(unittest.TestCase):
    def test_frame_size(self):
        image = Image.open("./test.png")
        height, width = image.size

        self.assertEqual(height, 1920)  # add assertion here
        self.assertEqual(width, 1080)  # add assertion here


if __name__ == '__main__':
    unittest.main()
