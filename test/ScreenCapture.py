import unittest
from PIL import Image
import numpy

""" Will split this tests up into other files"""


class SoccerVisionTest(unittest.TestCase):
    def test_frame_size(self):
        image = Image.open("./test.png")
        height, width = image.size

        self.assertEqual(height, 1920)  # add assertion here
        self.assertEqual(width, 1080)  # add assertion here

    def validating_ouput_video(self):
        """ check if the video is on disk end of program or not!"""
        pass

    def check_input_model(self):
        """ check the input shape of model"""
        pass

    def test_latency(self):
        pass

    def test_buffer_delay(self):
        pass

    def check_input_onyx_model(self):
        pass

    def check_output_onyx_model(self):
        pass

    def check_onyx_model_latency(self):
        pass


if __name__ == '__main__':
    unittest.main()
