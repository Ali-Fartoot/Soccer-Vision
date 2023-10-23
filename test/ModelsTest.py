import unittest
from SoccerVision.Models import yolo
import time


class ModelsTest(unittest.TestCase):
        def check_input_model(self):
            #latency
            self.start = time.perf_counter()
            self.net = YoloNasL(num_classes=4)
            print(self.net.__dict__)


        def check_output_model(self):
            path = "./foo.jpg"
            self.net(path)
            # latency
            self.end = time.perf_counter() - self.start

        def check_model_latency(self):

            print('{:.6f}s for the calculation'.format(self.end))

        def test_model(self):
            self.check_input_model()
            self.check_output_model()
            self.check_model_latency()


if __name__ == '__main__':
    unittest.main()
