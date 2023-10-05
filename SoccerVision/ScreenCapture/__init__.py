import cv2
import numpy as np
import os
import pyautogui


class ScreenCapture():

    def __init__(self, out_path):
        self.buffer = []

    def stream_detection(self, model):
        while (True):
            try:
                img = pyautogui.screenshot()
                image = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
                """ 
                TODO: transform, reshape and everything that needed!
                """
                detected_image = model(image)
                self.buffer.append(detected_image)

                """
                TODO: image show
                """
            except KeyboardInterrupt:
                break

    def save_to_video(self, out_path):
        height, width, channels = self.buffer[0].shape
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(out_path, fourcc, 20.0, (width, height))
        for image in self.buffer:
            out.write(image)
            StopIteration(0.5)
        out.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    img = pyautogui.screenshot()
    img.save(r"../../test/test.png")
