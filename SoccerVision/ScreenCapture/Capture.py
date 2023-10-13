import cv2
import numpy as np
import mss
import mss.tools
import time
from SoccerVision.Models.yolo_nas_l import YoloNasL
from PIL import Image



class ScreenCapture():

    def __init__(self, out_path):
        self.buffer = []

    def stream_detection(self, model, monitor_number: int ):

        prev_frame_time = 0
        new_frame_time = 0
        count = 0

        with mss.mss() as sct:
                    # Get information of monitor 2
                 mon = sct.monitors[monitor_number]

                    # The screen part to capture
                 monitor = {
                        "top": mon["top"],
                        "left": mon["left"],
                        "width": mon["width"],
                        "height": mon["height"],
                        "mon": monitor_number,
                    }
                 previous_time = 0

                 while True:
                        previous_time = time.time()
                        # Grab the data
                        sct_img = sct.grab(monitor)
                        frame = Image.frombytes('RGB', (sct_img.width, sct_img.height), sct_img.rgb)

                        # detected_image =
                        # image = image[ ::2, ::2, : ] # can be used to downgrade the input
                        frame = cv2.cvtColor(np.array(frame), cv2.COLOR_BGR2RGB)
                        detected_image = model(frame)
                        print(type(detected_image))

                        cv2.imshow('frame', detected_image[0].draw())

                        if cv2.waitKey(1) & 0xff == ord('q'):
                            cv2.destroyAllWindows()
                            break

                        txt1 = 'fps: %.1f' % (1 / (time.time() - previous_time))
                        print(txt1)

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
    stream = ScreenCapture(None)
    model = YoloNasL(4, None)
    stream.stream_detection(model,2)
