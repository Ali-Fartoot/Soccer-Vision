import cv2
import numpy as np
import mss
import mss.tools
import time
from SoccerVision.Models.yolo_nas_l import YoloNasL


def screenshot(monitor_number: int):
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
        output = f"./Cache/input.png".format(**monitor)

        # Grab the data
        sct_img = sct.grab(monitor)

        # Save to the picture file
        mss.tools.to_png(sct_img.rgb, sct_img.size, output=output)


class ScreenCapture():

    def __init__(self, out_path):
        self.buffer = []

    def stream_detection(self, model):

        prev_frame_time = 0
        new_frame_time = 0
        count = 0
        while (True):
            try:
                start = time.time()
                screenshot(2)
                # read screenshot (input)
                image = cv2.imread(f"./Cache/input.png")
                image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                """ 
                TODO: transform, reshape and everything that needed!
                """
                start_model = time.time()

                # feed forward to model
                detected_image = model(image)
                detected_image.save("./Cache/")

                # save output
                image = cv2.imread(f"./Cache/pred_0.jpg")
                end_model = time.time() - start

                print("model latency: ", end_model)
                new_frame_time = time.time()
                fps = 1 / (new_frame_time - prev_frame_time)
                prev_frame_time = new_frame_time

                # calculate fps and put it on image
                cv2.putText(image, str(fps), (3, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 255, 0), 1, cv2.LINE_AA)
                cv2.imshow("myphoto", image)
                if cv2.waitKey(1) == ord("q"):
                    break

                """
                TODO: image show
                """
                # check app latency
                end = time.time() - start
                print("app latency: ", end)
            except KeyboardInterrupt:
                break
        return count

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
    print(stream.stream_detection(model))
