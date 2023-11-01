import cv2
import numpy as np
import mss
import mss.tools
import time
from PIL import Image
from ultralytics import YOLO


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
                 colors = {
                     0: (0 ,255 ,0),
                     1: (255, 0, 0),
                     2: (0, 0, 255),
                     3: (255, 255, 0),
                 }

                 while True:
                        previous_time = time.time()

                        # Grab the monitor
                        sct_img = sct.grab(monitor)
                        frame = Image.frombytes('RGB', (sct_img.width, sct_img.height), sct_img.rgb)
                        frame = cv2.cvtColor(np.array(frame), cv2.COLOR_BGR2RGB)

                        # predicting
                        results = model.predict(frame, stream=True)

                        for result in results:
                            count = 0
                            boxes = result.boxes.cpu().numpy()
                            i = 0
                            for box in boxes:  # iterate boxes
                                r = box.xyxy[0].astype(int)  # get corner points as int
                                class_tag  = result.names[int(box.cls[i])] # find labels
                                cv2.rectangle(frame, r[:2], r[2:], colors[box.cls[i]], 2)
                                cv2.putText(frame,str(class_tag),(r[0], r[1] - 10), cv2.FONT_HERSHEY_SIMPLEX,1,(0, 225 ,0),2,3)

                                i = i + 0
                        # calculate fps
                        txt1 = 'fps: %.1f' % (1 / (time.time() - previous_time))
                        cv2.putText(
                            frame,
                            txt1,
                            (10,50),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (0, 225 ,0),
                            2,
                            3)

                        cv2.imshow('SoccerVision', frame)
                        if cv2.waitKey(1) & 0xff == ord('q'):
                            cv2.destroyAllWindows()
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
    stream = ScreenCapture(None)
    model = YOLO("../Models/yolov8/yolov8l.pt")
    stream.stream_detection(model, 2)
