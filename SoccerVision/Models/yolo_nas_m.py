import sys
from SoccerVision.Models.model import BasedModel


class YoloNasM(BasedModel):
    def __init__(self):
        super().__init__(self)
        self.model_path = "../../../SoccerVision/Notebook/yolo_nas_m/ckpt_best.pth"
