from SoccerVision.Models.model import BasedModel
import torch

class YoloNasL(BasedModel):
    def __init__(self, num_classes: int, device: str = None):

        self.model_path = "../../SoccerVision/Models/yolo_nas_l/ckpt_best.pth"
        self.name = "yolo_nas_l"
        self.num_classes = num_classes
        self.device: str =  device or 'cuda' if torch.cuda.is_available() else  "cpu"
        super().__init__(num_classes=self.num_classes, model_path=self.model_path, name=self.name, device=self.device)



if __name__ == "__main__":
    model = YoloNasL(4, None)
    output = model("../../test/foo.jpg")
    output.save("./Cache/")
