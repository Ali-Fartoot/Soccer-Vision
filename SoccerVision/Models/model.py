import numpy as np
from super_gradients.training import models
import torch

class BasedModel:
    def __init__(self,num_classes: int, model_path: str, name: str = None, device: str = None):
        self.model_path: str = model_path
        self.name: str = name
        self.num_classes: int = num_classes
        self.device: str = device or 'cuda' if torch.cuda.is_available() else 'cpu'


    def __call__(self, input_path:str):
        self._net = models.get(self.name, num_classes=self.num_classes,
                               checkpoint_path=self.model_path)

        return self._net.to(self.device).predict(input_path)




