

import torch
import torchvision

class Scale_Model(torch.nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.model = torchvision.models.segmentation.deeplabv3_resnet50(pretrained=True)
        self.scale_model = torch.nn.Sequential(torch.nn.Conv2d(3, 64, 5), torch.nn.ReLU(),torch.nn.Conv2d(64, 64, 5),torch.nn.ReLU())
        self.linear_scale = torch.nn.Linear(64, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input = x
        x = self.scale_model(x)
        x = torch.sum(x, dim=(2, 3)) / (x.shape[2] * x.shape[3])
        x = torch.nn.functional.relu(self.linear_scale(x))
        input = torch.nn.functional.interpolate(input, size=(input.shape[2]*x, input.shape[3]*x))
        x = self.model(input)
        return x