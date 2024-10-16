

import torch
from config import *
from unets import UNetSmall


class Scale_Model(torch.nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.model = UNetSmall(im_size, input_channels, num_classes)
        self.scale_model = torch.nn.Sequential(torch.nn.Conv2d(input_channels, 64, 5), torch.nn.ReLU(),torch.nn.Conv2d(64, 64, 5),torch.nn.ReLU())
        self.linear_scale = torch.nn.Linear(64, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input = x
        x = self.scale_model(x)
        x = torch.sum(x, dim=(2, 3)) / (x.shape[2] * x.shape[3])
        x = torch.nn.functional.relu(self.linear_scale(x)) + 16/im_size
        input= torch.nn.functional.interpolate(input, size=(int(im_size*x[0]),int(im_size*x[0])))
        #input = torch.nn.Upsample(scale_factor=x)(input)
        print(input.shape)
        x = self.model(input)
        x= torch.nn.functional.interpolate(x, size=(im_size,im_size), mode = "nearest")
        return x