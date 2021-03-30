import torch
from torch2trt import torch2trt


data = torch.randn(1, 3, 256, 256).cuda()


class Interpolate(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.nn.functional.interpolate(x, scale_factor=0.5, mode="nearest")


model = Interpolate().cuda()

print('Optimizing with torch2trt...')
model_trt = torch2trt(model, [data], fp16_mode=True)

data = torch.randn(1, 3, 256, 256).cuda()

err = torch.max(torch.abs(model(data) - model_trt(data)))

print(err)