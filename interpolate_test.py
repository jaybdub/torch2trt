import torch
from torch2trt import torch2trt
from torchvision.transforms import ToTensor, ToPILImage
from PIL import Image

data = torch.randn(1, 3, 256, 256).cuda()


class Interpolate(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.nn.functional.interpolate(x, scale_factor=0.5, mode="nearest")


model = Interpolate().cuda()

print('Optimizing with torch2trt...')
model_trt = torch2trt(model, [data], fp16_mode=True)

data = Image.open('grocery_256.jpg')
data = ToTensor()(data).cuda()[None, ...]

out_torch = model(data)
out_trt = model_trt(data)

image_torch = ToPILImage()(out_torch.cpu().detach()[0])
image_trt = ToPILImage()(out_trt.cpu().detach()[0])

image_torch.save('image_torch.jpg')
image_trt.save('image_trt.jpg')

err = torch.max(torch.abs(out_torch - out_trt))


print(err)