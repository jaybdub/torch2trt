from PIL import Image
import sys
import cv2
import torch
import numpy as np
sys.path.append('MiDaS')
from torch2trt import torch2trt, trt
from midas.midas_net_custom import MidasNet_small
from torchvision.transforms import Compose
from midas.midas_net import MidasNet
from midas.midas_net_custom import MidasNet_small
from utils import read_image
from midas.transforms import Resize, NormalizeImage, PrepareForNet
from torchvision.transforms import ToTensor
from torch2trt import TRTModule

IMAGE_PATH = 'grocery.jpg'
WIDTH = 256
HEIGHT = 256
TRT_MODEL_PATH = 'midas_trt.pth'
OUTPUT_PATH = 'output_trt.jpg'


print('Loading TensorRT model...')
model_trt = TRTModule()
model_trt.load_state_dict(torch.load(TRT_MODEL_PATH))

print('')
transform = Compose(
    [
        Resize(
            WIDTH,
            HEIGHT,
            resize_target=None,
            keep_aspect_ratio=False,  # we use False because we must guarantee 256x256 shape since this is what we optimized engine using
            ensure_multiple_of=32,
            resize_method="upper_bound",
            image_interpolation_method=cv2.INTER_CUBIC,
        ),
        NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        PrepareForNet(),
    ]
)

image = read_image(IMAGE_PATH)
data = transform({'image': image})['image']
data = torch.from_numpy(data)[None, ...].cuda()

output = model_trt(data)
output = output[0].detach().cpu().numpy()
output = (output - output.min()) / (output.max() - output.min())
output = (output * 255).astype(np.uint8)

print('Writing depth image output to {path}'.format(path=OUTPUT_PATH))
cv2.imwrite(OUTPUT_PATH, output)