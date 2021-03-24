import sys
import time
import torch
from PIL import Image
sys.path.append('EAST')
from model import * # from EAST for EAST model class
from detect import * # from EAST for detect, plot_boxes
from torch2trt import torch2trt, TRTModule

IMAGE = 'assets/sign.jpg'
TRT_MODEL_OUTPUT = 'east_trt.pth'
TRT_IMAGE_OUTPUT = 'assets/sign_east_trt.jpg'
TORCH_IMAGE_OUTPUT = 'assets/sign_east_torch.jpg'
WIDTH = 256
HEIGHT = 256

_device = torch.device('cuda') # don't change

print('Loading PyTorch model...')
model = EAST()
model = model.cuda().eval()
model.load_state_dict(torch.load('pths/east_vgg16.pth'))

print('Loading TensorRT model...')
model_trt = TRTModule()
model_trt.load_state_dict(torch.load(TRT_MODEL_OUTPUT))

print('Executing model with PyTorch')
img = Image.open(IMAGE).resize((WIDTH, HEIGHT))
boxes = detect(img, model, _device)
plot_img = plot_boxes(img, boxes)	
plot_img.save(TORCH_IMAGE_OUTPUT)

# execute tensorrt and save output
print('Executing model with TensorRT')
img = Image.open(IMAGE).resize((WIDTH, HEIGHT))
boxes = detect(img, model_trt, _device)
plot_img = plot_boxes(img, boxes)	
plot_img.save(TRT_IMAGE_OUTPUT)
