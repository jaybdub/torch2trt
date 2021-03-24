import sys
import time
import torch
from PIL import Image
sys.path.append('EAST')
from model import * # from EAST for EAST model class
from detect import * # from EAST for detect, plot_boxes
from torch2trt import torch2trt

IMAGE = 'assets/sign.jpg'
TRT_MODEL_OUTPUT = 'east_trt.pth'
TRT_IMAGE_OUTPUT = 'assets/sign_east_trt.jpg'
TORCH_IMAGE_OUTPUT = 'assets/sign_east_torch.jpg'
WIDTH = 256
HEIGHT = 256

_device = torch.device('cuda') # don't change

print('Loading model...')
model = EAST()
model = model.cuda().eval()
model.load_state_dict(torch.load('pths/east_vgg16.pth'))

# optimize with torch2trt
print('Optimizing model with TensorRT...')
data = torch.randn(1, 3, HEIGHT, WIDTH).cuda()
model_trt = torch2trt(model, [data], fp16_mode=True)
torch.save(model_trt.state_dict(), TRT_MODEL_OUTPUT)

# execute pytorch and save output
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

# profile pytorch vs. tensorrt
def benchmark_fps(data, model):
    torch.cuda.current_stream().synchronize()
    t0 = time.time()
    for i in range(50):
        out = model(data)
    torch.cuda.current_stream().synchronize()
    t1 = time.time()
    return 50 / (t1 - t0)

print('Profiling PyTorch vs. TensorRT')
fps_torch = benchmark_fps(data, model)
fps_tensorrt = benchmark_fps(data, model_trt)

print('\tPyTorch: {fps}FPS'.format(fps=fps_torch))
print('\tTensorRT: {fps}FPS'.format(fps=fps_tensorrt))
