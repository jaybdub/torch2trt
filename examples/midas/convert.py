from PIL import Image
import sys
import time
import cv2
import torch
sys.path.append('MiDaS')
from torch2trt import torch2trt, trt
from midas.midas_net_custom import MidasNet_small
from torchvision.transforms import Compose
from midas.midas_net import MidasNet
from midas.midas_net_custom import MidasNet_small
from midas.transforms import Resize, NormalizeImage, PrepareForNet


WEIGHT_PATH = 'model_small.pt'
TRT_OUTPUT_MODEL_PATH = 'midas_trt.pth'

# requires patching Interpolate block in midas to not use interpolate as attribute
model = MidasNet_small(WEIGHT_PATH, features=64, backbone="efficientnet_lite3", exportable=True, non_negative=True, blocks={'expand': True})

net_w, net_h = 256, 256


transform = Compose(
    [
        Resize(
            net_w,
            net_h,
            resize_target=None,
            keep_aspect_ratio=True,
            ensure_multiple_of=32,
            resize_method="upper_bound",
            image_interpolation_method=cv2.INTER_CUBIC,
        ),
        NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        PrepareForNet(),
    ]
)


model = model.cuda().eval()
data = torch.rand(1, 3, net_h, net_w).cuda()

print('Optimizing model with TensorRT...')
model_trt = torch2trt(model, [data])


torch.save(model_trt.state_dict(), TRT_OUTPUT_MODEL_PATH)

def benchmark_fps(data, model):
    torch.cuda.current_stream().synchronize()
    t0 = time.time()
    for i in range(50):
        out = model(data)
    torch.cuda.current_stream().synchronize()
    t1 = time.time()
    return 50 / (t1 - t0)

print('Profiling PyTorch vs. TensorRT')
model(data) # execute once for warmup
fps_torch = benchmark_fps(data, model)

model_trt(data) # execute once for warmup
fps_tensorrt = benchmark_fps(data, model_trt)

print('\tPyTorch: {fps}FPS'.format(fps=fps_torch))
print('\tTensorRT: {fps}FPS'.format(fps=fps_tensorrt))