from PIL import Image
import sys
import cv2
import torch
sys.path.append('MiDaS')
from torch2trt import torch2trt, trt
from midas.midas_net_custom import MidasNet_small
from torchvision.transforms import Compose
from midas.midas_net import MidasNet
from midas.midas_net_custom import MidasNet_small
from midas.transforms import Resize, NormalizeImage, PrepareForNet

model_path = 'model_small.pt'

# requires patching Interpolate block in midas to not use interpolate as attribute
model = MidasNet_small(model_path, features=64, backbone="efficientnet_lite3", exportable=True, non_negative=True, blocks={'expand': True})

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

print('Optimizing model...')
model_trt = torch2trt(model, [data])

torch.save(model_trt.state_dict(), 'midas_trt.pth')

print(model_trt.engine.num_layers)
print(torch.max(torch.abs(model(data) - model_trt(data))))

