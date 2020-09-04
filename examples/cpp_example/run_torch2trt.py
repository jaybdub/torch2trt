import torch
import torchvision
from torch2trt import torch2trt

model = torchvision.models.resnet18(pretrained=True).cuda().eval()

# convert with torch2trt
data = torch.zeros(1, 3, 224, 224).cuda()
model_trt = torch2trt(model, [data])
    
# save raw TensorRT engine
with open('model.engine', 'wb') as f:
    engine_bytes = model_trt.engine.serialize()
    f.write(engine_bytes)
    