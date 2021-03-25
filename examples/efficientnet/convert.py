import torch
from torch2trt import torch2trt


model = torch.hub.load(
    "rwightman/gen-efficientnet-pytorch",
    "tf_efficientnet_lite3",
    pretrained=True,
    exportable=True
)
model = model.cuda().eval()

data = torch.randn(1, 3, 224, 224).cuda()

model_trt = torch2trt(model, [data], fp16_mode=False)

data = torch.randn(1, 3, 224, 224).cuda()

abs_error = torch.max(torch.abs(model(data) - model_trt(data)))
print(abs_error)