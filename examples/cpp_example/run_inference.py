import tensorrt as trt
import pycuda
import pycuda.driver as cuda
import pycuda.autoinit
import pycuda.gpuarray
import cv2
import numpy as np

#=====================================
# Initialize TensorRT Engine / Context
#=====================================

logger = trt.Logger()
runtime = trt.Runtime(logger)

with open('model.engine', 'rb') as f:
    engine_bytes = f.read()
    engine = runtime.deserialize_cuda_engine(engine_bytes)
    
context = engine.create_execution_context()

#=====================================
# Load and preprocess image
#=====================================

image = cv2.imread('dog.jpg')

# resize
image = cv2.resize(image, (224, 224))

# convert bgr -> rgb
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# FP32
image = image.astype(np.float32)

# normalize
mean = 255.0 * np.array([0.485, 0.456, 0.406], dtype=np.float32)
stdev = 255.0 * np.array([0.229, 0.224, 0.225], dtype=np.float32)
image = (image - mean) / stdev

# convert HWC -> CHW
image = np.transpose(image, (2, 0, 1))

# make data contiguous (so underlying data is actually CHW)
image = np.ascontiguousarray(image)

#=====================================
# Execute image using TensorRT
#=====================================

output = np.zeros(1000, dtype=np.float32)
image_gpu = pycuda.gpuarray.to_gpu(image)
output_gpu = pycuda.gpuarray.to_gpu(output)

context.execute(batch_size=1, bindings=[image_gpu.ptr, output_gpu.ptr])

#=====================================
# Parse results and print
#=====================================

def softmax(x):
    expx = np.exp(x)
    return expx / np.sum(expx)

output = output_gpu.get()
output = softmax(output)

top_5_idx = np.argsort(output)[::-1][0:5]

with open('labels.txt', 'r') as f:
    labels = f.readlines()
    
for rank, idx in enumerate(top_5_idx):
    print('%f - %s' % (float(output[idx]), labels[idx].strip('\n')))