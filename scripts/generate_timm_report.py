import json
import glob
import os

FOLDER = "data/timm"
FOLDER
OUTPUT = "data/timm_table.md"

ONNX_JSONS = glob.glob("data/timm/**/*-onnx.json")

outputs = []

for path in ONNX_JSONS:
    with open(path, 'r') as f:
        data_onnx = json.load(f)
    
    path_no_onnx = path.replace("-onnx", "")
    if os.path.exists(path_no_onnx):
        with open(path_no_onnx, 'r') as f:
            data_t2t = json.load(f)
    else:
        data_t2t = None
    outputs.append((data_onnx, data_t2t))

outputs = [o for o in outputs if 'results' in o[0]]
outputs = sorted(outputs, key=lambda x: -float(x[0]['results']['fps_trt']))

with open(OUTPUT, 'w') as f:
    f.write("| Model | Input Size | FPS (Torch) | FPS (ONNX->TRT) | FPS (torch2trt->TRT) | Max Err (ONNX->TRT) | Max Err (torch2trt->TRT) |\n")
    f.write("|-------|---|----------|-----------------|----------------------|---------------------|--------------------------|\n")
    
   
    for data in outputs:
        model = data[0]['args']['model']
        data_onnx = data[0]
        data_t2t = data[1]
        size = data[0]['args']['size']
        input_size = f"{size}x{size}"
        fps_torch_onnx = round(data_onnx['results']['fps_torch'], 2)
        fps_trt_onnx = round(data_onnx['results']['fps_trt'], 2)
        err_onnx = round(data_onnx['results']['max_abs_error'], 5)
        if data_t2t is not None and 'results' in data_t2t:
            fps_torch_t2t = round(data_t2t['results']['fps_torch'], 2)
            fps_trt_t2t = round(data_t2t['results']['fps_trt'], 2)
            err_t2t = round(data_t2t['results']['max_abs_error'], 5)
        else:
            fps_torch_t2t = "NA"
            fps_trt_t2t = "NA"
            err_t2t = "NA"
            
        f.write(f"| {model} | {input_size} | {fps_torch_onnx} | {fps_trt_onnx} | {fps_trt_t2t} | {err_onnx} | {err_t2t} |\n")