# EAST Text Detection 

This example demonstrates how to perform TensorRT optimized text region detection using torch2trt.
For this example we optimize the PyTorch reimplementation of EAST located at https://github.com/SakuraRiven/EAST.

<img src="assets/sign_east_trt.jpg"></img>

The performance of this model (excluding pre/post-processing) is listed below

- PyTorch - 10.13 FPS
- TensorRT (FP16) - 95.58 FPS
- TensorRT (INT8) - 188.87 FPS
- TensorRT (INT8 / DLA) - 60FPS

All benchmarks are performed on Jetson Xavier NX, batch size 1, 256x256 image size.  

## Instructions

1. Clone the PyTorch EAST repository into this example directory

    ```bash
    git clone https://github.com/SakuraRiven/EAST
    ```

2. Download the model weights into EAST/pths as instructed in the EAST project's README, with the slight modification that they should be placed in this examples ``pths`` folder instead.
3. Run the optimization script, ``optimize_and_detect.py`` from this directory to optimize the model, visualize detections, and benchmark performance.  The output images will be stored in this directory.

    ```bash
    python3 optimize_and_detect.py
    ```
    
    > By default, this will optimize with FP16 precision.  Please see the torch2trt documentation for instructions on other precisions, or DLA usage.  If you have any questions, don't hesitate to start a discussion or raise an issue on torch2trt's GitHub.

4. (optional) Run the modified ``load_and_detect.py`` to visualize the output if you've already optimized the model.

That's it!  If you have any feedback on this example, feel free to open an issue or GitHub discussion.

## Special thanks

@amitgnv for his recommendation to optimize this PyTorch model.