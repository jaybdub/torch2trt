# Monocular Depth Estimation (MiDaS)

This example demonstrates monocular depth estimation using the MiDas model from https://github.com/intel-isl/MiDaS.

<img src="assets/grocery_256.jpg"/><img src="assets/output_trt.jpg"/>

- PyTorch - ~12FPS
- TensorRT FP32 - 54.87 FPS
- TensorRT FP16 - 100.78 FPS
- TensorRT INT8 - 140.87 FPS

These benchmarks are performed on NVIDIA Jetson Xavier NX, with batch size 1, 256x256 image resolution.

## Instructions

1. Clone the MiDaS repository into this example's folder

    ```bash
    git clone https://github.com/intel-isl/MiDaS
    ```

2. To resolve a torch2trt tracing error, modify the following line https://github.com/intel-isl/MiDaS/blob/f275ca1c6f9af17aabe6f1e024b2084d7b84abb0/midas/blocks.py#L117 to read

    ```bash
    x = nn.functional.interpolate(
        x, scale_factor=self.scale_factor, mode=self.mode, align_corners=False
    )
    ```

3. Download the [MiDaS small model](https://github.com/intel-isl/MiDaS/releases/download/v2_1/model-small-70d6b9c8.pt) weights into this example's folder.  Name them ``midas_small.pt``.
4. Run ``convert.py`` from this example's folder to generate the optimized TensorRT engine

    ```bash
    python3 convert.py
    ```
5. Run ``run.py`` to generate an output visualization.

    ```bash
    python3 run.py
    ```

That's it!  Feel free to dig through the example code to learn more.  Or if you have any questions, open a discussion or issue on torch2trt's GitHub.

