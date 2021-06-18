from torch2trt.torch2trt import *
from torch2trt.module_test import add_module_test


@tensorrt_converter('torch.matmul')
@tensorrt_converter('torch.Tensor.__matmul__')
def convert_matmul(ctx):
    input_a = ctx.method_args[0]
    input_b = ctx.method_args[1]
    input_a_trt, input_b_trt = add_missing_trt_tensors(ctx.network, [input_a, input_b])
    output = ctx.method_return
    layer = ctx.network.add_matrix_multiply(input_a_trt, trt.MatrixOperation.NONE, input_b_trt, trt.MatrixOperation.NONE)
    output._trt = layer.get_output(0)