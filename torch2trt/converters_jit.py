from .torch2trt import *


@tensorrt_converter_jit('prim::Constant')
def convert_constant(node):
    output = node.outputs()[0]
    layer = node.add_layer(trt.INetworkDefinition.add_constant, output.shape()[1:], node.value().cpu().numpy())
    output.set_trt(layer.get_output(0))
    
    
@tensorrt_converter_jit('prim::Param')
def convert_param(node):
    for val in node.outputs():
        if val.is_tensor():
            input_trt = node.add_layer(trt.INetworkDefinition.add_input, val.name(), val.dtype_trt(), val.shape()[1:])
            val.set_trt(input_trt)
            
            
@tensorrt_converter_jit('aten::mul')
def convert_mul(node):
    layer = node.add_layer(trt.INetworkDefinition.add_elementwise,
        node.inputs()[0].get_trt(),
        node.inputs()[1].get_trt(),
        trt.ElementWiseOperation.PROD
    )
    node.outputs()[0].set_trt(layer.get_output(0))