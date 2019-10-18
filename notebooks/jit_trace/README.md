torch._C.Graph
  outputs() -> torch._C.Value
  inputs() -> torch._C.Value
  nodes() -> torch._C.Node
torch._C.Value
  type() -> any
  isCompleteTensor() -> True / False indicating if type() returns tensor
torch._C.Node
  scopeName() -> str indicating module scope
  inputs() -> torch._C.Value
  outputs() 
Tensor
  sizes() -> List of int
  strides() -> List of int
  dim() -> List of int
   
graph.outputs() -> iter of torch._C.Value
torch._C.Value.node() -> torch._C.Node
