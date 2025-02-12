import torch
import random
import numpy as np
import cutlass

np.random.seed(1234)
random.seed(1234)
torch.manual_seed(123)

# This controls whether the C++ GEMM declaration will be printed at each step.
# Set to `False` to omit this information.
print_module = True

M = 2048
N = 2048
K = 8192

dtype = torch.bfloat16
type_A = torch.bfloat16
type_B = torch.bfloat16
type_C = torch.bfloat16
type_D = torch.bfloat16

scale = 4
tensor_A = torch.ceil(torch.randn(K, M, dtype=type_A, device="cuda") * scale)
tensor_B = torch.ceil(torch.randn(K, N, dtype=type_B, device="cuda") * scale)
tensor_C = torch.ceil(torch.randn(M, N, dtype=type_C, device="cuda") * scale)

alpha = 1.0
beta = 0.0

tensor_D = torch.zeros_like(tensor_C, dtype=type_D, device="cuda")

# We specify `element_accumulator` here so as to match the kernel run by NumPy below. However,
# specifying `element_accumulator` is not required if it is the same as `element`
plan = cutlass.Gemm(
    element=dtype,
    layout_A=cutlass.LayoutType.ColumnMajor,
    layout_B=cutlass.LayoutType.RowMajor,
    layout_C=cutlass.LayoutType.RowMajor,
    element_accumulator=np.float32,
)

tiles = plan.tile_descriptions()
print("{} tile descriptions returned".format(len(tiles)))
idx = 0
td = tiles[idx]
print("Tile description {} is: {}".format(idx, td))
plan.compile(td)

plan.run(tensor_A.t(), tensor_B, tensor_C, tensor_D, print_module=print_module)

tensor_D_pytorch = (alpha * (tensor_A.t() @ tensor_B)) + (beta * tensor_C)
print(torch.allclose(tensor_D, tensor_D_pytorch, 1e-6 * K, 1e-6 * K))
