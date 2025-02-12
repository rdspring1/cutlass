import torch
import random
import numpy as np
import cutlass
from cutlass.utils.profiler import CUDAEventProfiler

np.random.seed(1234)
random.seed(1234)
torch.manual_seed(123)

# This controls whether the C++ GEMM declaration will be printed at each step. 
# Set to `False` to omit this information.
print_module = True

M = 4096 
N = 14336
K = 5120

dtype = torch.bfloat16
type_A = torch.bfloat16
type_B = torch.bfloat16
type_C = torch.bfloat16
type_D = torch.bfloat16

scale = 4
tensor_A = torch.ceil(torch.randn(M, K, dtype=type_A, device='cuda') * scale)
tensor_B = torch.ceil(torch.randn(N, K, dtype=type_B, device='cuda') * scale)
tensor_C = torch.ceil(torch.randn(M, N, dtype=type_C, device='cuda') * scale)

alpha = 1.
beta = 0.

tensor_D = torch.zeros_like(tensor_C, dtype=type_D, device='cuda')

# We specify `element_accumulator` here so as to match the kernel run by NumPy below. However,
# specifying `element_accumulator` is not required if it is the same as `element`
plan = cutlass.Gemm(
    element=dtype,
    layout_A=cutlass.LayoutType.RowMajor,
    layout_B=cutlass.LayoutType.ColumnMajor,
    layout_C=cutlass.LayoutType.RowMajor,
    element_accumulator=np.float32,
)
plan.compile()

warmup_iterations = 10
profile_iterations = 50
# Profile CUTLASS fused kernel
duration = CUDAEventProfiler(
    plan, warmup_iterations, profile_iterations,
    tensor_A, tensor_B.t(), tensor_C, tensor_D)()
print(f"CUTLASS duration: {duration:.2f} ms")

tensor_D_pytorch = (alpha * (tensor_A @ tensor_B.t())) + (beta * tensor_C)
print(torch.allclose(tensor_D, tensor_D_pytorch, 1e-6 * K, 1e-6 * K))
