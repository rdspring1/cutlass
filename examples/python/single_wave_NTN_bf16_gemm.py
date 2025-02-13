import torch
import random
import numpy as np
import cutlass
from cutlass.utils.profiler import CUDAEventProfiler
from cutlass.backend.library import TileDescription, MathInstruction, OpcodeClass
from cutlass import KernelScheduleType, EpilogueScheduleType, TileSchedulerType
from cutlass import DataType

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

cluster_shape = (2, 1, 1)
threadblock_shape = (128, 256, 64)
warp_count = [4, 1, 1]
stages = 0
math_instruction = MathInstruction((64, 256, 16), DataType.bf16, DataType.bf16, DataType.f32, OpcodeClass.TensorOp)
kernel_schedule = KernelScheduleType.TmaWarpSpecialized
epilogue_schedule = EpilogueScheduleType.TmaWarpSpecialized
tile_scheduler = TileSchedulerType.Default
td = TileDescription(threadblock_shape, stages, warp_count, math_instruction, cluster_shape, kernel_schedule, epilogue_schedule, tile_scheduler)
print(td)

plan.compile(td)
plan.run(tensor_A.t(), tensor_B, tensor_C, tensor_D, print_module=print_module)

warmup_iterations = 10
profile_iterations = 50
# Profile CUTLASS fused kernel
duration = CUDAEventProfiler(
    plan, warmup_iterations, profile_iterations,
    tensor_A.t(), tensor_B, tensor_C, tensor_D)()
print(f"CUTLASS duration: {duration:.2f} ms")

tensor_D_pytorch = (alpha * (tensor_A.t() @ tensor_B)) + (beta * tensor_C)
print(torch.allclose(tensor_D, tensor_D_pytorch, 1e-6 * K, 1e-6 * K))
