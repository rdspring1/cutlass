import numpy as np
import random

import cutlass

# This controls whether the C++ GEMM declaration will be printed at each step. 
# Set to `False` to omit this information.
print_module = True

m = 2048
n = 2048
k = 8192

dtype = np.float16
type_A = np.float16
type_B = np.float16
type_C = np.float16
type_D = np.float16

np.random.seed(1234)
random.seed(1234)
scope_min = -4
scope_max = 4
tensor_A = np.ceil(np.random.uniform(low=scope_min, high=scope_max, size=(m, k)).astype(type_A))
tensor_B = np.ceil(np.random.uniform(low=scope_min, high=scope_max, size=(k, n)).astype(type_B))
tensor_C = np.ceil(np.random.uniform(low=scope_min, high=scope_max, size=(m, n)).astype(type_C))

alpha = np.float16(1.)
beta = np.float16(0.)

tensor_D = np.zeros(tensor_C.shape).astype(type_D)

# We specify `element_accumulator` here so as to match the kernel run by NumPy below. However,
# specifying `element_accumulator` is not required if it is the same as `element`
plan = cutlass.Gemm(element=dtype, layout=cutlass.LayoutType.RowMajor, element_accumulator=np.float32)
plan.run(tensor_A, tensor_B, tensor_C, tensor_D, print_module=print_module)

tensor_D_numpy = (alpha * (tensor_A @ tensor_B)) + (beta * tensor_C)
np.testing.assert_array_equal(tensor_D, tensor_D_numpy)
