# Native model inference on GPU backends

LPIPS, MoGe 2, and SAM2 use the native `.lfw` weight files on both the CUDA
and Vulkan tensor backends. Select Vulkan before loading the model and creating
its input tensors:

```sh
LFS_TENSOR_BACKEND=vulkan ./build/LichtFeld-Studio
```

The C++ equivalent is `GpuBackendScope{GpuBackend::Vulkan}` around weight loading
and input creation. Model execution follows the backend of its input and weights;
mixing CUDA and Vulkan tensors in a model call is rejected. The preprocessing
MoGe path and Python SAM2 bindings also accept the Vulkan backend. Vulkan inference
does not require an available CUDA device.

The Vulkan implementation uses the tensor backend's existing device, memory pool,
and command recorder. Weight uploads use owned staging memory, and inference uses
Vulkan tensor programs plus shaders for image sampling, convolution packing,
pooling, activation functions, and coordinate grids. It does not transfer the
model to CUDA for execution.

FP32 and FP16 model storage are supported. Vulkan computes these native inference
operations in FP32 and converts their outputs to the model's storage dtype.
FP16 outputs can therefore differ from CUDA's specialized FP16 kernels. Attention
processes queries in tiles with an 8 MiB score target, and ordinary convolution
packs at most a 16 MiB target of columns per tile; these targets do not include
model weights, activations, or every temporary tensor.

The existing LPIPS and MoGe reference tests can be run against either backend
with `LFS_TENSOR_BACKEND`. Full MoGe parity is opt-in through
`LFS_MOGE2_WEIGHTS=/path/to/moge-2-vitb-normal.lfw`. SAM2 fixture tests require the
fixtures described in `tests/test_sam2.cpp`. Keep fixture comparisons separate
from direct comparisons between CUDA and Vulkan model outputs.

This support covers inference. The application's training path still requires
CUDA, and Vulkan inference currently uses the general tensor GEMM implementation.
