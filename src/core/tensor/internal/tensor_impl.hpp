#pragma once
// Private seam for lfs_core. Backend operation headers stay here so a public
// include of core/tensor.hpp does not parse them.
#include "core/detail/gpu_backend_ops.hpp"
#include "core/detail/pointwise_lowering.hpp"
#include "core/detail/tensor_cpu_apply.hpp"
#include "core/detail/tensor_dtype_dispatch.hpp"
#include "core/detail/tensor_impl.hpp"
#include "core/tensor/internal/private_access.hpp"
