/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "internal/tensor_broadcast.hpp"
#include "core/logger.hpp"
#include "internal/cuda_stream_context.hpp"
#include "internal/tensor_impl.hpp"
#include "internal/tensor_ops.hpp"

namespace lfs::core {

    Tensor broadcast_to(const Tensor& src, const TensorShape& target) {
        LFS_ASSERT_MSG(src.is_valid(),
                       "Cannot broadcast an invalid tensor");
        const bool supported_dtype =
            src.dtype() == DataType::Float32 || src.dtype() == DataType::Bool ||
            (src.device() == Device::CPU && src.dtype() == DataType::Int32);
        LFS_ASSERT_MSG(supported_dtype,
                       std::format("broadcast_to does not support {} on {}",
                                   dtype_name(src.dtype()), device_name(src.device())));

        // An empty dimension vector represents a scalar, not an incompatibility.
        if (src.shape() == target) {
            return src.clone();
        }

        if (src.numel() == 0 || target.elements() == 0) {
            return Tensor::empty(target, src.device(), src.dtype());
        }

        // Check if shapes are compatible for broadcasting
        auto src_dims = src.shape().dims();
        auto target_dims = target.dims();

        // Validate broadcasting rules
        auto broadcast_shape = broadcast::shape(src_dims, target_dims);
        LFS_ASSERT_MSG(!broadcast_shape.empty() && broadcast_shape == target_dims,
                       std::format("Cannot broadcast shape {} to {}", src.shape().str(), target.str()));

        Tensor result;
        if (src.device() == Device::CUDA) {
            const cudaStream_t execution_stream =
                getCurrentCUDAStream() ? getCurrentCUDAStream() : src.stream();
            src.sync_to_stream(execution_stream);
            CUDAStreamGuard guard(execution_stream);
            result = Tensor::empty(target, src.device(), src.dtype());
        } else {
            result = Tensor::empty(target, src.device(), src.dtype());
        }

        if (src.device() == Device::CUDA) {
            const auto& src_strides = src.strides();
            const bool strided = !src.is_contiguous();

            if (src.dtype() == DataType::Bool) {
                if (strided) {
                    tensor_ops::launch_broadcast_strided_bool(
                        src.ptr<unsigned char>(), result.ptr<unsigned char>(),
                        src_dims.data(), src_strides.data(), target_dims.data(),
                        src_dims.size(), target_dims.size(), result.numel(), result.stream());
                } else {
                    tensor_ops::launch_broadcast_bool(
                        src.ptr<unsigned char>(), result.ptr<unsigned char>(),
                        src_dims.data(), target_dims.data(),
                        src_dims.size(), target_dims.size(), result.numel(), result.stream());
                }
            } else if (src.dtype() == DataType::Float32) {
                if (strided) {
                    tensor_ops::launch_broadcast_strided(
                        src.ptr<float>(), result.ptr<float>(),
                        src_dims.data(), src_strides.data(), target_dims.data(),
                        src_dims.size(), target_dims.size(), result.numel(), result.stream());
                } else {
                    tensor_ops::launch_broadcast(
                        src.ptr<float>(), result.ptr<float>(),
                        src_dims.data(), target_dims.data(),
                        src_dims.size(), target_dims.size(), result.numel(), result.stream());
                }
            } else {
                LFS_ASSERT_MSG(false,
                               "Unsupported dtype reached CUDA broadcast dispatch");
            }
        } else {
            if (!src.is_contiguous())
                return broadcast_to(src.contiguous(), target);
            if (src.dtype() == DataType::Bool) {
                const unsigned char* src_data = src.ptr<unsigned char>();
                unsigned char* dst_data = result.ptr<unsigned char>();
                for (size_t i = 0; i < result.numel(); ++i) {
                    size_t src_idx = broadcast::index(i, target_dims, src_dims);
                    LFS_DEBUG_ASSERT_MSG(src_idx < src.numel(),
                                         std::format("broadcast source index must be in range "
                                                     "(source_index={}, source_numel={}, "
                                                     "output_index={}, output_numel={}, "
                                                     "source_shape={}, target_shape={})",
                                                     src_idx, src.numel(), i, result.numel(),
                                                     src.shape().str(), target.str()));
                    dst_data[i] = src_data[src_idx];
                }
            } else if (src.dtype() == DataType::Float32) {
                const float* src_data = src.ptr<float>();
                float* dst_data = result.ptr<float>();
                for (size_t i = 0; i < result.numel(); ++i) {
                    size_t src_idx = broadcast::index(i, target_dims, src_dims);
                    LFS_DEBUG_ASSERT_MSG(src_idx < src.numel(),
                                         std::format("broadcast source index must be in range "
                                                     "(source_index={}, source_numel={}, "
                                                     "output_index={}, output_numel={}, "
                                                     "source_shape={}, target_shape={})",
                                                     src_idx, src.numel(), i, result.numel(),
                                                     src.shape().str(), target.str()));
                    dst_data[i] = src_data[src_idx];
                }
            } else if (src.dtype() == DataType::Int32) {
                const int* src_data = src.ptr<int>();
                int* dst_data = result.ptr<int>();
                for (size_t i = 0; i < result.numel(); ++i) {
                    size_t src_idx = broadcast::index(i, target_dims, src_dims);
                    LFS_DEBUG_ASSERT_MSG(src_idx < src.numel(),
                                         std::format("broadcast source index must be in range "
                                                     "(source_index={}, source_numel={}, "
                                                     "output_index={}, output_numel={}, "
                                                     "source_shape={}, target_shape={})",
                                                     src_idx, src.numel(), i, result.numel(),
                                                     src.shape().str(), target.str()));
                    dst_data[i] = src_data[src_idx];
                }
            } else {
                LFS_ASSERT_MSG(false,
                               "Unsupported dtype reached CPU broadcast dispatch");
            }
        }

        return result;
    }

} // namespace lfs::core
