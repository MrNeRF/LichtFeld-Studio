/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "internal/tensor_broadcast.hpp"
#include "internal/tensor_impl.hpp"
#include "internal/tensor_ops.hpp"

namespace lfs::core {

    Tensor broadcast_to(const Tensor& src, const TensorShape& target) {

        if (!src.is_valid()) {
            LOG_ERROR("Cannot broadcast invalid tensor");
            return Tensor();
        }

        if (src.numel() == 0 || target.elements() == 0) {
            return Tensor::empty(target, src.device(), src.dtype());
        }

        // Check if shapes are compatible for broadcasting
        auto src_dims = src.shape().dims();
        auto target_dims = target.dims();

        // Validate broadcasting rules
        auto broadcast_shape = broadcast::shape(src_dims, target_dims);
        if (broadcast_shape.empty()) {
            LOG_ERROR("Cannot broadcast shape {} to {}", src.shape().str(), target.str());
            return Tensor();
        }

        if (broadcast_shape != target_dims) {
            LOG_ERROR("Broadcast shape mismatch: expected {}, got {}",
                      target.str(), TensorShape(broadcast_shape).str());
            return Tensor();
        }

        // If shapes match, just clone
        if (src.shape() == target) {
            return src.clone();
        }

        auto result = Tensor::empty(target, src.device(), src.dtype());

        // Dispatch based on device and dtype
        if (src.device() == Device::CUDA) {
            if (src.dtype() == DataType::Bool) {
                tensor_ops::launch_broadcast_bool(
                    src.ptr<unsigned char>(), result.ptr<unsigned char>(),
                    src_dims.data(), target_dims.data(),
                    src_dims.size(), target_dims.size(),
                    result.numel(), 0);
                cudaDeviceSynchronize();
            } else if (src.dtype() == DataType::Float32) {
                tensor_ops::launch_broadcast(
                    src.ptr<float>(), result.ptr<float>(),
                    src_dims.data(), target_dims.data(),
                    src_dims.size(), target_dims.size(),
                    result.numel(), 0);
                cudaDeviceSynchronize();
            } else {
                LOG_ERROR("Unsupported dtype for CUDA broadcasting: {}", dtype_name(src.dtype()));
                return Tensor();
            }
        } else {
            // CPU fallback using the index helper
            if (src.dtype() == DataType::Bool) {
                const unsigned char* src_data = src.ptr<unsigned char>();
                unsigned char* dst_data = result.ptr<unsigned char>();
                for (size_t i = 0; i < result.numel(); ++i) {
                    size_t src_idx = broadcast::index(i, target_dims, src_dims);
                    dst_data[i] = src_data[src_idx];
                }
            } else if (src.dtype() == DataType::Float32) {
                const float* src_data = src.ptr<float>();
                float* dst_data = result.ptr<float>();
                for (size_t i = 0; i < result.numel(); ++i) {
                    size_t src_idx = broadcast::index(i, target_dims, src_dims);
                    dst_data[i] = src_data[src_idx];
                }
            } else if (src.dtype() == DataType::Int32) {
                const int* src_data = src.ptr<int>();
                int* dst_data = result.ptr<int>();
                for (size_t i = 0; i < result.numel(); ++i) {
                    size_t src_idx = broadcast::index(i, target_dims, src_dims);
                    dst_data[i] = src_data[src_idx];
                }
            } else {
                LOG_ERROR("Unsupported dtype for CPU broadcasting: {}", dtype_name(src.dtype()));
                return Tensor();
            }
        }

        return result;
    }

} // namespace lfs::core