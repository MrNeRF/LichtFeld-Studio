/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/logger.hpp"
#include "internal/tensor_impl.hpp"
#include "internal/tensor_ops.hpp"
#include <algorithm>
#include <cmath>
#include <functional>
#include <numeric>

#define CHECK_CUDA(call)                                        \
    do {                                                        \
        const cudaError_t error = (call);                       \
        LFS_ASSERT_MSG(error == cudaSuccess,                    \
                       std::string("CUDA operation failed: ") + \
                           cudaGetErrorString(error));          \
    } while (0)

namespace lfs::core {

    // ============= Tensor Static Factory Methods =============

    Tensor Tensor::linspace(float start, float end, size_t steps, Device device) {
        LFS_ASSERT_MSG(steps > 0, "linspace steps must be positive");
        LFS_ASSERT_MSG(device == Device::CPU || device == Device::CUDA,
                       "linspace received an invalid device");
        LFS_ASSERT_MSG(std::isfinite(start) && std::isfinite(end),
                       "linspace endpoints must be finite");

        if (steps == 1) {
            return Tensor::full({1}, start, device);
        }

        auto t = Tensor::empty({steps}, device);

        // Generate on CPU first
        std::vector<float> data(steps);
        float step = (end - start) / (steps - 1);
        for (size_t i = 0; i < steps; ++i) {
            data[i] = start + i * step;
        }

        if (device == Device::CUDA) {
            CHECK_CUDA(cudaMemcpy(t.ptr<float>(), data.data(), steps * sizeof(float),
                                  cudaMemcpyHostToDevice));
        } else {
            std::memcpy(t.ptr<float>(), data.data(), steps * sizeof(float));
        }

        return t;
    }

    Tensor Tensor::diag(const Tensor& diagonal) {
        LFS_ASSERT_MSG(diagonal.is_valid(), "diag requires a valid tensor");
        LFS_ASSERT_MSG(diagonal.ndim() == 1, "diag requires a rank-1 tensor");
        LFS_ASSERT_MSG(diagonal.dtype() == DataType::Float32,
                       "diag currently supports only Float32");

        size_t n = diagonal.numel();
        auto result = Tensor::zeros({n, n}, diagonal.device());
        if (n == 0) {
            return result;
        }

        if (diagonal.device() == Device::CUDA) {
            CHECK_CUDA(cudaGetLastError());
            tensor_ops::launch_diag(diagonal.ptr<float>(), result.ptr<float>(), n, result.stream());
            CHECK_CUDA(cudaGetLastError());
            // No sync - returns tensor
        } else {
            const float* diag_data = diagonal.ptr<float>();
            float* mat_data = result.ptr<float>();
            for (size_t i = 0; i < n; ++i) {
                mat_data[i * n + i] = diag_data[i];
            }
        }

        return result;
    }

} // namespace lfs::core

// ============= MemoryInfo Implementation =============
namespace lfs::core {

    MemoryInfo MemoryInfo::cuda() {
        MemoryInfo info;

        size_t free_bytes, total_bytes;
        CHECK_CUDA(cudaMemGetInfo(&free_bytes, &total_bytes));

        info.free_bytes = free_bytes;
        info.total_bytes = total_bytes;
        info.allocated_bytes = total_bytes - free_bytes;
        info.device_id = 0;

        return info;
    }

    MemoryInfo MemoryInfo::cpu() {
        MemoryInfo info;
        info.free_bytes = 0;
        info.total_bytes = 0;
        info.allocated_bytes = 0;
        info.device_id = -1;
        return info;
    }

    void MemoryInfo::log() const {
        LOG_INFO("Memory Info - Device: {}, Allocated: {:.2f} MB, Free: {:.2f} MB, Total: {:.2f} MB",
                 device_id,
                 allocated_bytes / (1024.0 * 1024.0),
                 free_bytes / (1024.0 * 1024.0),
                 total_bytes / (1024.0 * 1024.0));
    }

} // namespace lfs::core

// ============= Functional Operations Implementation =============
namespace lfs::core::functional {

    Tensor map(const Tensor& input, std::function<float(float)> func) {
        LFS_ASSERT_MSG(input.is_valid(), "functional::map requires a valid tensor");
        LFS_ASSERT_MSG(input.dtype() == DataType::Float32,
                       "functional::map currently supports only Float32");
        LFS_ASSERT_MSG(static_cast<bool>(func), "functional::map requires a callable");
        auto result = Tensor::empty(input.shape(), input.device());

        if (input.device() == Device::CUDA) {
            auto cpu_input = input.to(Device::CPU);
            const float* src = cpu_input.ptr<float>();
            std::vector<float> dst_data(input.numel());

            for (size_t i = 0; i < input.numel(); ++i) {
                dst_data[i] = func(src[i]);
            }

            if (!dst_data.empty()) {
                CHECK_CUDA(cudaMemcpy(result.ptr<float>(), dst_data.data(),
                                      dst_data.size() * sizeof(float), cudaMemcpyHostToDevice));
            }
        } else {
            const float* src = input.ptr<float>();
            float* dst = result.ptr<float>();

            for (size_t i = 0; i < input.numel(); ++i) {
                dst[i] = func(src[i]);
            }
        }

        return result;
    }

    float reduce(const Tensor& input, float init, std::function<float(float, float)> func) {
        LFS_ASSERT_MSG(input.is_valid(), "functional::reduce requires a valid tensor");
        LFS_ASSERT_MSG(input.dtype() == DataType::Float32,
                       "functional::reduce currently supports only Float32");
        LFS_ASSERT_MSG(std::isfinite(init), "functional::reduce initial value must be finite");
        LFS_ASSERT_MSG(static_cast<bool>(func), "functional::reduce requires a callable");
        auto values = input.to_vector();
        float result = init;

        for (float val : values) {
            result = func(result, val);
        }

        return result;
    }

    Tensor filter(const Tensor& input, std::function<bool(float)> predicate) {
        LFS_ASSERT_MSG(input.is_valid(), "functional::filter requires a valid tensor");
        LFS_ASSERT_MSG(input.dtype() == DataType::Float32,
                       "functional::filter currently supports only Float32");
        LFS_ASSERT_MSG(static_cast<bool>(predicate), "functional::filter requires a callable");
        auto result = Tensor::empty(input.shape(), input.device());

        if (input.device() == Device::CUDA) {
            auto cpu_input = input.to(Device::CPU);
            const float* src = cpu_input.ptr<float>();
            std::vector<float> dst_data(input.numel());

            for (size_t i = 0; i < input.numel(); ++i) {
                dst_data[i] = predicate(src[i]) ? 1.0f : 0.0f;
            }

            if (!dst_data.empty()) {
                CHECK_CUDA(cudaMemcpy(result.ptr<float>(), dst_data.data(),
                                      dst_data.size() * sizeof(float), cudaMemcpyHostToDevice));
            }
        } else {
            const float* src = input.ptr<float>();
            float* dst = result.ptr<float>();

            for (size_t i = 0; i < input.numel(); ++i) {
                dst[i] = predicate(src[i]) ? 1.0f : 0.0f;
            }
        }

        return result;
    }

} // namespace lfs::core::functional

#undef CHECK_CUDA
