/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "internal/tensor_impl.hpp"
#include "internal/tensor_ops.hpp"
#include <algorithm>
#include <cmath>
#include <numeric>

#define CHECK_CUDA(call)                              \
    do {                                              \
        cudaError_t error = call;                     \
        if (error != cudaSuccess) {                   \
            LOG_ERROR("CUDA error at {}:{} - {}: {}", \
                      __FILE__, __LINE__,             \
                      cudaGetErrorName(error),        \
                      cudaGetErrorString(error));     \
        }                                             \
    } while (0)

namespace lfs::core {

    // ============= Tensor Static Factory Methods =============

    Tensor Tensor::linspace(float start, float end, size_t steps, Device device) {
        if (steps == 0) {
            LOG_ERROR("Steps must be > 0");
            return Tensor();
        }

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
        if (diagonal.ndim() != 1) {
            LOG_ERROR("diag requires 1D tensor");
            return Tensor();
        }

        size_t n = diagonal.numel();
        auto result = Tensor::zeros({n, n}, diagonal.device());

        if (diagonal.device() == Device::CUDA) {
            tensor_ops::launch_diag(diagonal.ptr<float>(), result.ptr<float>(), n, 0);
            CHECK_CUDA(cudaDeviceSynchronize());
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
        cudaMemGetInfo(&free_bytes, &total_bytes);

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
        auto result = Tensor::empty(input.shape(), input.device());

        if (input.device() == Device::CUDA) {
            auto cpu_input = input.to(Device::CPU);
            const float* src = cpu_input.ptr<float>();
            std::vector<float> dst_data(input.numel());

            for (size_t i = 0; i < input.numel(); ++i) {
                dst_data[i] = func(src[i]);
            }

            cudaMemcpy(result.ptr<float>(), dst_data.data(),
                       dst_data.size() * sizeof(float), cudaMemcpyHostToDevice);
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
        auto values = input.to_vector();
        float result = init;

        for (float val : values) {
            result = func(result, val);
        }

        return result;
    }

    Tensor filter(const Tensor& input, std::function<bool(float)> predicate) {
        auto result = Tensor::empty(input.shape(), input.device());

        if (input.device() == Device::CUDA) {
            auto cpu_input = input.to(Device::CPU);
            const float* src = cpu_input.ptr<float>();
            std::vector<float> dst_data(input.numel());

            for (size_t i = 0; i < input.numel(); ++i) {
                dst_data[i] = predicate(src[i]) ? 1.0f : 0.0f;
            }

            cudaMemcpy(result.ptr<float>(), dst_data.data(),
                       dst_data.size() * sizeof(float), cudaMemcpyHostToDevice);
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