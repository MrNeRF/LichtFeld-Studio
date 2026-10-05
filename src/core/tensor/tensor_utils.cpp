/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/logger.hpp"
#include "core/tensor_cuda_interop.hpp"
#include "internal/tensor_impl.hpp"
#include <algorithm>
#include <cmath>
#include <numeric>

namespace lfs::core {

    // ============= Tensor Static Factory Methods =============

    Tensor Tensor::linspace(float start, float end, size_t steps, Device device) {
        LFS_ASSERT_MSG(steps > 0,
                       "linspace steps must be positive");
        LFS_ASSERT_MSG(device == Device::CPU || device == Device::GPU,
                       "linspace received an invalid device");
        LFS_ASSERT_MSG(std::isfinite(start) && std::isfinite(end),
                       "linspace endpoints must be finite");

        if (steps == 1) {
            return Tensor::full({1}, start, device);
        }

        auto t = Tensor::empty({steps}, device);

        // Generate on CPU first
        std::vector<float> data(steps);
        const float step = (end - start) / static_cast<float>(steps - 1);
        for (size_t i = 0; i < steps; ++i) {
            data[i] = i < steps / 2
                          ? start + step * static_cast<float>(i)
                          : end - step * static_cast<float>(steps - i - 1);
        }

        if (device == Device::GPU) {
            internal::backend_ops_for(t).copy_host_to_device(internal::CopyRequest{
                .src = internal::raw_storage_ref(data.data(), DataType::Float32),
                .dst = internal::storage_ref(t),
                .bytes = steps * sizeof(float),
                .synchronous = true,
                .context = internal::ExecContext{nullptr},
            });
        } else {
            std::memcpy(t.ptr<float>(), data.data(), steps * sizeof(float));
        }

        return t;
    }

} // namespace lfs::core

// ============= MemoryInfo Implementation =============
namespace lfs::core {

    MemoryInfo MemoryInfo::cuda() {
        return internal::backend_ops(GpuBackend::CUDA).stats();
    }

} // namespace lfs::core
