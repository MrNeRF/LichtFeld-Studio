/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once

#include "core/export.hpp"
#include "core/gpu_backend_fwd.hpp"
#include "core/tensor_fwd.hpp"
#include <cstdint>
#include <memory>
#include <span>

namespace lfs::core {
    struct TensorCompletionAccess;

    struct VulkanTimelinePoint {
        void* semaphore = nullptr;
        uint64_t value = 0;
        std::shared_ptr<void> keep_alive;
    };

    // Tensor captures settle only their pending work. Backend obligations cover
    // all work through wait(), including work submitted after include().
    class LFS_CORE_API TensorCompletion {
    public:
        TensorCompletion() = default;
        explicit TensorCompletion(std::span<const Tensor* const> tensors);
        ~TensorCompletion();
        void include(GpuBackend backend);
        void include(const Tensor& tensor);
        // Before decoding a GPU payload, honor the calling thread's scope.
        void include_current_gpu();
        // Backend-wide obligations require wait(); ready() never drains a device.
        [[nodiscard]] bool ready() const;
        void wait() const;
        [[nodiscard]] VulkanTimelinePoint timeline() const;

    private:
        struct Impl;
        std::shared_ptr<Impl> impl_;
        friend struct TensorCompletionAccess;
    };

} // namespace lfs::core
