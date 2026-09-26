/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#include "core/cuda_types.hpp"
#include "core/tensor/internal/private_access.hpp"
#include "core/tensor_completion.hpp"

namespace lfs::core {
    struct TensorCompletionAccess {
        static TensorCompletion cuda(cudaStream_t stream, VulkanTimelinePoint point = {});
        static TensorCompletion vulkan(uint64_t value);
        // Completes with the Metal batch `serial`; `point` is signaled with it.
        static TensorCompletion metal(uint64_t serial, VulkanTimelinePoint point = {});
        static TensorCompletion external(void* device, VulkanTimelinePoint point);
    };
} // namespace lfs::core
