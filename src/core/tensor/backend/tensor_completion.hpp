/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#include "core/tensor/internal/private_access.hpp"
#include "core/tensor_completion.hpp"
#include <cuda_runtime.h>

namespace lfs::core {
    struct TensorCompletionAccess {
        static TensorCompletion cuda(cudaStream_t stream, VulkanTimelinePoint point = {});
        static TensorCompletion vulkan(uint64_t value);
        static TensorCompletion external(void* device, VulkanTimelinePoint point);
    };
} // namespace lfs::core
