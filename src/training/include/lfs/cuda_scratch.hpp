/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/checked_arithmetic.hpp"
#include "core/cuda_allocation.hpp"
#include "diagnostics/vram_profiler.hpp"

#include <cuda_runtime.h>
#include <string_view>

namespace lfs::training::cuda_scratch {

    inline size_t checked_bytes(const size_t count,
                                const size_t element_size,
                                const std::string_view allocation) {
        return lfs::core::checked_product(count, element_size, allocation);
    }

    struct VramProfilerAllocationHooks {
        void before_allocate(std::string_view) const noexcept {}

        void after_allocate(void* ptr,
                            const size_t bytes,
                            const std::string_view label) const noexcept {
#if CUDART_VERSION >= 11020
            constexpr auto method = diagnostics::VramAllocationMethod::Async;
#else
            constexpr auto method = diagnostics::VramAllocationMethod::Direct;
#endif
            try {
                diagnostics::VramProfiler::instance().recordAllocation(ptr, bytes, method, label);
            } catch (...) {
            }
        }

        void before_deallocate(void* ptr) const noexcept {
            try {
                diagnostics::VramProfiler::instance().recordDeallocation(ptr);
            } catch (...) {
            }
        }
    };

    struct TrainingCubWorkspaceTraits {
        static constexpr std::string_view allocation_label = "training.cub_workspace";
        static constexpr std::string_view diagnostic_scope = "training";
    };

    using DeviceBuffer = lfs::core::UniqueCudaAllocation<
        lfs::core::StreamOrderedCudaAllocator, VramProfilerAllocationHooks>;
    using CubWorkspace = lfs::core::CudaCubWorkspace<DeviceBuffer, TrainingCubWorkspaceTraits>;

} // namespace lfs::training::cuda_scratch
