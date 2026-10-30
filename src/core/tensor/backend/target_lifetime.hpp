/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#include "core/tensor_cuda_interop.hpp"
namespace lfs::core::internal {
    // Compatibility observation for caller-owned stream handles. Implemented
    // beside the native lifetime registry, or as a no-op without that SDK.
    void observe_native_target(cudaStream_t target) noexcept;
} // namespace lfs::core::internal
