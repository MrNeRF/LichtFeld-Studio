/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "target_lifetime.hpp"
namespace lfs::core::internal {
    void observe_native_target(cudaStream_t) noexcept {}
} // namespace lfs::core::internal
namespace lfs::core {
    bool is_stream_retired(cudaStream_t) noexcept { return false; }
} // namespace lfs::core
