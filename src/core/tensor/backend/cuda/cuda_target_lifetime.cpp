/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "../target_lifetime.hpp"
#include "runtime/stream_lifetime.hpp"
namespace lfs::core::internal {
    void observe_native_target(cudaStream_t target) noexcept { unretire_stream(target); }
} // namespace lfs::core::internal
