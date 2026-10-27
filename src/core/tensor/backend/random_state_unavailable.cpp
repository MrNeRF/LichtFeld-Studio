/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "core/tensor.hpp"
#include "random_state.hpp"
namespace lfs::core::internal {
    std::unique_ptr<NativeRandomState> make_native_random_state(uint64_t, const char* message) { throw TensorError(message); }
} // namespace lfs::core::internal
