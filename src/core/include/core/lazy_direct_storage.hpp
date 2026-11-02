/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once

#include "core/export.hpp"

#include <cstddef>

namespace lfs::core {

    // Device bytes that zeros_direct storages have committed past the rows
    // their tensors have grown into (see Tensor::commit_rows): memory already
    // held for the next rows, so growth planning can count it as free.
    [[nodiscard]] LFS_CORE_API std::size_t lazy_direct_storage_ahead_bytes() noexcept;

    // Hands those bytes back to the device, keeping the grown rows; growth
    // maps them again. Returns the bytes released.
    LFS_CORE_API std::size_t release_lazy_direct_storage_ahead();

} // namespace lfs::core
