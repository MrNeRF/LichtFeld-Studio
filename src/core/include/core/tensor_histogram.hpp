/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once

#include "core/export.hpp"

namespace lfs::core {
    class Tensor;

    // Count nonzero UInt8/Bool values into their byte-value bins. Overwrites a
    // contiguous Int32 vector with at least 256 entries; extra bins stay zero.
    // Bin zero stays empty. Input may be strided, but may not alias
    // counts. Both tensors must use the same device and storage backend.
    // At most INT32_MAX input elements are supported. GPU work is asynchronous.
    LFS_CORE_API void histogram_u8(const Tensor& values, Tensor& counts);
} // namespace lfs::core
