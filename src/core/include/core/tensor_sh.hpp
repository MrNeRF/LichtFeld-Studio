/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#include "core/export.hpp"
#include "core/tensor_fwd.hpp"
#include <cstddef>
#include <cstdint>

namespace lfs::core {
    enum class ShFormat : uint32_t { Canonical,
                                     Float32,
                                     Float16,
                                     Q16 };

    struct ShCodec {
        ShFormat source_format = ShFormat::Float32;
        ShFormat destination_format = ShFormat::Float32;
        size_t source_rows = 0, destination_rows = 0, count = 0;
        uint32_t source_rest = 0, destination_rest = 0;
        size_t source_offset = 0, destination_offset = 0;
        bool scatter = false;
    };

    // Convert/copy a row range, or gather/scatter indexed rows. Canonical is
    // Float32 [N,K,3]; resident buffers are flat. Q16 uses a Float16 container
    // for raw u16 codes and Float32 [ceil(N/256)*2] bounds.
    // Indices must be in range; scatter indices must be unique. Aliased inputs
    // are snapshotted before writes. Untouched destination rows are preserved.
    // Q16 output requires a complete destination (no scatter or offset).
    LFS_CORE_API void sh_codec(const Tensor& source, Tensor& destination, const ShCodec& codec,
                               const Tensor* indices = nullptr, const Tensor* source_bounds = nullptr,
                               Tensor* destination_bounds = nullptr);
    LFS_CORE_API ShFormat sh_storage_format(const Tensor& values, const Tensor& bounds);
} // namespace lfs::core
