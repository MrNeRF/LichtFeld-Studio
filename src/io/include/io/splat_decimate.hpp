/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#include <core/export.hpp>
#include <core/splat_data.hpp>
#include <cstddef>
#include <functional>
#include <io/error.hpp>
#include <string>

namespace lfs::io {
    struct DecimateOptions {
        size_t target_count = 0;
        bool use_gpu = true;
        std::function<bool(float progress, const std::string& stage)> progress;
    };

    // Ignores deleted rows. Output owns CUDA tensors; input is never mutated.
    LFS_IO_API Result<lfs::core::SplatData>
    decimate_splats(const lfs::core::SplatData& input, const DecimateOptions& options);
} // namespace lfs::io
