/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once

#include <cstddef>
#include <cstdint>
#include <cuda_runtime_api.h>

namespace lfs::core::detail {
    void gather_preview_sh(const void* source, const float* bounds, bool half,
                           const int64_t* rows, size_t count, uint32_t layout_rest,
                           uint32_t selected_rest, float* output, cudaStream_t stream);
}
