/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#include "core/tensor/internal/private_access.hpp"
#include "core/tensor_ppisp.hpp"
#include <cuda_runtime.h>
namespace lfs::core::tensor_ops {
    void launch_ppisp_apply(const float* input, float* output, int width, int height, const PpispParams& p, cudaStream_t stream);
}
