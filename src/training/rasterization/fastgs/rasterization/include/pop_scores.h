/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once
#include "rasterization_api.h"
namespace fast_lfs::rasterization {
    void accumulate_pop_scores(const ForwardContext& ctx, int width, int height,
                               const float* background, const float* background_image,
                               double* scores, cudaStream_t stream);
}
