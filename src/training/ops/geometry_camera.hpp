/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#include "lfs/training/ops/geometry.hpp"
namespace lfs::training {
    std::vector<gpu_ops::AnchorSample> collect_camera_anchor_samples(gpu_ops::In points, gpu_ops::In view, gpu_ops::In prior, const gpu_ops::AnchorParams& params);
}
