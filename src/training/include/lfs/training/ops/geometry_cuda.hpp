/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "lfs/training/ops/geometry.hpp"

namespace lfs::training {
    const gpu_ops::GeometryLossOps& cuda_geometry_ops();
} // namespace lfs::training
