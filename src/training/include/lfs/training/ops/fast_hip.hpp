/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#include "lfs/training/ops/raster.hpp"
namespace lfs::training {
    const gpu_ops::FastRasterOps& hip_fast_ops();
}
