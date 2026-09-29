/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#include "core/gpu_elapsed.hpp"
#include "lfs/training/ops/raster.hpp"
namespace lfs::training {
    // Borrowed 16-event timer. Wait for recorded work before destroying it.
    // Pairs 0/1 and 2/3: projection; 4/5 and 6/7: binning/sort;
    // 8/9: raster forward; 10/11: raster backward; 12/13 and 14/15: projection backward/Adam and its gradient initialization.
    void vulkan_fast_set_timer(gpu_ops::FastSaved&, core::GpuElapsed*);
    const gpu_ops::FastRasterOps& vulkan_fast_ops();
} // namespace lfs::training
