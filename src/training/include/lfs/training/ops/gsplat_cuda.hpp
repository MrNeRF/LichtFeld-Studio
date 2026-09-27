/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#include "lfs/training/ops/gsplat_services.hpp"
namespace lfs::training {
    const lfs::gpu_ops::GsplatRasterOps& cuda_gsplat_ops();
} // namespace lfs::training
