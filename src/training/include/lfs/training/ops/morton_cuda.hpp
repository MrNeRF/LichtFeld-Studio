/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "lfs/training/ops/morton.hpp"

namespace lfs::training {
    const lfs::gpu_ops::MortonOps& cuda_morton_ops();
} // namespace lfs::training
