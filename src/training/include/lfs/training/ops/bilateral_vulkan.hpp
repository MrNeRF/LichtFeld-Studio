/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "lfs/training/ops/bilateral.hpp"

namespace lfs::training {
    const gpu_ops::BilateralOps& vulkan_bilateral_ops();
}
