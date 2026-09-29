/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "lfs/training/ops/adam.hpp"

namespace lfs::training {
    // Fast accepts 8- and 16-bit ordinary moments; standalone rows stay 16-bit.
    void vulkan_fused_adam_rows(const gpu_ops::JointStep&, const gpu_ops::AdamMasks&,
                                const gpu_ops::AdamHyper&, const gpu_ops::AdamModifiers&);
    const lfs::gpu_ops::AdamOps& vulkan_adam_ops();
} // namespace lfs::training
