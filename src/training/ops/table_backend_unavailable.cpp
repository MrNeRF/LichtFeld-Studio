/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "lfs/training/ops/registry.hpp"

namespace lfs::training {
    const TrainingOps& vulkan_training_ops_table() {
        static const TrainingOps unavailable{.backend = core::GpuBackend::Vulkan};
        return unavailable;
    }
} // namespace lfs::training
