/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "lfs/training/ops/registry.hpp"

namespace lfs::training {

    const TrainingOps& metal_training_ops_table() {
        static const TrainingOps table{.backend = core::GpuBackend::Metal};
        return table;
    }

} // namespace lfs::training
