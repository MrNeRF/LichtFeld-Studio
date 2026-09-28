/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "lfs/training/ops/registry.hpp"

#include "lfs/training/ops/session_metal.hpp"

namespace lfs::training {

    const TrainingOps& metal_training_ops_table() {
        static const TrainingOps table{
            .backend = core::GpuBackend::Metal,
            .session = &metal_session_ops(),
        };
        return table;
    }

} // namespace lfs::training
