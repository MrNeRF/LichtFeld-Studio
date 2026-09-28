/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "metal_families.hpp"

namespace lfs::training {

    const TrainingOps& metal_training_ops_table() {
        static const TrainingOps table{
            .backend = core::GpuBackend::Metal,
            .training_image = &metal_training_image_ops(),
            .session = &metal_session_ops(),
        };
        return table;
    }

} // namespace lfs::training
