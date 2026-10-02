/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "lfs/training/ops/registry.hpp"

namespace lfs::training {

    const TrainingOps& cuda_training_ops_table() {
        static const TrainingOps table{.backend = core::GpuBackend::CUDA};
        return table;
    }

} // namespace lfs::training
