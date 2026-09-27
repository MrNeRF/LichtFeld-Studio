/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "lfs/training/ops/extra_loss.hpp"

namespace lfs::training {
    const lfs::gpu_ops::ExtraLossOps& cuda_extra_loss_ops();
} // namespace lfs::training
