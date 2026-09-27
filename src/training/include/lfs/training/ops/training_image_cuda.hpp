/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "lfs/training/ops/training_image.hpp"

namespace lfs::training {
    const lfs::gpu_ops::TrainingImageOps& cuda_training_image_ops();
} // namespace lfs::training
