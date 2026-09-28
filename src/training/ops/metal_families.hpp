/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once

#include "lfs/training/ops/registry.hpp"

// The Metal ops table's families, one accessor per *_metal.cpp.
namespace lfs::training {
    const lfs::gpu_ops::SessionOps& metal_session_ops();
    const lfs::gpu_ops::TrainingImageOps& metal_training_image_ops();
} // namespace lfs::training
