/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "lfs/training/ops/loss.hpp"

#include <cstddef>

namespace lfs::training {

    const lfs::gpu_ops::PhotometricOps& cuda_photometric_ops();

    struct PhotoWorkspaceBytes {
        size_t required = 0;
        size_t allocated = 0;
        size_t error_map = 0;
    };

    // Arena shrink and reset at the trainer's named memory boundaries.
    PhotoWorkspaceBytes photo_workspace_bytes(const lfs::gpu_ops::PhotoSaved& saved);
    void photo_shrink_to_required(lfs::gpu_ops::PhotoSaved& saved);
    void photo_reset(lfs::gpu_ops::PhotoSaved& saved);

} // namespace lfs::training
