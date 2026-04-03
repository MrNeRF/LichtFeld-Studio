/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <cstddef>
#include <cuda_runtime.h>

namespace lfs::training::kernels {

    void launch_update_camera_loss_heatmap(
        const float* loss_scalar,
        int camera_slot,
        float ema_alpha,
        float* latest_losses,
        float* ema_losses,
        std::size_t slot_count,
        cudaStream_t stream);

} // namespace lfs::training::kernels
