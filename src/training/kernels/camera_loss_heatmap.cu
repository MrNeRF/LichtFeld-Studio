/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "camera_loss_heatmap.cuh"

namespace lfs::training::kernels {

    namespace {
        __global__ void update_camera_loss_heatmap_kernel(
            const float* __restrict__ loss_scalar,
            const int camera_slot,
            const float ema_alpha,
            float* __restrict__ latest_losses,
            float* __restrict__ ema_losses,
            const std::size_t slot_count) {

            if (camera_slot < 0 || static_cast<std::size_t>(camera_slot) >= slot_count) {
                return;
            }

            const float loss = loss_scalar[0];
            latest_losses[camera_slot] = loss;

            const float prev = ema_losses[camera_slot];
            ema_losses[camera_slot] = prev >= 0.0f
                                          ? prev + ema_alpha * (loss - prev)
                                          : loss;
        }
    } // namespace

    void launch_update_camera_loss_heatmap(
        const float* loss_scalar,
        const int camera_slot,
        const float ema_alpha,
        float* latest_losses,
        float* ema_losses,
        const std::size_t slot_count,
        cudaStream_t stream) {

        update_camera_loss_heatmap_kernel<<<1, 1, 0, stream>>>(
            loss_scalar, camera_slot, ema_alpha, latest_losses, ema_losses, slot_count);
    }

} // namespace lfs::training::kernels
