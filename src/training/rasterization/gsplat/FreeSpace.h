/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once

#include <cstdint>
#include <cuda_runtime.h>

namespace gsplat_lfs {

    // Free space in front of a laser scan. T_front is the transmittance of the
    // splats a ray hits nearer than its scan depth less a margin; 1 where the
    // pixel has no scan depth. Off while gt_depth is null.
    struct FreeSpaceRender {
        const float* gt_depth = nullptr;  // [C, gt_h, gt_w] scan ray depth in file units, 0 = none
        float* T_front = nullptr;         // [C, H, W], written by the forward
        const float* v_T_front = nullptr; // [C, H, W], backward only
        uint32_t gt_w = 0;
        uint32_t gt_h = 0;
        uint32_t depth_channel = 3; // render channel holding the splat ray depth
        float unit = 1.0f;          // scene units per file unit
        float margin = 0.0f;        // scene units
        float margin_rel = 0.0f;    // fraction of the scan depth

        [[nodiscard]] bool enabled() const { return gt_depth != nullptr && T_front != nullptr; }

#ifdef __CUDACC__
        // Scan depth at the nearest GT pixel, in scene units; 0 where there is none.
        __device__ __forceinline__ float scan_depth(const uint32_t cid, const uint32_t x, const uint32_t y,
                                                    const uint32_t W, const uint32_t H) const {
            const uint32_t xd = min(gt_w - 1, static_cast<uint32_t>((x + 0.5f) * gt_w / W));
            const uint32_t yd = min(gt_h - 1, static_cast<uint32_t>((y + 0.5f) * gt_h / H));
            return gt_depth[(static_cast<size_t>(cid) * gt_h + yd) * gt_w + xd] * unit;
        }

        // Splats nearer than this count toward T_front; 0 where there is no scan.
        __device__ __forceinline__ float limit(const uint32_t cid, const uint32_t x, const uint32_t y,
                                               const uint32_t W, const uint32_t H) const {
            const float d = scan_depth(cid, x, y, W, H);
            return d > 0.0f ? d - fmaxf(margin, margin_rel * d) : 0.0f;
        }
#endif
    };

    // loss = weight * mean(1 - T_front) over pixels with scan depth.
    // Writes loss[0] and v_T_front; partials holds free_space_partial_count floats.
    [[nodiscard]] uint32_t free_space_partial_count(uint32_t pixels);
    void free_space_loss(const FreeSpaceRender& free_space, uint32_t width, uint32_t height,
                         float weight, float* v_T_front, float* loss, float* partials,
                         cudaStream_t stream);

} // namespace gsplat_lfs
