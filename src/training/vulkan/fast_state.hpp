/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#include "core/gpu_elapsed.hpp"
#include "core/tensor_readback.hpp"
#include "dispatch.hpp"
#include "lfs/training/ops/raster.hpp"
#include <array>
#include <vector>
namespace lfs::training::vulkan {
    struct FastPush {
        uint64_t means, scales, rotations, opacities, sh0, sh, view, camera;
        uint64_t background, background_image, screen_share;
        uint64_t projected, visibility, offsets, original_to_work, work_to_original;
        uint64_t counts, keys, values, ranges, image, alpha, depth, normal, transmittance, last;
        uint64_t status;
        uint32_t count, visible, width, height, grid_width, grid_height, active_bases, rest;
        uint32_t depth_bits, mip, render_normal, render_depth, stage, instances;
        float fx, fy, cx, cy, clip_left, clip_right, clip_top, clip_bottom;
        uint64_t sh_bounds;
        uint32_t sh_format, padding;
    };
    inline uint32_t specialization(const FastPush& p, uint32_t stage) {
        return stage | (p.active_bases << 4) | (p.rest << 9) | (p.sh_format << 13) |
               (p.render_normal << 15) | (p.render_depth << 16) | (p.mip << 17);
    }
    static_assert(sizeof(FastPush) == 320);
    static_assert(offsetof(FastPush, count) == 216);
    static_assert(offsetof(FastPush, fx) == 272);
    // Dense and compact projection rows share this layout with fast_common.slangh.
    struct Projected {
        float mean[2], conic[3], opacity, color[3], depth, normal[3];
        uint32_t bounds[4], depth_key;
    };
    static_assert(sizeof(Projected) == 72);
    static_assert(offsetof(Projected, bounds) == 52);
    struct FastState : gpu_ops::BackendState {
        core::GpuElapsed* timer = nullptr;
        core::TensorReadback scalar_readback;
        void mark(size_t event) {
            if (timer)
                (void)timer->mark(event, core::TensorExecutionTarget::current());
        }
        std::array<core::Tensor, 27> scratch;
        core::Tensor temporary(size_t slot, size_t count,
                               core::DataType dtype = core::DataType::Float32, bool clear = false) {
            if (count == 0)
                return core::Tensor::empty({0}, core::Device::GPU, dtype);
            auto& buffer = scratch.at(slot);
            if (!buffer.is_valid() || buffer.dtype() != dtype || buffer.numel() < count) {
                const size_t capacity = (count + (count + 3) / 4 + 255) & ~size_t{255};
                buffer = core::Tensor::empty({capacity}, core::Device::GPU, dtype);
            }
            auto result = buffer.slice(0, 0, count);
            if (clear)
                result.zero_();
            return result;
        }
        FastPush push{};
        std::vector<core::Tensor> inputs;
        core::Tensor projected, visibility, offsets, original_to_work, work_to_original;
        core::Tensor counts, keys_a, keys_b, values_a, values_b, ranges, transmittance, last;
        core::Tensor image, alpha, depth, normal;
        bool live = false;
        std::string message;
    };
    gpu_ops::RasterResult fast_forward(gpu_ops::FastSaved&, const gpu_ops::SplatInputs&,
                                       gpu_ops::In view, gpu_ops::In camera, gpu_ops::In background, gpu_ops::In background_image,
                                       const gpu_ops::FastParams&, const gpu_ops::RenderOutputs&, gpu_ops::Out screen_share);
    void fast_backward(gpu_ops::FastSaved&, const gpu_ops::RenderGradients&, gpu_ops::Out densification,
                       gpu_ops::In error, gpu_ops::In edge, gpu_ops::Out scores, const gpu_ops::BackwardAdam&, DensificationType);
    std::vector<StorageRef> fast_storage(const FastState&);
} // namespace lfs::training::vulkan
