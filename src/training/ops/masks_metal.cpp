/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

// Metal MaskOps; see ops/masks_cuda.cpp for the reference behaviour.

#include "metal_families.hpp"
#include "metal_kernels.hpp"

#include "core/assert.hpp"
#include "lfs/training/ops/masks.hpp"
#include "mesh_mask_camera.hpp"

#include <algorithm>
#include <cstddef>
#include <format>

namespace lfs::training {
    namespace {
        using namespace lfs::gpu_ops;
        namespace mk = metal;

        constexpr uint32_t kMaxGroups = 1024;

        struct MaskParams {
            uint64_t alpha, mask, roi, out, partials;
            uint32_t count, mask_bytes, has_roi, mode;
            float power, grad_scale, keep_min, segment_min;
        };

        struct ReduceParams {
            uint64_t partials, result, mask_sum;
            uint32_t count, masked;
            float channels, scale, divisor, offset;
        };

        struct MeshCoverageParams {
            uint64_t vertices, indices, samples, large_faces, counters, mask;
            int32_t vertex_count, face_count;
            float z_near;
            int32_t padding;
            MeshMaskCameraBlock camera;
        };
        static_assert(offsetof(MeshCoverageParams, camera) == 64);

        bool has_roi(In roi) { return roi.is_valid() && roi.numel() > 0; }

        MaskParams params_for(In alpha, In mask, In roi, Out out, Out partials, const uint32_t mode) {
            LFS_ASSERT_MSG(mask.ndim() == 2, std::format("mask ops expect a 2D mask (shape={})", mask.shape().str()));
            const size_t count = mask.shape()[0] * mask.shape()[1];
            LFS_ASSERT_MSG(count <= UINT32_MAX, std::format("mask ops support at most 2^32-1 pixels (count={})", count));
            const bool bytes = mask.dtype() == core::DataType::UInt8 || mask.dtype() == core::DataType::Bool;
            LFS_ASSERT_MSG(bytes || mask.dtype() == core::DataType::Float32,
                           std::format("mask dtype must be Float32, UInt8 or Bool (dtype={})", static_cast<int>(mask.dtype())));
            return {.alpha = mk::address(alpha),
                    .mask = mk::address(mask),
                    .roi = has_roi(roi) ? mk::address(roi) : 0,
                    .out = mk::address(out),
                    .partials = mk::address(partials),
                    .count = static_cast<uint32_t>(count),
                    .mask_bytes = bytes ? 1u : 0u,
                    .has_roi = has_roi(roi) ? 1u : 0u,
                    .mode = mode,
                    .keep_min = kernels::kMaskKeepMin,
                    .segment_min = kernels::kMaskSegmentMin};
        }

        uint32_t reduce_groups(const uint32_t count) { return std::min<uint32_t>((count + 255) / 256, kMaxGroups); }

        // loss = scale * (sum / count)
        void finish(Out partials, const uint32_t groups, Out loss, const float scale, const uint32_t count) {
            const ReduceParams params{mk::address(partials), mk::address(loss), 0, groups, 0, 0.f, scale,
                                      static_cast<float>(count), 0.f};
            mk::launch("loss_reduce_final", params, {&partials, &loss}, 1, kMaxGroups);
        }

        void photometric_weight(In mask, In roi, Out weight, MaskPhotoMode mode) {
            const Tensor none;
            Tensor unused;
            const MaskParams params = params_for(none, mask, roi, weight, unused, static_cast<uint32_t>(mode));
            if (params.count == 0)
                return;
            mk::launch_items("mask_photometric_weight", params, {&mask, &roi, &weight}, params.count);
        }

        void opacity_penalty(In alpha, In mask, In roi, Out grad_alpha, Out reduction_temp, Out loss,
                             MaskOpacityMode mode, float power, float scale) {
            MaskParams params = params_for(alpha, mask, roi, grad_alpha, reduction_temp, static_cast<uint32_t>(mode));
            if (params.count == 0)
                return;
            if (scale == 0.0f) {
                // Still zero grads for a clean steady state.
                grad_alpha.zero_();
                loss.zero_();
                return;
            }
            const uint32_t groups = reduce_groups(params.count);
            params.power = power;
            params.grad_scale = scale / static_cast<float>(params.count);
            mk::launch("mask_opacity_penalty", params, {&alpha, &mask, &roi, &grad_alpha, &reduction_temp}, groups);
            finish(reduction_temp, groups, loss, scale, params.count);
        }

        void alpha_consistency(In alpha, In mask, In roi, Out grad_alpha, Out reduction_temp, Out loss, float weight) {
            MaskParams params = params_for(alpha, mask, roi, grad_alpha, reduction_temp, 0);
            if (params.count == 0)
                return;
            const uint32_t groups = reduce_groups(params.count);
            params.grad_scale = weight / static_cast<float>(params.count);
            mk::launch("mask_alpha_consistency", params, {&alpha, &mask, &roi, &grad_alpha, &reduction_temp}, groups);
            finish(reduction_temp, groups, loss, weight, params.count);
        }

        core::Tensor mesh_coverage(In vertices, In indices, const MeshMaskCamera& camera, In samples,
                                   const core::UndistortParams* distortion, const float z_near) {
            const core::Tensor* const sample_map = distortion ? &samples : nullptr;
            validate_mesh_coverage(vertices, indices, camera, sample_map, distortion, z_near);
            auto mask = core::Tensor::zeros({static_cast<size_t>(camera.height), static_cast<size_t>(camera.width)},
                                            core::Device::GPU, core::DataType::UInt8);
            const auto face_count = static_cast<uint32_t>(indices.shape()[0]);
            if (face_count == 0)
                return mask;
            const auto input_vertices = vertices.contiguous();
            const auto input_indices = indices.contiguous();
            const auto input_samples = sample_map ? sample_map->contiguous() : core::Tensor{};
            auto large_faces = core::Tensor::empty({face_count}, core::Device::GPU, core::DataType::Int32);
            auto counters = core::Tensor::zeros({size_t{2}}, core::Device::GPU, core::DataType::Int32);
            const MeshCoverageParams params{
                .vertices = mk::address(input_vertices),
                .indices = mk::address(input_indices),
                .samples = sample_map ? mk::address(input_samples) : 0,
                .large_faces = mk::address(large_faces),
                .counters = mk::address(counters),
                .mask = mk::address(mask),
                .vertex_count = static_cast<int32_t>(input_vertices.shape()[0]),
                .face_count = static_cast<int32_t>(face_count),
                .z_near = z_near,
                .padding = 0,
                .camera = pack_mesh_mask_camera(camera, sample_map, distortion)};
            const uint32_t prepare_groups =
                std::min(kMeshMaskMaxPrepareGroups,
                         (face_count + kMeshMaskThreads - 1) / kMeshMaskThreads);
            mk::launch("mesh_coverage_prepare", params,
                       {&input_vertices, &input_indices, &input_samples, &large_faces, &counters, &mask},
                       prepare_groups, kMeshMaskThreads);
            mk::launch("mesh_coverage_large", params,
                       {&input_vertices, &input_indices, &input_samples, &large_faces, &counters, &mask},
                       std::min(kMeshMaskMaxLargeGroups, face_count), kMeshMaskThreads);
            return mask;
        }
        core::Tensor point_coverage(In means, const MeshMaskCamera& camera, int radius,
                                    const core::UndistortParams* distortion) {
            auto mask = core::Tensor::zeros({static_cast<size_t>(camera.height), static_cast<size_t>(camera.width)}, core::Device::GPU, core::DataType::UInt8);
            if (means.shape()[0] == 0)
                return mask;
            const auto points = means.contiguous();
            const MeshCoverageParams params{.vertices = mk::address(points), .mask = mk::address(mask), .vertex_count = static_cast<int32_t>(points.shape()[0]), .padding = radius, .camera = pack_mesh_mask_camera(camera, nullptr, distortion)};
            mk::launch_items("point_coverage", params, {&points, &mask}, points.shape()[0]);
            return mask;
        }
    } // namespace

    const lfs::gpu_ops::MaskOps& metal_masks_ops() {
        static const lfs::gpu_ops::MaskOps ops{
            .point_coverage = point_coverage,
            .photometric_weight = photometric_weight,
            .opacity_penalty = opacity_penalty,
            .alpha_consistency = alpha_consistency,
            .mesh_coverage = mesh_coverage,
        };
        return ops;
    }
} // namespace lfs::training
