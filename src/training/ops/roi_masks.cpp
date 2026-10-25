/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "roi_masks.hpp"
#include "core/assert.hpp"
#include "core/error.hpp"
#include "core/gpu_kernel_module.hpp"
#include "core/tensor_backend.hpp"
#include "roi_masks_program.hpp"
#include <array>
#include <cmath>
#include <format>

namespace lfs::training {
    float box_units_per_model_unit(const glm::mat4& model_to_cropbox) {
        // Largest stretch of the linear part: power iteration on A^T A.
        const glm::mat3 a(model_to_cropbox);
        const glm::mat3 ata = glm::transpose(a) * a;
        glm::vec3 v(1.0f, 0.7f, 0.4f);
        for (int i = 0; i < 32; ++i) {
            const glm::vec3 next = ata * v;
            const float length = glm::length(next);
            if (!(length > 0.0f))
                return 0.0f;
            v = next / length;
        }
        return std::sqrt(glm::length(ata * v));
    }

    void roi_reject_mask(gpu_ops::In means, gpu_ops::In scaling_raw, gpu_ops::In deleted,
                         gpu_ops::In cameras_in_box, const RoiBox& box,
                         gpu_ops::Out rejected, gpu_ops::Out in_roi) {
        using namespace core;
        const size_t n = means.is_valid() ? means.shape()[0] : 0;
        if (n == 0)
            return;
        const bool has_scales = scaling_raw.is_valid() && scaling_raw.numel() > 0;
        const bool has_deleted = deleted.is_valid() && deleted.numel() > 0;
        const bool has_in_roi = in_roi.is_valid();
        LFS_ASSERT_MSG(means.ndim() == 2 && means.shape()[1] == 3 && means.dtype() == DataType::Float32 && means.is_contiguous(),
                       std::format("ROI means must be contiguous float [N,3], got {}", means.shape().str()));
        LFS_ASSERT_MSG(!has_scales || (scaling_raw.numel() == n * 3 && scaling_raw.dtype() == DataType::Float32 &&
                                       scaling_raw.is_contiguous()),
                       std::format("ROI scales must be contiguous float [{},3], got {}", n, scaling_raw.shape().str()));
        LFS_ASSERT_MSG(!has_deleted || (deleted.numel() == n && deleted.dtype() == DataType::Bool && deleted.is_contiguous()),
                       std::format("ROI deleted rows must be contiguous bool [{}], got {}", n, deleted.shape().str()));
        LFS_ASSERT_MSG(cameras_in_box.is_valid() && cameras_in_box.ndim() == 2 && cameras_in_box.shape()[1] == 3 &&
                           cameras_in_box.dtype() == DataType::Float32 && cameras_in_box.is_contiguous(),
                       std::format("ROI cameras must be contiguous float [M,3], got {}", cameras_in_box.shape().str()));
        const auto bool_rows = [n](const Tensor& t) {
            return t.numel() == n && t.dtype() == DataType::Bool && t.is_contiguous();
        };
        LFS_ASSERT_MSG(bool_rows(rejected) && (!has_in_roi || bool_rows(in_roi)),
                       std::format("ROI masks must be contiguous bool [{}], got {} and {}", n, rejected.shape().str(),
                                   in_roi.is_valid() ? in_roi.shape().str() : std::string("none")));
        const auto& polygon = box.polygon;
        const bool has_polygon = polygon.is_valid() && polygon.numel() > 0;
        LFS_ASSERT_MSG(!has_polygon || (polygon.ndim() == 2 && polygon.shape()[1] == 2 && polygon.shape()[0] >= 3 &&
                                        polygon.dtype() == DataType::Float32 && polygon.is_contiguous()),
                       std::format("ROI polygon must be contiguous float [M>=3,2], got {}", polygon.shape().str()));
        LFS_ASSERT_MSG(n <= UINT32_MAX && cameras_in_box.shape()[0] <= UINT32_MAX,
                       std::format("ROI mask sizes exceed 32-bit indexing: {} splats, {} cameras", n, cameras_in_box.shape()[0]));

        const auto backend = gpu_backend_of(means).value_or(default_gpu_backend());
        const GpuBackendScope scope(backend);
        auto loaded = GpuKernelModule::load(roi_masks_program_entries(), backend);
        if (!loaded)
            throw Exception(std::move(loaded).error());
        const glm::mat4& m = box.model_to_cropbox;
        const auto model = Tensor::from_vector({m[0][0], m[1][0], m[2][0], m[3][0],
                                                m[0][1], m[1][1], m[2][1], m[3][1],
                                                m[0][2], m[1][2], m[2][2], m[3][2]},
                                               {12}, Device::GPU);
        struct Parameters {
            uint64_t means = 0, scales = 0, model = 0, cameras = 0, deleted = 0, rejected = 0, in_roi = 0, polygon = 0;
            float lo_x, lo_y, lo_z, hi_x, hi_y, hi_z, margin_scale;
            uint32_t count, camera_count, inverse, polygon_count, padding = 0;
        } p{.lo_x = box.min.x, .lo_y = box.min.y, .lo_z = box.min.z, .hi_x = box.max.x, .hi_y = box.max.y, .hi_z = box.max.z, .margin_scale = box_units_per_model_unit(m), .count = uint32_t(n), .camera_count = uint32_t(cameras_in_box.shape()[0]), .inverse = box.inverse ? 1u : 0u, .polygon_count = has_polygon ? uint32_t(polygon.shape()[0]) : 0u};
        static_assert(sizeof(Parameters) == 112 && offsetof(Parameters, count) == 92 && offsetof(Parameters, polygon_count) == 104);
        using Binding = GpuKernelModule::Binding;
        const std::array bindings{Binding{0, &means}, Binding{8, has_scales ? &scaling_raw : nullptr}, Binding{16, &model},
                                  Binding{24, &cameras_in_box}, Binding{32, has_deleted ? &deleted : nullptr},
                                  Binding{40, &rejected, GpuKernelModule::Access::ReadWrite},
                                  Binding{48, has_in_roi ? &in_roi : nullptr, GpuKernelModule::Access::ReadWrite},
                                  Binding{56, has_polygon ? &polygon : nullptr}};
        auto result = (*loaded)->dispatch({.function = "reject",
                                           .arguments = {std::as_bytes(std::span(&p, 1)), bindings},
                                           .groups = {uint32_t((n + 255) / 256), 1, 1},
                                           .group = {256, 1, 1}});
        if (!result)
            throw Exception(std::move(result).error());
    }
} // namespace lfs::training
