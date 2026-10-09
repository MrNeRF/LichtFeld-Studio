/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "lfs/training/ops/gsplat_services.hpp"

#include "optimizer/adam_optimizer.hpp"

#include <cstdint>
#include <stdexcept>
#include <string>

namespace lfs::training {

    lfs::gpu_ops::GsplatGradients gsplat_gradients(AdamOptimizer& optimizer, const core::Tensor& depth, const core::Tensor& normal, const float flatten_weight) {
        return {&optimizer, [](void* owner, lfs::gpu_ops::AdamSlot slot) -> core::Tensor& {
                    auto& optimizer = *static_cast<AdamOptimizer*>(owner);
                    switch (slot) {
                    case lfs::gpu_ops::AdamSlot::Means: return optimizer.get_grad(ParamType::Means);
                    case lfs::gpu_ops::AdamSlot::Scaling: return optimizer.get_grad(ParamType::Scaling);
                    case lfs::gpu_ops::AdamSlot::Rotation: return optimizer.get_grad(ParamType::Rotation);
                    case lfs::gpu_ops::AdamSlot::Opacity: return optimizer.get_grad(ParamType::Opacity);
                    case lfs::gpu_ops::AdamSlot::Sh0: return optimizer.get_grad(ParamType::Sh0);
                    case lfs::gpu_ops::AdamSlot::ShN: return optimizer.get_grad(ParamType::ShN);
                    }
                    throw std::logic_error("Invalid gsplat gradient slot");
                },
                depth, normal, flatten_weight};
    }

    lfs::Error gsplat_raster_error(const lfs::gpu_ops::RasterResult& result) {
        const bool exhausted = result.code == lfs::gpu_ops::RasterResult::Code::ResourceExhausted;
        return lfs::make_error(lfs::ErrorInit{
            .code = exhausted ? lfs::ErrorCode::ResourceExhausted : lfs::ErrorCode::Internal,
            .domain = lfs::ErrorDomain::Rendering,
            .user_message = exhausted ? "Ran out of GPU memory while rendering during training." : "Gsplat forward rasterization failed.",
            .detail = std::string(result.message),
            .detection = LFS_SOURCE_SITE_CURRENT(),
        });
    }

    lfs::gpu_ops::RasterResult gsplat_render(
        const lfs::gpu_ops::GsplatRasterOps& ops, lfs::gpu_ops::GsplatSaved& saved,
        const core::Camera& camera, core::SplatData& model, const core::Tensor& bg_color,
        int tile_x, int tile_y, int tile_w, int tile_h,
        float scaling_modifier, bool antialiased, GsplatRenderMode mode,
        const core::Tensor& bg_image, RenderOutput& output) {
        const auto [fx, fy, cx, cy] = camera.get_intrinsics();
        const auto degree = model.get_active_sh_degree();
        const auto max_degree = model.get_max_sh_degree();
        const bool q16 = model.shN_value_quantized();
        const bool undistorted = camera.is_undistort_prepared();
        const core::Tensor empty;
        const lfs::gpu_ops::GsplatParams params{
            .full_image = {camera.image_height(), camera.image_width()},
            .intrinsics = {fx, fy, cx, cy},
            .tile_x = tile_x,
            .tile_y = tile_y,
            .tile_w = tile_w,
            .tile_h = tile_h,
            .sh = {.storage = q16 ? lfs::gpu_ops::ShStorage::Q16 : model.shN_ieee_f16() ? lfs::gpu_ops::ShStorage::IeeeFloat16
                                                                                        : lfs::gpu_ops::ShStorage::Float32,
                   .active_bases = static_cast<uint32_t>((degree + 1) * (degree + 1)),
                   .layout_bases = static_cast<uint32_t>((max_degree + 1) * (max_degree + 1))},
            .camera_model = undistorted ? core::CameraModelType::PINHOLE : camera.camera_model_type(),
            .render_mode = mode,
            .scaling_modifier = scaling_modifier,
            .antialiased = antialiased,
        };
        const lfs::gpu_ops::SplatInputs splats{
            model.means(), model.scaling_raw(), model.rotation_raw(), model.opacity_raw(),
            model.sh0(), model.shN(), q16 ? model.shN_value_bounds() : empty};
        saved.equirectangular = params.camera_model == core::CameraModelType::EQUIRECTANGULAR;
        const auto result = ops.forward(saved, splats, camera.world_view_transform(),
                                        undistorted ? empty : camera.radial_distortion(),
                                        undistorted ? empty : camera.tangential_distortion(), bg_color, bg_image, params,
                                        {output.image, output.alpha, output.depth, output.normal});
        output.width = tile_w > 0 ? tile_w : camera.image_width();
        output.height = tile_h > 0 ? tile_h : camera.image_height();
        return result;
    }

    RenderOutput gsplat_infer(
        const lfs::gpu_ops::GsplatRasterOps& ops, lfs::gpu_ops::GsplatSaved& saved,
        const core::Camera& camera, core::SplatData& model, const core::Tensor& bg_color,
        float scaling_modifier, bool antialiased, GsplatRenderMode mode) {
        RenderOutput output;
        const auto result = gsplat_render(ops, saved, camera, model, bg_color,
                                          0, 0, 0, 0, scaling_modifier, antialiased, mode, {}, output);
        if (result.code != lfs::gpu_ops::RasterResult::Code::Success)
            throw lfs::Exception(gsplat_raster_error(result));
        ops.release(saved);
        return output;
    }

} // namespace lfs::training
