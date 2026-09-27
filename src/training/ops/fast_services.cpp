/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "lfs/training/ops/fast_services.hpp"

#include <cstdint>
#include <string>

namespace lfs::training {
    namespace {

        lfs::gpu_ops::ShStorage fast_sh_storage(const core::SplatData& model) {
            if (model.shN_value_quantized()) {
                return lfs::gpu_ops::ShStorage::Q16;
            }
            if (model.shN_ieee_f16()) {
                return lfs::gpu_ops::ShStorage::IeeeFloat16;
            }
            return lfs::gpu_ops::ShStorage::Float32;
        }

    } // namespace

    lfs::Error fast_raster_error(const lfs::gpu_ops::RasterResult& result) {
        using Code = lfs::gpu_ops::RasterResult::Code;
        const std::string detail{result.message};
        if (detail.find("n_primitives is 0") != std::string::npos) {
            return lfs::make_error(lfs::ErrorInit{
                .code = lfs::ErrorCode::InvalidArgument,
                .domain = lfs::ErrorDomain::Rendering,
                .user_message = "FastGS cannot render an empty Gaussian model.",
                .detail = detail,
                .detection = LFS_SOURCE_SITE_CURRENT(),
            });
        }
        if (result.code == Code::InstanceOverflow) {
            return lfs::make_error(lfs::ErrorInit{
                .code = lfs::ErrorCode::FailedPrecondition,
                .domain = lfs::ErrorDomain::CUDA,
                .user_message =
                    "Pathological splat extents produced more tile instances "
                    "than the 32-bit FastGS path can represent; skipping step.",
                .detail = detail,
                .detection = LFS_SOURCE_SITE_CURRENT(),
            });
        }
        const bool exhausted = result.code == Code::ResourceExhausted;
        return lfs::make_error(lfs::ErrorInit{
            .code = exhausted ? lfs::ErrorCode::ResourceExhausted : lfs::ErrorCode::Internal,
            .domain = lfs::ErrorDomain::CUDA,
            .user_message = exhausted ? "Ran out of GPU memory while rendering during training."
                                      : "FastGS forward rasterization failed.",
            .detail = detail,
            .detection = LFS_SOURCE_SITE_CURRENT(),
        });
    }

    lfs::gpu_ops::RasterResult fast_render(
        const lfs::gpu_ops::FastRasterOps& ops,
        lfs::gpu_ops::FastSaved& saved,
        core::Camera& camera,
        core::SplatData& model,
        core::Tensor& bg_color,
        const int tile_x,
        const int tile_y,
        const int tile_width,
        const int tile_height,
        const bool mip_filter,
        const core::Tensor& bg_image,
        const bool render_normal,
        const bool render_depth,
        RenderOutput& output) {
        const int sh_degree = model.get_active_sh_degree();
        const int max_sh_degree = model.get_max_sh_degree();
        auto [fx, fy, cx, cy] = camera.get_intrinsics();
        core::Tensor no_bounds;
        const bool q16 = model.shN_value_quantized();
        const lfs::gpu_ops::FastParams params{
            .full_image = {camera.image_height(), camera.image_width()},
            .intrinsics = {fx, fy, cx, cy},
            .tile_x = tile_x,
            .tile_y = tile_y,
            .tile_w = tile_width,
            .tile_h = tile_height,
            .sh = {
                .storage = fast_sh_storage(model),
                .active_bases = static_cast<uint32_t>((sh_degree + 1) * (sh_degree + 1)),
                .layout_bases = static_cast<uint32_t>((max_sh_degree + 1) * (max_sh_degree + 1)),
            },
            .mip_filter = mip_filter,
            .render_normal = render_normal,
            .render_depth = render_depth,
        };
        const lfs::gpu_ops::SplatInputs splats{
            .means = model.means(),
            .raw_scales = model.scaling_raw(),
            .raw_rotations = model.rotation_raw(),
            .raw_opacities = model.opacity_raw(),
            .sh0 = model.sh0(),
            .shN = model.shN(),
            .sh_value_bounds = q16 ? model.shN_value_bounds() : no_bounds,
        };
        const lfs::gpu_ops::RenderOutputs rendered{
            .image = output.image,
            .alpha = output.alpha,
            .depth = output.depth,
            .normal = output.normal,
        };
        const auto result = ops.forward(
            saved, splats, camera.world_view_transform(), camera.cam_position(),
            bg_color, bg_image, params, rendered, model._max_screen_share);
        if (result.code == lfs::gpu_ops::RasterResult::Code::Success) {
            output.width = tile_width > 0 ? tile_width : camera.image_width();
            output.height = tile_height > 0 ? tile_height : camera.image_height();
        }
        return result;
    }

    RenderOutput fast_infer(
        const lfs::gpu_ops::FastRasterOps& ops,
        lfs::gpu_ops::FastSaved& saved,
        core::Camera& camera,
        core::SplatData& model,
        core::Tensor& bg_color,
        const bool mip_filter,
        const core::Tensor& bg_image,
        const bool render_normal) {
        RenderOutput output;
        const auto result = fast_render(
            ops, saved, camera, model, bg_color, 0, 0, 0, 0, mip_filter, bg_image, render_normal, true, output);
        if (result.code != lfs::gpu_ops::RasterResult::Code::Success) {
            throw lfs::Exception(fast_raster_error(result));
        }
        ops.release(saved);
        return output;
    }

} // namespace lfs::training
