/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "metal_families.hpp"
#include "metal_kernels.hpp"

#include <format>
#include <stdexcept>

namespace lfs::training {
    namespace {
        using lfs::gpu_ops::RoiParams;
        using lfs::gpu_ops::Tensor;
        namespace mk = metal;

        struct HeatmapParams {
            uint64_t loss, latest, ema;
            int32_t slot;
            uint32_t slot_count;
            float ema_alpha;
        };

        void heatmap(const Tensor& loss, Tensor& latest, Tensor& ema, const int slot, const float ema_alpha) {
            const HeatmapParams params{mk::address(loss), mk::address(latest), mk::address(ema), slot,
                                       mk::count32(latest.numel(), "EMA"), ema_alpha};
            mk::launch("training_image_heatmap", params, {&loss, &latest, &ema}, 1, 1);
        }

        struct RoiKernelParams {
            uint64_t world_to_camera, camera_position, output;
            mk::Float4 world_to_cropbox[3];
            mk::Float3 crop_min, crop_max;
            float fx, fy, cx, cy;
            int32_t width, height;
            float outside_weight;
            uint32_t inverse;
        };

        void roi(const Tensor& view, const Tensor& camera_position, Tensor& weights, const RoiParams& p) {
            if (p.image.w <= 0 || p.image.h <= 0 || !(p.intrinsics.fx > 0.f) || !(p.intrinsics.fy > 0.f))
                throw std::invalid_argument(std::format("ROI needs a positive image and focal length, got {}x{} f=({}, {})",
                                                        p.image.w, p.image.h, p.intrinsics.fx, p.intrinsics.fy));
            // world_to_cropbox is column-major; rows of its upper 3x4 block.
            const auto& m = p.world_to_cropbox;
            RoiKernelParams params{
                .world_to_camera = mk::address(view),
                .camera_position = mk::address(camera_position),
                .output = mk::address(weights),
                .world_to_cropbox = {{m[0], m[4], m[8], m[12]}, {m[1], m[5], m[9], m[13]}, {m[2], m[6], m[10], m[14]}},
                .crop_min = {p.minimum[0], p.minimum[1], p.minimum[2]},
                .crop_max = {p.maximum[0], p.maximum[1], p.maximum[2]},
                .fx = p.intrinsics.fx,
                .fy = p.intrinsics.fy,
                .cx = p.intrinsics.cx,
                .cy = p.intrinsics.cy,
                .width = p.image.w,
                .height = p.image.h,
                .outside_weight = p.outside_weight,
                .inverse = p.inverse ? 1u : 0u,
            };
            mk::launch_items("training_image_roi", params, {&view, &camera_position, &weights},
                             static_cast<size_t>(p.image.w) * static_cast<size_t>(p.image.h));
        }

        struct ResizeParams {
            uint64_t source, destination;
            int32_t channels, source_h, source_w, destination_h, destination_w;
        };

        void resize_background(const Tensor& source, Tensor& destination) {
            const ResizeParams params{mk::address(source), mk::address(destination),
                                      static_cast<int32_t>(source.shape()[0]), static_cast<int32_t>(source.shape()[1]),
                                      static_cast<int32_t>(source.shape()[2]),
                                      static_cast<int32_t>(destination.shape()[1]),
                                      static_cast<int32_t>(destination.shape()[2])};
            mk::launch_items("training_image_resize", params, {&source, &destination},
                             destination.shape()[1] * destination.shape()[2]);
        }

        struct RandomBackgroundParams {
            uint64_t output;
            int32_t plane;
            uint32_t seed;
        };

        void random_background(Tensor& destination, const uint64_t seed) {
            const size_t plane = destination.shape()[1] * destination.shape()[2];
            const RandomBackgroundParams params{mk::address(destination), static_cast<int32_t>(plane),
                                                static_cast<uint32_t>(seed)};
            mk::launch_items("training_image_random_background", params, {&destination}, plane);
        }

        struct CannyParams {
            uint64_t input, output;
            int32_t height, width;
        };

        void canny(const Tensor& image, Tensor& edges) {
            const int height = static_cast<int>(image.shape()[1]);
            const int width = static_cast<int>(image.shape()[2]);
            const CannyParams params{mk::address(image), mk::address(edges), height, width};
            const bool bytes = image.dtype() == core::DataType::UInt8;
            mk::launch_2d("training_image_canny", params, {&image, &edges},
                          core::GpuKernelModule::groups_for(width, 16), core::GpuKernelModule::groups_for(height, 16),
                          16, 16, {{0, bytes ? 1u : 0u}});
        }

        struct NormalizeScalarParams {
            uint64_t values, scalar;
            uint32_t count;
            float skip_below;
        };

        void normalize_scalar(Tensor& values, const Tensor& scalar, const float skip_below) {
            if (values.numel() == 0)
                return;
            const NormalizeScalarParams params{mk::address(values), mk::address(scalar),
                                               mk::count32(values.numel(), "values"), skip_below};
            mk::launch_items("training_image_normalize_scalar", params, {&values, &scalar}, values.numel());
        }
    } // namespace

    const lfs::gpu_ops::TrainingImageOps& metal_training_image_ops() {
        static const lfs::gpu_ops::TrainingImageOps ops{
            .heatmap = heatmap,
            .roi = roi,
            .resize_background = resize_background,
            .random_background = random_background,
            .canny = canny,
            .normalize_scalar = normalize_scalar,
        };
        return ops;
    }
} // namespace lfs::training
