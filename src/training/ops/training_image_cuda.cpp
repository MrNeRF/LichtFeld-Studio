/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "lfs/training/ops/training_image_cuda.hpp"

#include "core/tensor_cuda_interop.hpp"
#include "kernels/camera_loss_heatmap.cuh"
#include "kernels/grad_alpha.hpp"
#include "kernels/image_kernels.hpp"
#include "kernels/roi_weight_map.hpp"

#include <glm/gtc/type_ptr.hpp>

namespace lfs::training {
    core::Tensor cuda_flip_error_map(const core::Tensor&, const core::Tensor&, float);
    core::Tensor cuda_flip_error_image(const core::Tensor&);

    namespace {
        using lfs::gpu_ops::RoiParams;
        using lfs::gpu_ops::Tensor;

        void heatmap(const Tensor& loss, Tensor& latest, Tensor& ema, int slot, float ema_alpha) {
            kernels::launch_update_camera_loss_heatmap(
                loss.ptr<float>(), slot, ema_alpha, latest.ptr<float>(), ema.ptr<float>(),
                latest.numel(), core::getCurrentCUDAStream());
        }

        void roi(const Tensor& view, const Tensor& camera_position, Tensor& weights, const RoiParams& p) {
            kernels::launch_roi_weight_map(
                view.ptr<float>(), camera_position.ptr<float>(),
                p.intrinsics.fx, p.intrinsics.fy, p.intrinsics.cx, p.intrinsics.cy,
                p.image.w, p.image.h, glm::make_mat4(p.world_to_cropbox.data()),
                glm::make_vec3(p.minimum.data()), glm::make_vec3(p.maximum.data()),
                p.inverse, p.outside_weight, weights.ptr<float>(), core::getCurrentCUDAStream());
        }

        void resize_background(const Tensor& source, Tensor& destination) {
            kernels::launch_bilinear_resize_chw(
                source.ptr<float>(), destination.ptr<float>(), source.shape()[0],
                source.shape()[1], source.shape()[2], destination.shape()[1], destination.shape()[2],
                core::getCurrentCUDAStream());
        }

        void random_background(Tensor& destination, uint64_t seed) {
            kernels::launch_random_background(destination.ptr<float>(), destination.shape()[1],
                                              destination.shape()[2], seed, core::getCurrentCUDAStream());
        }

        void canny(const Tensor& image, Tensor& edges) {
            const int height = image.shape()[1];
            const int width = image.shape()[2];
            const auto stream = core::getCurrentCUDAStream();
            if (image.dtype() == core::DataType::UInt8) {
                kernels::launch_fused_canny_edge_filter_chw(image.ptr<uint8_t>(), edges.ptr<float>(), height, width, stream);
            } else {
                kernels::launch_fused_canny_edge_filter_chw(image.ptr<float>(), edges.ptr<float>(), height, width, stream);
            }
        }

        void normalize_scalar(Tensor& values, const Tensor& scalar, float skip_below) {
            // Preserve the launcher's current-stream fallback used by the trainer.
            kernels::launch_normalize_by_device_scalar(values.ptr<float>(), values.numel(), scalar.ptr<float>(), skip_below);
        }

        const lfs::gpu_ops::TrainingImageOps kCudaTrainingImageOps{
            .heatmap = heatmap,
            .roi = roi,
            .resize_background = resize_background,
            .random_background = random_background,
            .canny = canny,
            .normalize_scalar = normalize_scalar,
            .quantize_to_8bit_grid = kernels::quantize_to_8bit_grid,
            .quantize_to_grid = kernels::quantize_to_grid,
            .flip_error_map = cuda_flip_error_map,
            .flip_error_image = cuda_flip_error_image,
        };
    } // namespace

    const lfs::gpu_ops::TrainingImageOps& cuda_training_image_ops() {
        return kCudaTrainingImageOps;
    }
} // namespace lfs::training
