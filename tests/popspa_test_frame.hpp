/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/camera.hpp"
#include "core/splat_data.hpp"
#include "core/tensor.hpp"
#include "core/tensor_backend.hpp"
#include "lfs/training/ops/fast_services.hpp"
#include "lfs/training/ops/gsplat_services.hpp"
#include "lfs/training/ops/registry.hpp"
#include "optimizer/adam_optimizer.hpp"

#include <cstddef>
#include <vector>

namespace lfs::test {

    // One forward frame of either renderer through the default backend's ops table.
    struct PopFrame {
        const training::TrainingOps& ops = training::training_ops(core::default_gpu_backend());
        bool fast;
        gpu_ops::FastSaved fast_saved;
        gpu_ops::GsplatSaved gsplat_saved;
        training::RenderOutput output;
        bool has_work = false;

        explicit PopFrame(const bool use_fast) : fast(use_fast) {
            if (fast)
                fast_saved.backend = ops.fast->create();
            else
                gsplat_saved.backend = ops.gsplat->create();
        }
        ~PopFrame() {
            if (fast)
                ops.fast->release(fast_saved);
            else
                ops.gsplat->release(gsplat_saved);
        }
        PopFrame(const PopFrame&) = delete;
        PopFrame& operator=(const PopFrame&) = delete;

        [[nodiscard]] bool render(core::Camera& camera, core::SplatData& model, core::Tensor& background,
                                  const core::Tensor& bg_image = {}, const bool mip_filter = false) {
            const auto result = fast ? training::fast_render(*ops.fast, fast_saved, camera, model, background, 0, 0, 0, 0,
                                                             mip_filter, bg_image, false, true, output)
                                     : training::gsplat_render(*ops.gsplat, gsplat_saved, camera, model, background, 0, 0, 0,
                                                               0, 1.f, false, training::GsplatRenderMode::RGB, bg_image, output);
            has_work = result.has_work;
            return result.code == gpu_ops::RasterResult::Code::Success;
        }
        void score(const core::Tensor& background, const core::Tensor& bg_image, core::Tensor& scores) const {
            if (fast)
                ops.fast->pop_scores(fast_saved, background, bg_image, scores);
            else
                ops.gsplat->pop_scores(gsplat_saved, background, bg_image, scores);
        }
        // Adds the raw gradients of an image loss into the optimizer gradients; no Adam step.
        void backward(const core::Tensor& image_gradient, training::AdamOptimizer& optimizer) {
            core::Tensor none;
            if (fast) {
                ops.fast->backward(fast_saved, {.image = image_gradient, .alpha = none, .depth = none, .normal = none},
                                   none, none, none, none,
                                   optimizer.prepare_fastgs_gradient_capture(core::TensorExecutionTarget::current()),
                                   ::DensificationType::None);
            } else {
                auto alpha_gradient = core::Tensor::zeros(
                    {1, static_cast<size_t>(output.height), static_cast<size_t>(output.width)}, core::Device::GPU);
                ops.gsplat->backward(gsplat_saved, image_gradient, alpha_gradient, training::gsplat_gradients(optimizer),
                                     none, none, none, none, none);
            }
        }
    };

    // POPSpa scores [N, 8] in the default backend's encoding, as doubles.
    inline std::vector<double> decode_pop_scores(const core::Tensor& scores) {
        const auto host = scores.cpu().contiguous();
        const auto encoding = training::training_ops(core::default_gpu_backend()).popspa->score_encoding;
        std::vector<double> values(host.shape()[0]);
        for (size_t i = 0; i < values.size(); ++i)
            values[i] = gpu_ops::decode_pop_score(static_cast<const std::byte*>(host.data_ptr()) + 8 * i, encoding);
        return values;
    }

} // namespace lfs::test
