/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

// Metal PPISPOps and ControllerOps; see ops/ppisp_cuda.cpp for the reference behaviour.

#include "metal_families.hpp"
#include "metal_kernels.hpp"

#include <algorithm>
#include <cmath>
#include <format>
#include <stdexcept>

namespace lfs::training {
    namespace {
        using namespace lfs::gpu_ops;
        namespace mk = metal;

        struct ForwardParams {
            uint64_t exposure, vignetting, color, crf, rgb_in, rgb_out;
            int32_t height, width, y_offset, full_height, camera_index, frame_index, x_offset, full_width;
        };

        void forward(const PPISPInputs& p, In rgb, Out corrected, const PPISPRegion& r) {
            const int height = static_cast<int>(rgb.shape()[1]);
            const int width = static_cast<int>(rgb.shape()[2]);
            const int full_width = r.full_width > 0 ? r.full_width : width;
            if (r.x_offset < 0 || r.x_offset + width > full_width)
                throw std::invalid_argument("PPISP crop exceeds full width");
            if (height <= 0 || width <= 0 || r.y_offset < 0 || r.y_offset + height > r.full_height)
                throw std::invalid_argument(std::format("PPISP band [{}, {}) of width {} must lie inside {} rows",
                                                        r.y_offset, r.y_offset + height, width, r.full_height));
            const ForwardParams params{mk::address(p.exposure), mk::address(p.vignetting), mk::address(p.color),
                                       mk::address(p.crf), mk::address(rgb), mk::address(corrected),
                                       height, width, r.y_offset, r.full_height, r.camera_index, r.frame_index, r.x_offset, full_width};
            mk::launch_items("ppisp_forward", params, {&p.exposure, &p.vignetting, &p.color, &p.crf, &rgb, &corrected},
                             static_cast<size_t>(height) * width);
        }

        struct BackwardParams {
            uint64_t exposure, vignetting, color, crf, rgb_in, grad_rgb_out;
            uint64_t grad_exposure, grad_vignetting, grad_color, grad_crf, grad_rgb_in;
            int32_t height, width, camera_index, frame_index;
        };

        // Parameter gradients accumulate, as the CUDA atomics do.
        void backward(const PPISPInputs& p, In rgb, In grad_output, const PPISPOutputs& g, Out grad_rgb,
                      int /*cameras*/, int /*frames*/, const int camera_index, const int frame_index) {
            const int height = static_cast<int>(rgb.shape()[1]);
            const int width = static_cast<int>(rgb.shape()[2]);
            const BackwardParams params{mk::address(p.exposure), mk::address(p.vignetting), mk::address(p.color),
                                        mk::address(p.crf), mk::address(rgb), mk::address(grad_output),
                                        mk::address(g.exposure), mk::address(g.vignetting), mk::address(g.color),
                                        mk::address(g.crf), mk::address(grad_rgb),
                                        height, width, camera_index, frame_index};
            mk::launch_items("ppisp_backward", params,
                             {&p.exposure, &p.vignetting, &p.color, &p.crf, &rgb, &grad_output, &g.exposure,
                              &g.vignetting, &g.color, &g.crf, &grad_rgb},
                             static_cast<size_t>(height) * width);
        }

        struct AdamGroup {
            uint64_t parameter, moment1, moment2, gradient;
            uint32_t count, padding;
        };

        struct AdamParams {
            AdamGroup groups[4];
            float lr, beta1, beta2, bc1_rcp, bc2_sqrt_rcp, eps;
        };

        AdamGroup bind(const PPISPAdamGroup& g) {
            if (!g.parameter.is_valid())
                return {};
            return {mk::address(g.parameter), mk::address(g.moment1), mk::address(g.moment2), mk::address(g.gradient),
                    mk::count32(g.parameter.numel(), "PPISP parameter"), 0};
        }

        size_t adam_items(const AdamParams& params) {
            size_t total = 0;
            for (const auto& g : params.groups)
                total += g.count;
            return total;
        }

        void adam(const PPISPAdamGroup& g, const PPISPAdamUpdateParams& h) {
            const AdamParams params{{bind(g), {}, {}, {}}, h.lr, h.beta1, h.beta2, h.bc1_rcp, h.bc2_sqrt_rcp, h.eps};
            if (const size_t items = adam_items(params))
                mk::launch_items("ppisp_adam", params, {&g.parameter, &g.moment1, &g.moment2, &g.gradient}, items);
        }

        void adam_batch(const std::array<PPISPAdamGroup, 4>& groups, const PPISPAdamUpdateParams& h) {
            const auto& [a, b, c, d] = groups;
            const AdamParams params{{bind(a), bind(b), bind(c), bind(d)},
                                    h.lr,
                                    h.beta1,
                                    h.beta2,
                                    h.bc1_rcp,
                                    h.bc2_sqrt_rcp,
                                    h.eps};
            if (const size_t items = adam_items(params))
                mk::launch_items("ppisp_adam", params,
                                 {&a.parameter, &a.moment1, &a.moment2, &a.gradient, &b.parameter, &b.moment1,
                                  &b.moment2, &b.gradient, &c.parameter, &c.moment1, &c.moment2, &c.gradient,
                                  &d.parameter, &d.moment1, &d.moment2, &d.gradient},
                                 items);
        }

        struct VignettingRegParams {
            uint64_t vignetting, gradient, loss;
            int32_t cameras;
            float center, channel, non_positive;
        };

        void vignetting_regularization(In parameters, Out gradient, Out loss, const float center, const float channel,
                                       const float non_positive) {
            const int cameras = static_cast<int>(parameters.numel() / 15);
            if (cameras <= 0)
                return;
            const VignettingRegParams params{mk::address(parameters), mk::address(gradient), mk::address(loss),
                                             cameras, center, channel, non_positive};
            mk::launch_items("ppisp_vignetting_reg", params, {&parameters, &gradient, &loss},
                             static_cast<size_t>(cameras));
        }

        struct ProjectMeanParams {
            uint64_t exposure, color;
            int32_t frames;
        };

        void project_mean(Out exposure, Out color) {
            const int frames = static_cast<int>(exposure.numel());
            if (frames <= 0)
                return;
            const ProjectMeanParams params{mk::address(exposure), mk::address(color), frames};
            mk::launch("ppisp_project_mean", params, {&exposure, &color}, 2);
        }

        struct InitializeParams {
            uint64_t exposure, vignetting, color, crf;
            int32_t cameras, frames;
            float toe, shoulder, gamma;
        };

        // Raw value whose bounded-positive transform is `target`.
        float bounded_positive_inverse(const float target, const float min_value) {
            return std::log(std::exp(target - min_value) - 1.0f);
        }

        void initialize(const PPISPOutputs& p) {
            const int cameras = static_cast<int>(p.vignetting.numel() / 15);
            const int frames = static_cast<int>(p.exposure.numel());
            const InitializeParams params{mk::address(p.exposure), mk::address(p.vignetting), mk::address(p.color),
                                          mk::address(p.crf), cameras, frames,
                                          bounded_positive_inverse(1.0f, 0.3f), bounded_positive_inverse(1.0f, 0.3f),
                                          bounded_positive_inverse(1.0f, 0.1f)};
            const size_t items = std::max({frames, cameras * 15, frames * 8, cameras * 12});
            mk::launch_items("ppisp_initialize", params, {&p.exposure, &p.vignetting, &p.color, &p.crf}, items);
        }

        constexpr int kFlatFeatures = 1600;

        struct PrepareParams {
            uint64_t features, fc_input;
            float exposure_prior;
            uint32_t write_prior;
        };

        // Launch parameters are copied at encode time, so the prior needs no host wait.
        void prepare_input(In features, Out fc_input, const float exposure_prior) {
            if (fc_input.numel() <= static_cast<size_t>(kFlatFeatures) ||
                (features.is_valid() && features.numel() < static_cast<size_t>(kFlatFeatures)))
                throw std::invalid_argument(std::format("controller input needs {} features and {} input slots, got {} and {}",
                                                        kFlatFeatures, kFlatFeatures + 1,
                                                        features.is_valid() ? features.numel() : 0, fc_input.numel()));
            const bool write_prior = !features.is_valid() || exposure_prior != 1.0f;
            const PrepareParams params{mk::address(features), mk::address(fc_input), exposure_prior, write_prior ? 1u : 0u};
            mk::launch_items("controller_prepare_input", params, {&features, &fc_input}, kFlatFeatures + 1);
        }

        struct OuterParams {
            uint64_t grad, activation, weight_gradient, bias_gradient;
            int32_t m, n;
        };

        struct InputGradParams {
            uint64_t grad, activation, weight, grad_input;
            int32_t m, n;
        };

        void backward_layer(In grad_output, In activation, In weight, Out weight_gradient, Out bias_gradient,
                            Out grad_input) {
            const int m = static_cast<int>(grad_output.numel());
            const int n = static_cast<int>(activation.numel());
            const OuterParams outer{mk::address(grad_output), mk::address(activation), mk::address(weight_gradient),
                                    mk::address(bias_gradient), m, n};
            mk::launch_items("controller_outer_product", outer, {&grad_output, &activation, &weight_gradient, &bias_gradient},
                             static_cast<size_t>(m) * n);
            if (!grad_input.is_valid())
                return;
            const InputGradParams input{mk::address(grad_output), mk::address(activation), mk::address(weight),
                                        mk::address(grad_input), m, n};
            mk::launch_items("controller_input_grad", input, {&grad_output, &activation, &weight, &grad_input},
                             static_cast<size_t>(n));
        }
    } // namespace

    const gpu_ops::PPISPOps& metal_ppisp_ops() {
        static const gpu_ops::PPISPOps ops{
            .forward = forward,
            .backward = backward,
            .adam = adam,
            .adam_batch = adam_batch,
            .vignetting_regularization = vignetting_regularization,
            .project_mean = project_mean,
            .initialize = initialize,
        };
        return ops;
    }

    const gpu_ops::ControllerOps& metal_controller_ops() {
        static const gpu_ops::ControllerOps ops{
            .prepare_input = prepare_input,
            .backward_layer = backward_layer,
        };
        return ops;
    }
} // namespace lfs::training
