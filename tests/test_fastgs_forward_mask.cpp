/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "backward.h"
#include "core/alloc_counter.hpp"
#include "core/camera.hpp"
#include "core/splat_data.hpp"
#include "core/tensor.hpp"
#include "forward.h"
#include "lfs/training/sh_value_storage.hpp"
#include "training/optimizer/adam_optimizer.hpp"
#include "training/rasterization/fast_rasterizer.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <gtest/gtest.h>
#include <stdexcept>
#include <vector>

namespace {
    using namespace lfs::core;
    using namespace lfs::training;

    struct CullMode {
        const int previous = fast_lfs::rasterization::warp_cull_mode_for_testing();
        explicit CullMode(int mode) { fast_lfs::rasterization::set_warp_cull_mode_for_testing(mode); }
        ~CullMode() { fast_lfs::rasterization::set_warp_cull_mode_for_testing(previous); }
    };

    struct Capture {
        std::array<Tensor, 4> images;
        std::vector<unsigned> indices;
        std::vector<unsigned> last;
        std::vector<std::array<unsigned, 4>> contributors;
        std::vector<uint2> ranges;
    };

    void copy(void* to, const void* from, size_t bytes, cudaStream_t stream) {
        if (bytes && cudaMemcpyAsync(to, from, bytes, cudaMemcpyDeviceToHost, stream) != cudaSuccess)
            throw std::runtime_error("Could not capture renderer indices");
    }

    Capture capture(Camera& camera, SplatData& model, int mode, bool mip, bool tile) {
        const CullMode culling(mode);
        const int width = tile ? 31 : 65;
        const int height = tile ? 29 : 63;
        const auto background = Tensor::full({3, static_cast<size_t>(height), static_cast<size_t>(width)}, 0.17f, Device::CUDA);
        auto background_color = Tensor::full({3}, 0.1f, Device::CUDA);
        auto result = fast_rasterize_forward(camera, model, background_color,
                                             tile ? 7 : 0, tile ? 5 : 0, tile ? width : 0, tile ? height : 0,
                                             mip, background, true, true);
        if (!result)
            throw std::runtime_error(lfs::format_for_developer(result.error()));
        auto& [output, context] = *result;
        Capture data;
        data.images = {output.image.to(Device::CPU), output.alpha.to(Device::CPU),
                       output.depth.to(Device::CPU), output.normal.to(Device::CPU)};
        const auto& raw = context.forward_ctx;
        const auto view = fast_lfs::rasterization::forward_capture_view(raw, width, height);
        data.indices.resize(raw.n_instances);
        data.last.resize(width * height);
        data.ranges.resize(((width + 15) / 16) * ((height + 15) / 16));
        copy(data.indices.data(), raw.sorted_primitive_indices, data.indices.size() * sizeof(unsigned), raw.stream);
        copy(data.last.data(), view.last_contributors, data.last.size() * sizeof(unsigned), raw.stream);
        copy(data.ranges.data(), view.tile_ranges, data.ranges.size() * sizeof(uint2), raw.stream);
        if (raw.contributor_masks) {
            std::vector<unsigned> positions(raw.n_instances);
            std::vector<unsigned char> masks(raw.n_instances);
            copy(positions.data(), raw.contributor_positions, positions.size() * sizeof(unsigned), raw.stream);
            copy(masks.data(), raw.contributor_masks, masks.size(), raw.stream);
            if (cudaStreamSynchronize(raw.stream) != cudaSuccess)
                throw std::runtime_error("Could not synchronize contributor capture");
            for (size_t tile = 0; tile < data.ranges.size(); ++tile) {
                const auto range = data.ranges[tile];
                for (unsigned i = range.x; i < range.y; ++i) {
                    if (masks[i])
                        data.contributors.push_back({static_cast<unsigned>(tile), positions[i], data.indices[i], masks[i]});
                }
            }
        }
        if (cudaStreamSynchronize(raw.stream) != cudaSuccess)
            throw std::runtime_error("Could not synchronize renderer capture");
        return data;
    }

    TEST(FastGSForwardMask, CulledCompactedRenderMatchesUnculledRender) {
        constexpr size_t count = 259;
        auto means = Tensor::empty({count, 3}, Device::CPU);
        auto rotations = Tensor::zeros({count, 4}, Device::CPU);
        auto scales = Tensor::full({count, 3}, -3.2f, Device::CPU);
        for (size_t i = 0; i < count; ++i) {
            means.ptr<float>()[3 * i] = (static_cast<int>(i % 13) - 6) * 0.1f;
            means.ptr<float>()[3 * i + 1] = (static_cast<int>((i / 13) % 11) - 5) * 0.1f;
            means.ptr<float>()[3 * i + 2] = 0.001f * i;
            rotations.ptr<float>()[4 * i] = 1.0f;
            if (i % 17 == 0)
                scales.ptr<float>()[3 * i] = scales.ptr<float>()[3 * i + 1] = -0.2f;
        }
        SplatData model(0, means.cuda(), Tensor::full({count, 1, 3}, 0.3f, Device::CUDA),
                        Tensor::zeros({count, 0, 3}, Device::CUDA), scales.cuda(), rotations.cuda(),
                        Tensor::full({count}, 1.0f, Device::CUDA), 1.0f);
        auto translation = Tensor::zeros({3}, Device::CPU);
        translation.ptr<float>()[2] = 3.0f;
        Camera camera(Tensor::eye(3, Device::CUDA), translation.cuda(), 60.0f, 61.0f, 32.0f, 31.0f,
                      Tensor{}, Tensor{}, CameraModelType::PINHOLE, "mask", "", std::filesystem::path{}, 65, 63, 0);
        for (bool mip : {false, true}) {
            for (bool tile : {false, true}) {
                SCOPED_TRACE(::testing::Message() << "mip=" << mip << " tile=" << tile);
                const auto culled = capture(camera, model, 0, mip, tile);
                const auto reference = capture(camera, model, 1, mip, tile);
                ASSERT_FALSE(culled.contributors.empty());
                EXPECT_EQ(culled.last, reference.last);
                EXPECT_EQ(culled.contributors, reference.contributors);
                for (size_t channel = 0; channel < culled.images.size(); ++channel) {
                    ASSERT_EQ(culled.images[channel].numel(), reference.images[channel].numel());
                    EXPECT_EQ(std::memcmp(culled.images[channel].data_ptr(), reference.images[channel].data_ptr(),
                                          culled.images[channel].numel() * sizeof(float)),
                              0)
                        << "channel=" << channel;
                }
                // Compaction keeps each tile's surviving entries in their original order.
                for (size_t i = 1; i < culled.contributors.size(); ++i) {
                    const auto& previous = culled.contributors[i - 1];
                    const auto& entry = culled.contributors[i];
                    if (entry[0] == previous[0])
                        EXPECT_LT(previous[1], entry[1]);
                }
                for (const auto& entry : culled.contributors) {
                    ASSERT_LT(entry[2], count);
                    EXPECT_NE(entry[3], 0u);
                }
                const auto negative = capture(camera, model, 2, mip, tile);
                EXPECT_NE(culled.last, negative.last);
                EXPECT_TRUE(std::all_of(negative.last.begin(), negative.last.end(), [](unsigned value) { return value == 0; }));
            }
        }
    }

    TEST(FastGSForwardMask, LiveRenderPreservesIdsAboveTwentyFourBits) {
        constexpr size_t count = (size_t{1} << 24) + 3;
        auto means = Tensor::full({count, 3}, -100.0f, Device::CUDA);
        auto rotations = Tensor::zeros({count, 4}, Device::CUDA);
        const float visible_means[] = {-0.05f, 0.0f, 0.0f, 0.05f, 0.0f, 0.1f};
        const float visible_rotations[] = {1, 0, 0, 0, 1, 0, 0, 0};
        ASSERT_EQ(cudaMemcpy(means.ptr<float>() + 3 * (count - 2), visible_means, sizeof(visible_means), cudaMemcpyHostToDevice), cudaSuccess);
        ASSERT_EQ(cudaMemcpy(rotations.ptr<float>() + 4 * (count - 2), visible_rotations, sizeof(visible_rotations), cudaMemcpyHostToDevice), cudaSuccess);
        SplatData model(0, std::move(means), Tensor::full({count, 1, 3}, 0.3f, Device::CUDA),
                        Tensor::zeros({count, 0, 3}, Device::CUDA), Tensor::full({count, 3}, -2.0f, Device::CUDA), std::move(rotations),
                        Tensor::full({count}, 0.0f, Device::CUDA), 1.0f);
        auto translation = Tensor::zeros({3}, Device::CPU);
        translation.ptr<float>()[2] = 3.0f;
        Camera camera(Tensor::eye(3, Device::CUDA), translation.cuda(), 60.0f, 61.0f, 32.0f, 31.0f,
                      Tensor{}, Tensor{}, CameraModelType::PINHOLE, "large_ids", "", std::filesystem::path{}, 65, 63, 0);
        const auto compact = capture(camera, model, 0, false, false);
        const auto reference = capture(camera, model, 1, false, false);
        ASSERT_FALSE(compact.contributors.empty());
        EXPECT_EQ(compact.contributors, reference.contributors);
        EXPECT_EQ(compact.last, reference.last);
        for (const auto& entry : compact.contributors)
            EXPECT_TRUE(entry[2] == count - 2 || entry[2] == count - 1);
    }

    TEST(FastGSForwardMask, HalfWarpBackwardMatchesUnculledGradients) {
        constexpr size_t count = 320;
        constexpr int width = 65;
        constexpr int height = 63;

        auto make_model = []() {
            auto means = Tensor::empty({count, 3}, Device::CPU);
            auto rotations = Tensor::zeros({count, 4}, Device::CPU);
            auto scales = Tensor::full({count, 3}, -3.6f, Device::CPU);
            auto sh0 = Tensor::empty({count, 1, 3}, Device::CPU);
            auto shN = Tensor::empty({count, 15, 3}, Device::CPU);
            auto opacity = Tensor::empty({count}, Device::CPU);
            for (size_t i = 0; i < count; ++i) {
                means.ptr<float>()[3 * i] = (static_cast<int>(i % 16) - 7.5f) * 0.09f;
                means.ptr<float>()[3 * i + 1] = (static_cast<int>((i / 16) % 20) - 9.5f) * 0.08f;
                means.ptr<float>()[3 * i + 2] = 0.002f * static_cast<float>(i);
                rotations.ptr<float>()[4 * i] = 1.0f;
                rotations.ptr<float>()[4 * i + 1] = 0.05f * std::sin(static_cast<float>(i));
                rotations.ptr<float>()[4 * i + 2] = 0.04f * std::cos(static_cast<float>(i));
                if (i % 9 == 0) {
                    scales.ptr<float>()[3 * i] = -2.2f;
                    scales.ptr<float>()[3 * i + 1] = -2.4f;
                }
                for (int c = 0; c < 3; ++c)
                    sh0.ptr<float>()[3 * i + c] = 0.1f + 0.05f * static_cast<float>((i + c) % 7);
                for (size_t k = 0; k < 45; ++k)
                    shN.ptr<float>()[45 * i + k] = 0.01f * std::sin(static_cast<float>(i * 7 + k));
                opacity.ptr<float>()[i] = -0.3f + 0.15f * static_cast<float>(i % 11);
            }
            SplatData model(3, means.cuda(), sh0.cuda(), shN.cuda(), scales.cuda(), rotations.cuda(), opacity.cuda(),
                            1.0f);
            model.set_active_sh_degree(3);
            model._densification_info = Tensor::zeros({2, count}, Device::CUDA);
            return model;
        };

        auto translation = Tensor::zeros({3}, Device::CPU);
        translation.ptr<float>()[2] = 3.0f;
        Camera camera(Tensor::eye(3, Device::CUDA), translation.cuda(), 60.0f, 61.0f, 32.0f, 31.0f,
                      Tensor{}, Tensor{}, CameraModelType::PINHOLE, "bwd_parity", "", std::filesystem::path{},
                      width, height, 0);

        auto grad_image_cpu = Tensor::empty({3, static_cast<size_t>(height), static_cast<size_t>(width)}, Device::CPU);
        for (size_t i = 0; i < grad_image_cpu.numel(); ++i)
            grad_image_cpu.ptr<float>()[i] = 0.01f * std::sin(static_cast<float>(i) * 0.13f);
        const Tensor grad_image = grad_image_cpu.cuda();

        auto error_map_cpu = Tensor::empty({static_cast<size_t>(height), static_cast<size_t>(width)}, Device::CPU);
        auto edge_map_cpu = Tensor::empty({static_cast<size_t>(height), static_cast<size_t>(width)}, Device::CPU);
        for (size_t i = 0; i < error_map_cpu.numel(); ++i) {
            error_map_cpu.ptr<float>()[i] = 0.25f + 0.5f * std::abs(std::cos(static_cast<float>(i) * 0.07f));
            edge_map_cpu.ptr<float>()[i] = 0.1f + 0.9f * std::abs(std::sin(static_cast<float>(i) * 0.11f));
        }
        const Tensor error_map = error_map_cpu.cuda();
        const Tensor edge_map = edge_map_cpu.cuda();
        auto bg = Tensor::full({3}, 0.1f, Device::CUDA);

        struct BackwardCapture {
            std::array<Tensor, 6> param_grads;
            Tensor dens_info;
            Tensor edge_score;
            Tensor blend_peak;
        };

        auto run_backward = [&](const int mode, const bool with_edge) -> BackwardCapture {
            const CullMode culling(mode);
            SplatData model = make_model();
            AdamConfig cfg{.lr = 1e-3f, .beta1 = 0.9, .beta2 = 0.999, .eps = 1e-15, .sh_warmup = false};
            AdamOptimizer opt(model, cfg);
            opt.allocate_gradients();
            for (const auto type : AdamOptimizer::all_param_types())
                (void)opt.get_grad(type);
            opt.zero_grad(0);

            auto fwd = fast_rasterize_forward(camera, model, bg, 0, 0, 0, 0, false, {}, false, false);
            if (!fwd)
                throw std::runtime_error(lfs::format_for_developer(fwd.error()));

            Tensor edge_score = Tensor::zeros({count}, Device::CUDA);
            Tensor blend_peak = Tensor::zeros({count}, Device::CUDA);
            FastGSFusedExtraGradients extra;
            extra.blend_peak = blend_peak.ptr<float>();
            if (with_edge) {
                extra.edge_weight_map = edge_map.ptr<float>();
                extra.edge_score_out = edge_score.ptr<float>();
            }
            fast_rasterize_backward(fwd->second, grad_image, model, opt, {}, error_map,
                                    DensificationType::MRNF, /*iteration=*/1, extra, {}, {},
                                    /*defer_optimizer_step=*/true);
            if (cudaDeviceSynchronize() != cudaSuccess)
                throw std::runtime_error("cudaDeviceSynchronize failed");

            BackwardCapture out;
            const auto types = AdamOptimizer::all_param_types();
            for (size_t i = 0; i < types.size(); ++i)
                out.param_grads[i] = opt.get_grad(types[i]).to(Device::CPU).contiguous();
            out.dens_info = model._densification_info.to(Device::CPU).contiguous();
            out.edge_score = edge_score.to(Device::CPU).contiguous();
            out.blend_peak = blend_peak.to(Device::CPU).contiguous();
            return out;
        };

        auto expect_close = [](const Tensor& a, const Tensor& b, const char* label) {
            ASSERT_EQ(a.numel(), b.numel()) << label;
            const float* pa = a.ptr<float>();
            const float* pb = b.ptr<float>();
            float max_abs = 0.0f;
            float max_rel = 0.0f;
            float max_mag = 0.0f;
            for (size_t i = 0; i < a.numel(); ++i) {
                ASSERT_TRUE(std::isfinite(pa[i]) && std::isfinite(pb[i])) << label << " i=" << i;
                max_mag = std::max(max_mag, std::max(std::abs(pa[i]), std::abs(pb[i])));
            }
            const float rel_floor = std::max(1e-6f, 1e-2f * max_mag);
            for (size_t i = 0; i < a.numel(); ++i) {
                const float diff = std::abs(pa[i] - pb[i]);
                const float mag = std::max(std::abs(pa[i]), std::abs(pb[i]));
                max_abs = std::max(max_abs, diff);
                if (mag > rel_floor)
                    max_rel = std::max(max_rel, diff / mag);
            }
            EXPECT_GT(max_mag, 0.0f) << label << " must be non-trivial";
            EXPECT_LE(max_abs, 1e-5f * max_mag) << label << " max_abs=" << max_abs << " max_mag=" << max_mag;
            EXPECT_LE(max_rel, 1e-4f) << label << " max_rel=" << max_rel;
        };

        constexpr const char* kParamNames[6] = {"means", "sh0", "shN", "scaling", "rotation", "opacity"};
        for (const bool with_edge : {false, true}) {
            SCOPED_TRACE(::testing::Message() << "with_edge=" << with_edge);
            const auto dual_issue = run_backward(0, with_edge);
            const auto reference = run_backward(1, with_edge);
            for (size_t i = 0; i < 6; ++i)
                expect_close(dual_issue.param_grads[i], reference.param_grads[i], kParamNames[i]);
            expect_close(dual_issue.dens_info, reference.dens_info, "densification_info");
            expect_close(dual_issue.blend_peak, reference.blend_peak, "blend_peak");
            if (with_edge)
                expect_close(dual_issue.edge_score, reference.edge_score, "edge_score");
        }
    }

    TEST(FastGSForwardMask, SparseRenderedOnlyBackwardMatchesDenseAndPreservesUntouched) {
        // 320 primitives span two 256-splat bound blocks: half in front of the camera
        // (contributing) and half behind the camera (invisible / untouched).
        constexpr int width = 64;
        constexpr int height = 64;
        constexpr size_t count = 320;

        struct GenericMode {
            const bool previous = fast_lfs::rasterization::force_generic_preprocess_for_testing();
            explicit GenericMode(bool force) {
                fast_lfs::rasterization::set_force_generic_preprocess_for_testing(force);
            }
            ~GenericMode() { fast_lfs::rasterization::set_force_generic_preprocess_for_testing(previous); }
        };

        auto make_model = [&](const bool zero_shN, const bool quantize_shN) {
            auto means = Tensor::zeros({count, 3}, Device::CPU);
            auto scales = Tensor::full({count, 3}, -2.8f, Device::CPU);
            auto rotations = Tensor::zeros({count, 4}, Device::CPU);
            auto sh0 = Tensor::zeros({count, 1, 3}, Device::CPU);
            auto shN = Tensor::zeros({count, 15, 3}, Device::CPU);
            auto opacity = Tensor::zeros({count}, Device::CPU);
            for (size_t i = 0; i < count; ++i) {
                const bool behind = (i % 2 == 1);
                means.ptr<float>()[3 * i] = -0.6f + 0.08f * static_cast<float>((i / 2) % 16);
                means.ptr<float>()[3 * i + 1] = -0.6f + 0.12f * static_cast<float>(((i / 2) / 16) % 10);
                means.ptr<float>()[3 * i + 2] = behind ? -6.0f : (0.1f + 0.005f * static_cast<float>(i % 20));
                rotations.ptr<float>()[4 * i] = 1.0f;
                for (int c = 0; c < 3; ++c)
                    sh0.ptr<float>()[3 * i + c] = 0.2f + 0.04f * static_cast<float>((i + c) % 5);
                if (!zero_shN) {
                    for (size_t k = 0; k < 45; ++k)
                        shN.ptr<float>()[45 * i + k] = 0.05f * std::sin(static_cast<float>(i * 5 + k));
                }
                opacity.ptr<float>()[i] = 0.8f;
            }
            SplatData model(3, means.cuda(), sh0.cuda(), shN.cuda(), scales.cuda(), rotations.cuda(), opacity.cuda(),
                            1.0f);
            model.set_active_sh_degree(3);
            if (quantize_shN)
                lfs::training::sh_value::apply_shN_value_quant(model);
            return model;
        };

        auto translation = Tensor::zeros({3}, Device::CPU);
        translation.ptr<float>()[2] = 3.0f;
        Camera camera(Tensor::eye(3, Device::CUDA), translation.cuda(), 60.0f, 60.0f, 32.0f, 32.0f,
                      Tensor{}, Tensor{}, CameraModelType::PINHOLE, "sparse_bwd", "", std::filesystem::path{},
                      width, height, 0);
        auto grad_image_cpu = Tensor::empty({3, static_cast<size_t>(height), static_cast<size_t>(width)}, Device::CPU);
        for (size_t i = 0; i < grad_image_cpu.numel(); ++i)
            grad_image_cpu.ptr<float>()[i] = 0.02f * std::sin(static_cast<float>(i) * 0.17f);
        const Tensor grad_image = grad_image_cpu.cuda();
        auto bg = Tensor::full({3}, 0.1f, Device::CUDA);

        auto run_steps = [&](const bool force_generic, const bool zero_shN, const bool quantize_shN,
                             const float lr, const int n_steps, const bool check_zero_allocs) {
            const GenericMode mode(force_generic);
            SplatData model = make_model(zero_shN, quantize_shN);
            const Tensor init_means = model.means().to(Device::CPU).contiguous();
            const Tensor init_shN = model.shN_canonical().to(Device::CPU).contiguous();

            AdamConfig cfg{.lr = lr, .beta1 = 0.9, .beta2 = 0.999, .eps = 1e-8, .sh_warmup = false};
            AdamOptimizer opt(model, cfg);
            opt.allocate_gradients();

            Tensor rendered_count = Tensor::zeros({count}, Device::CUDA);
            Tensor scale_loss = Tensor::zeros({1}, Device::CUDA);
            Tensor erank_loss = Tensor::zeros({1}, Device::CUDA);
            Tensor dc_loss = Tensor::zeros({1}, Device::CUDA);
            Tensor sh_rest_loss = Tensor::zeros({1}, Device::CUDA);

            for (int step = 1; step <= n_steps; ++step) {
                auto fwd = fast_rasterize_forward(camera, model, bg, 0, 0, 0, 0, false, {}, false, false);
                if (!fwd)
                    throw std::runtime_error(lfs::format_for_developer(fwd.error()));
                FastGSFusedExtraGradients extra;
                extra.reg_rendered_only = true;
                extra.rendered_count = rendered_count.ptr<float>();
                extra.scale_reg_weight = 0.01f;
                extra.scale_reg_log = true;
                extra.scale_reg_normalizer = 1.0f;
                extra.erank_reg_weight = 0.001f;
                extra.dc_reg_weight = 0.001f;
                extra.sh_rest_reg_weight = 0.1f;
                if (step == n_steps) {
                    scale_loss.zero_();
                    erank_loss.zero_();
                    dc_loss.zero_();
                    sh_rest_loss.zero_();
                    extra.scale_reg_loss_out = scale_loss.ptr<float>();
                    extra.erank_reg_loss_out = erank_loss.ptr<float>();
                    extra.dc_reg_loss_out = dc_loss.ptr<float>();
                    extra.sh_rest_reg_loss_out = sh_rest_loss.ptr<float>();
                }
                const auto snap = alloc_counter::snapshot();
                fast_rasterize_backward(fwd->second, grad_image, model, opt, {}, {},
                                        DensificationType::None, step, extra, {}, {}, false);
                if (check_zero_allocs && step == n_steps) {
                    EXPECT_EQ(alloc_counter::delta_since(snap), 0u);
                }
            }
            if (cudaDeviceSynchronize() != cudaSuccess)
                throw std::runtime_error("cudaDeviceSynchronize failed");

            struct Out {
                Tensor means, sh0, shN, scaling, rotation, opacity, rendered_count, init_means, init_shN;
                Tensor shN_bounds;
                float scale_loss, erank_loss, dc_loss, sh_rest_loss;
            };
            Out out;
            out.means = model.means().to(Device::CPU).contiguous();
            out.sh0 = model.sh0().to(Device::CPU).contiguous();
            out.shN = model.shN_canonical().to(Device::CPU).contiguous();
            out.scaling = model.scaling_raw().to(Device::CPU).contiguous();
            out.rotation = model.rotation_raw().to(Device::CPU).contiguous();
            out.opacity = model.opacity_raw().to(Device::CPU).contiguous();
            out.rendered_count = rendered_count.to(Device::CPU).contiguous();
            out.init_means = init_means;
            out.init_shN = init_shN;
            if (model.shN_value_quantized())
                out.shN_bounds = model.shN_value_bounds().to(Device::CPU).contiguous();
            out.scale_loss = scale_loss.to(Device::CPU).item<float>();
            out.erank_loss = erank_loss.to(Device::CPU).item<float>();
            out.dc_loss = dc_loss.to(Device::CPU).item<float>();
            out.sh_rest_loss = sh_rest_loss.to(Device::CPU).item<float>();
            return out;
        };

        // 1. Step-1 parity against dense generic path on contributing splats, while
        //    untouched (behind-camera) splats stay exact and zero-alloc holds in steady state.
        const auto sparse = run_steps(/*force_generic=*/false, /*zero_shN=*/true, /*quantize_shN=*/true,
                                      /*lr=*/5e-3f, /*n_steps=*/3, /*check_zero_allocs=*/true);
        const auto dense = run_steps(/*force_generic=*/true, /*zero_shN=*/true, /*quantize_shN=*/true,
                                     /*lr=*/5e-3f, /*n_steps=*/1, /*check_zero_allocs=*/false);
        const auto sparse1 = run_steps(/*force_generic=*/false, /*zero_shN=*/true, /*quantize_shN=*/true,
                                       /*lr=*/5e-3f, /*n_steps=*/1, /*check_zero_allocs=*/false);

        int contributing = 0;
        int untouched = 0;
        for (size_t i = 0; i < count; ++i) {
            const float rc_s = sparse1.rendered_count.ptr<float>()[i];
            const float rc_d = dense.rendered_count.ptr<float>()[i];
            EXPECT_FLOAT_EQ(rc_s, rc_d) << "rendered_count mismatch at " << i;
            if (rc_s > 0.0f) {
                ++contributing;
                for (int c = 0; c < 3; ++c) {
                    EXPECT_NEAR(sparse1.means.ptr<float>()[3 * i + c], dense.means.ptr<float>()[3 * i + c], 2e-5f);
                    EXPECT_NEAR(sparse1.sh0.ptr<float>()[3 * i + c], dense.sh0.ptr<float>()[3 * i + c], 2e-5f);
                    EXPECT_NEAR(sparse1.scaling.ptr<float>()[3 * i + c], dense.scaling.ptr<float>()[3 * i + c], 2e-5f);
                }
                for (int c = 0; c < 4; ++c)
                    EXPECT_NEAR(sparse1.rotation.ptr<float>()[4 * i + c], dense.rotation.ptr<float>()[4 * i + c], 2e-5f);
                EXPECT_NEAR(sparse1.opacity.ptr<float>()[i], dense.opacity.ptr<float>()[i], 2e-5f);
                for (size_t k = 0; k < 45; ++k) {
                    EXPECT_NEAR(sparse1.shN.ptr<float>()[45 * i + k], dense.shN.ptr<float>()[45 * i + k], 1e-4f);
                }
            } else {
                ++untouched;
                for (int c = 0; c < 3; ++c)
                    EXPECT_FLOAT_EQ(sparse.means.ptr<float>()[3 * i + c], sparse.init_means.ptr<float>()[3 * i + c]);
                // Untouched zero-initialized q16 SH-rest cells stay at zero (within float32 ULP of decode).
                for (size_t k = 0; k < 45; ++k)
                    EXPECT_NEAR(sparse.shN.ptr<float>()[45 * i + k], 0.0f, 1e-8f);
            }
        }
        EXPECT_GT(contributing, 0);
        EXPECT_GT(untouched, 0);
        EXPECT_TRUE(std::isfinite(sparse.scale_loss) && std::isfinite(sparse.erank_loss));
        EXPECT_TRUE(std::isfinite(sparse.dc_loss) && std::isfinite(sparse.sh_rest_loss));

        // 2. Large LR forces q16 SH-value overflow beyond [-1, +1] and triggers post-rebound
        //    without corrupting untouched non-zero SH-rest splats in the same 256-splat block.
        const auto overflow = run_steps(/*force_generic=*/false, /*zero_shN=*/false, /*quantize_shN=*/true,
                                        /*lr=*/1.5f, /*n_steps=*/2, /*check_zero_allocs=*/false);
        ASSERT_TRUE(overflow.shN_bounds.is_valid());
        bool widened_past_one = false;
        for (size_t b = 0; b < overflow.shN_bounds.shape()[0]; ++b) {
            const float lo = overflow.shN_bounds.ptr<float>()[2 * b + 0];
            const float hi = overflow.shN_bounds.ptr<float>()[2 * b + 1];
            if (lo < -1.5f && hi > 1.5f)
                widened_past_one = true;
        }
        EXPECT_TRUE(widened_past_one) << "expected q16 sh_value_bounds to expand after overflow";
        for (size_t i = 0; i < count; ++i) {
            if (overflow.rendered_count.ptr<float>()[i] == 0.0f) {
                for (size_t k = 0; k < 45; ++k) {
                    EXPECT_NEAR(overflow.shN.ptr<float>()[45 * i + k],
                                overflow.init_shN.ptr<float>()[45 * i + k], 2e-4f)
                        << "untouched q16 SH-rest corrupted at prim=" << i << " k=" << k;
                }
            }
        }
    }

    TEST(FastGSForwardMask, SparseLazyCatchUpMatchesDenseAcrossAlternatingViews) {
        constexpr int width = 64;
        constexpr int height = 64;
        constexpr size_t count = 320;

        struct GenericMode {
            const bool previous = fast_lfs::rasterization::force_generic_preprocess_for_testing();
            explicit GenericMode(bool force) {
                fast_lfs::rasterization::set_force_generic_preprocess_for_testing(force);
            }
            ~GenericMode() { fast_lfs::rasterization::set_force_generic_preprocess_for_testing(previous); }
        };

        auto make_model = [&](const int sh_degree) {
            auto means = Tensor::zeros({count, 3}, Device::CPU);
            auto scales = Tensor::zeros({count, 3}, Device::CPU);
            auto rotations = Tensor::zeros({count, 4}, Device::CPU);
            auto sh0 = Tensor::zeros({count, 1, 3}, Device::CPU);
            auto shN = Tensor::zeros({count, 15, 3}, Device::CPU);
            auto opacity = Tensor::full({count}, 0.8f, Device::CPU);
            for (size_t i = 0; i < count; ++i) {
                const bool group_b = (i >= count / 2);
                const size_t local = i % (count / 2);
                means.ptr<float>()[3 * i] = (group_b ? 8.0f : -8.0f) - 0.5f + 0.07f * static_cast<float>(local % 16);
                means.ptr<float>()[3 * i + 1] = -0.5f + 0.1f * static_cast<float>((local / 16) % 10);
                means.ptr<float>()[3 * i + 2] = 0.1f + 0.005f * static_cast<float>(local % 20);
                // Non-degenerate anisotropic scales so rotation has genuine gradients
                // (exact spheres have zero analytical rotation gradient -> 0/0 noise at eps=1e-15).
                scales.ptr<float>()[3 * i + 0] = -2.8f;
                scales.ptr<float>()[3 * i + 1] = -2.5f;
                scales.ptr<float>()[3 * i + 2] = -3.1f;
                rotations.ptr<float>()[4 * i] = 1.0f;
                for (int c = 0; c < 3; ++c)
                    sh0.ptr<float>()[3 * i + c] = (i % 11 == 0) ? 2.1f : (0.2f + 0.04f * static_cast<float>((i + c) % 5));
                if (sh_degree > 0) {
                    for (size_t k = 0; k < 45; ++k)
                        shN.ptr<float>()[45 * i + k] = 0.08f * std::sin(static_cast<float>(i * 7 + k));
                }
            }
            SplatData model(sh_degree, means.cuda(), sh0.cuda(),
                            sh_degree > 0 ? shN.cuda() : Tensor::empty({count, 0, 3}, Device::CUDA),
                            scales.cuda(), rotations.cuda(), opacity.cuda(), 1.0f);
            model.set_active_sh_degree(sh_degree);
            if (sh_degree > 0)
                lfs::training::sh_value::apply_shN_value_quant(model);
            return model;
        };

        auto make_cam = [&](const float tx, const char* name) {
            auto translation = Tensor::zeros({3}, Device::CPU);
            translation.ptr<float>()[0] = tx;
            translation.ptr<float>()[2] = 3.0f;
            return Camera(Tensor::eye(3, Device::CUDA), translation.cuda(), 60.0f, 60.0f, 32.0f, 32.0f,
                          Tensor{}, Tensor{}, CameraModelType::PINHOLE, name, "", std::filesystem::path{},
                          width, height, 0);
        };
        Camera cam_a = make_cam(8.0f, "cam_a");
        Camera cam_b = make_cam(-8.0f, "cam_b");

        auto grad_image_cpu = Tensor::empty({3, static_cast<size_t>(height), static_cast<size_t>(width)}, Device::CPU);
        for (size_t i = 0; i < grad_image_cpu.numel(); ++i)
            grad_image_cpu.ptr<float>()[i] = 0.015f * std::sin(static_cast<float>(i) * 0.19f);
        const Tensor grad_image = grad_image_cpu.cuda();
        auto bg = Tensor::full({3}, 0.1f, Device::CUDA);

        // Schedule: cam_a (1), cam_b (2..6 -> 5 missed steps for group A), cam_a (7), cam_b (8), cam_a (9).
        constexpr std::array<int, 9> kCamSchedule = {0, 1, 1, 1, 1, 1, 0, 1, 0};

        struct RunOut {
            Tensor means, sh0, shN, scaling, rotation, opacity, rendered_a;
        };
        auto run_schedule = [&](const bool force_generic, const int sh_degree) -> RunOut {
            const GenericMode mode(force_generic);
            SplatData model = make_model(sh_degree);
            AdamConfig cfg{.lr = 1e-3f, .beta1 = 0.9, .beta2 = 0.999, .eps = 1e-15, .sh_warmup = false};
            AdamOptimizer opt(model, cfg);
            opt.allocate_gradients();

            Tensor rendered_a = Tensor::zeros({count}, Device::CUDA);
            for (size_t s = 0; s < kCamSchedule.size(); ++s) {
                Camera& cam = (kCamSchedule[s] == 0) ? cam_a : cam_b;
                auto fwd = fast_rasterize_forward(cam, model, bg, 0, 0, 0, 0, false, {}, false, false);
                if (!fwd)
                    throw std::runtime_error(lfs::format_for_developer(fwd.error()));
                FastGSFusedExtraGradients extra;
                extra.reg_rendered_only = true;
                extra.dc_reg_weight = 0.001f;
                extra.sh_rest_reg_weight = sh_degree > 0 ? 0.375f : 0.0f;
                if (s == 0)
                    extra.rendered_count = rendered_a.ptr<float>();
                fast_rasterize_backward(fwd->second, grad_image, model, opt, {}, {},
                                        DensificationType::None, static_cast<int>(s + 1), extra, {}, {}, false);
            }
            if (!force_generic) {
                const float sh_rest_w = sh_degree > 0 ? 0.375f : 0.0f;
                fast_rasterize_flush_lazy_adam(model, opt, static_cast<int>(kCamSchedule.size()), 0.001f, sh_rest_w);
                // Second flush at the same step must be an exact no-op.
                fast_rasterize_flush_lazy_adam(model, opt, static_cast<int>(kCamSchedule.size()), 0.001f, sh_rest_w);
            }
            if (cudaDeviceSynchronize() != cudaSuccess)
                throw std::runtime_error("cudaDeviceSynchronize failed");

            return RunOut{
                .means = model.means().to(Device::CPU).contiguous(),
                .sh0 = model.sh0().to(Device::CPU).contiguous(),
                .shN = sh_degree > 0 ? model.shN_canonical().to(Device::CPU).contiguous() : Tensor{},
                .scaling = model.scaling_raw().to(Device::CPU).contiguous(),
                .rotation = model.rotation_raw().to(Device::CPU).contiguous(),
                .opacity = model.opacity_raw().to(Device::CPU).contiguous(),
                .rendered_a = rendered_a.to(Device::CPU).contiguous(),
            };
        };

        // 1. SH-degree 0: geometry + sh0 (including invisible dc_reg + momentum decay) must match
        //    dense tightly across alternating views.
        {
            const auto sparse0 = run_schedule(/*force_generic=*/false, /*sh_degree=*/0);
            const auto dense0 = run_schedule(/*force_generic=*/true, /*sh_degree=*/0);
            float m_diff = 0.0f, sh0_diff = 0.0f, s_diff = 0.0f, r_diff = 0.0f, o_diff = 0.0f;
            for (size_t i = 0; i < count; ++i) {
                for (int c = 0; c < 3; ++c) {
                    m_diff = std::max(m_diff, std::abs(sparse0.means.ptr<float>()[3 * i + c] - dense0.means.ptr<float>()[3 * i + c]));
                    sh0_diff = std::max(sh0_diff, std::abs(sparse0.sh0.ptr<float>()[3 * i + c] - dense0.sh0.ptr<float>()[3 * i + c]));
                    s_diff = std::max(s_diff, std::abs(sparse0.scaling.ptr<float>()[3 * i + c] - dense0.scaling.ptr<float>()[3 * i + c]));
                }
                for (int c = 0; c < 4; ++c)
                    r_diff = std::max(r_diff, std::abs(sparse0.rotation.ptr<float>()[4 * i + c] - dense0.rotation.ptr<float>()[4 * i + c]));
                o_diff = std::max(o_diff, std::abs(sparse0.opacity.ptr<float>()[i] - dense0.opacity.ptr<float>()[i]));
            }
            EXPECT_LT(m_diff, 5e-5f) << "sh0-only max_means_diff=" << m_diff;
            EXPECT_LT(sh0_diff, 5e-5f) << "sh0-only max_sh0_diff=" << sh0_diff;
            EXPECT_LT(s_diff, 5e-5f) << "sh0-only max_scale_diff=" << s_diff;
            EXPECT_LT(r_diff, 5e-5f) << "sh0-only max_rot_diff=" << r_diff;
            EXPECT_LT(o_diff, 5e-5f) << "sh0-only max_opa_diff=" << o_diff;
        }

        // 2. SH-degree 3: shN uses lazy catch-up across the 5 missed steps while geometry + sh0
        //    stay exact every step.
        const auto sparse = run_schedule(/*force_generic=*/false, /*sh_degree=*/3);
        const auto dense = run_schedule(/*force_generic=*/true, /*sh_degree=*/3);

        int checked_a = 0;
        float max_means_diff = 0.0f;
        float max_sh0_diff = 0.0f;
        float max_shN_diff = 0.0f;
        float max_scale_diff = 0.0f;
        float max_rot_diff = 0.0f;
        float max_opa_diff = 0.0f;
        for (size_t i = 0; i < count / 2; ++i) {
            if (sparse.rendered_a.ptr<float>()[i] <= 0.0f)
                continue;
            ++checked_a;
            for (int c = 0; c < 3; ++c) {
                max_means_diff = std::max(max_means_diff,
                                          std::abs(sparse.means.ptr<float>()[3 * i + c] - dense.means.ptr<float>()[3 * i + c]));
                max_sh0_diff = std::max(max_sh0_diff,
                                        std::abs(sparse.sh0.ptr<float>()[3 * i + c] - dense.sh0.ptr<float>()[3 * i + c]));
                max_scale_diff = std::max(max_scale_diff,
                                          std::abs(sparse.scaling.ptr<float>()[3 * i + c] - dense.scaling.ptr<float>()[3 * i + c]));
            }
            for (int c = 0; c < 4; ++c) {
                max_rot_diff = std::max(max_rot_diff,
                                        std::abs(sparse.rotation.ptr<float>()[4 * i + c] - dense.rotation.ptr<float>()[4 * i + c]));
            }
            max_opa_diff = std::max(max_opa_diff,
                                    std::abs(sparse.opacity.ptr<float>()[i] - dense.opacity.ptr<float>()[i]));
            for (size_t k = 0; k < 45; ++k) {
                max_shN_diff = std::max(max_shN_diff,
                                        std::abs(sparse.shN.ptr<float>()[45 * i + k] - dense.shN.ptr<float>()[45 * i + k]));
            }
        }
        EXPECT_GT(checked_a, 20);
        // Geometry + sh0 are updated every step (minor second-order coupling through shN color
        // when group A re-enters view at step 7); shN uses closed-form lazy catch-up.
        EXPECT_LT(max_means_diff, 1e-3f) << "max_means_diff=" << max_means_diff;
        EXPECT_LT(max_sh0_diff, 1e-3f) << "max_sh0_diff=" << max_sh0_diff;
        EXPECT_LT(max_scale_diff, 1.5e-3f) << "max_scale_diff=" << max_scale_diff;
        EXPECT_LT(max_rot_diff, 1e-3f) << "max_rot_diff=" << max_rot_diff;
        EXPECT_LT(max_opa_diff, 2e-3f) << "max_opa_diff=" << max_opa_diff;
        EXPECT_LT(max_shN_diff, 3e-3f) << "max_shN_diff=" << max_shN_diff;
    }

    TEST(FastGSForwardMask, SubtileBackwardMatchesUnculledReferenceAcrossModes) {
        constexpr int width = 77;
        constexpr int height = 61;
        constexpr size_t count = 320;

        auto make_model = [&]() {
            auto means = Tensor::zeros({count, 3}, Device::CPU);
            auto scales = Tensor::zeros({count, 3}, Device::CPU);
            auto rotations = Tensor::zeros({count, 4}, Device::CPU);
            auto sh0 = Tensor::zeros({count, 1, 3}, Device::CPU);
            auto shN = Tensor::zeros({count, 0, 3}, Device::CPU);
            auto opacity = Tensor::zeros({count}, Device::CPU);
            for (size_t i = 0; i < count; ++i) {
                const float u = static_cast<float>(i % 20) / 19.0f;
                const float v = static_cast<float>((i / 20) % 16) / 15.0f;
                means.ptr<float>()[3 * i + 0] = (u - 0.5f) * 2.4f;
                means.ptr<float>()[3 * i + 1] = (v - 0.5f) * 2.0f;
                means.ptr<float>()[3 * i + 2] = 0.05f * static_cast<float>(i % 13);
                // Mix small single-subtile splats and large multi-subtile splats.
                const float s_base = (i % 7 == 0) ? -1.1f : -2.2f;
                scales.ptr<float>()[3 * i + 0] = s_base;
                scales.ptr<float>()[3 * i + 1] = s_base + 0.3f;
                scales.ptr<float>()[3 * i + 2] = s_base - 0.25f;
                rotations.ptr<float>()[4 * i + 0] = 1.0f;
                rotations.ptr<float>()[4 * i + 1] = 0.15f * std::sin(static_cast<float>(i));
                rotations.ptr<float>()[4 * i + 2] = 0.12f * std::cos(static_cast<float>(i));
                for (int c = 0; c < 3; ++c) {
                    const bool overbright = (i % 17 == 0) && (static_cast<size_t>(c) == (i % 3));
                    sh0.ptr<float>()[3 * i + c] = overbright ? 12.0f : (0.25f + 0.1f * static_cast<float>((i + c) % 6));
                }
                opacity.ptr<float>()[i] = 0.6f + 0.2f * static_cast<float>(i % 5);
            }
            return SplatData(0, means.cuda(), sh0.cuda(), shN.cuda(), scales.cuda(), rotations.cuda(), opacity.cuda(),
                             1.0f);
        };

        auto translation = Tensor::zeros({3}, Device::CPU);
        translation.ptr<float>()[2] = 3.2f;
        Camera camera(Tensor::eye(3, Device::CUDA), translation.cuda(), 90.0f, 90.0f,
                      width * 0.5f, height * 0.5f, Tensor{}, Tensor{}, CameraModelType::PINHOLE,
                      "quad_bwd", "", std::filesystem::path{}, width, height, 0);

        auto grad_cpu = Tensor::empty({3, static_cast<size_t>(height), static_cast<size_t>(width)}, Device::CPU);
        auto err_cpu = Tensor::empty({static_cast<size_t>(height), static_cast<size_t>(width)}, Device::CPU);
        auto edge_cpu = Tensor::empty({static_cast<size_t>(height), static_cast<size_t>(width)}, Device::CPU);
        for (size_t i = 0; i < grad_cpu.numel(); ++i)
            grad_cpu.ptr<float>()[i] = 0.03f * std::sin(static_cast<float>(i) * 0.13f);
        for (size_t i = 0; i < err_cpu.numel(); ++i) {
            err_cpu.ptr<float>()[i] = 0.5f + 0.4f * std::cos(static_cast<float>(i) * 0.11f);
            edge_cpu.ptr<float>()[i] = 0.2f + 0.8f * std::abs(std::sin(static_cast<float>(i) * 0.07f));
        }
        const Tensor grad_image = grad_cpu.cuda();
        const Tensor err_map = err_cpu.cuda();
        const Tensor edge_map = edge_cpu.cuda();
        Tensor bg = Tensor::full({3}, 0.15f, Device::CUDA);

        struct Out {
            Tensor means, sh0, scaling, rotation, opacity, dens, blend_peak, edge_score;
        };
        auto run_case = [&](const int bwd_mode, const DensificationType dens_type,
                            const bool use_blend_peak, const bool use_edge) -> Out {
            fast_lfs::rasterization::set_warp_cull_mode_for_testing(0);
            SplatData model = make_model();
            if (dens_type == DensificationType::MRNF)
                model._densification_info = Tensor::zeros({2, count}, Device::CUDA);
            else if (dens_type == DensificationType::MCMC)
                model._densification_info = Tensor::zeros({count}, Device::CUDA);
            auto fwd = fast_rasterize_forward(camera, model, bg, 0, 0, 0, 0, false, {}, false, false);
            if (!fwd)
                throw std::runtime_error(lfs::format_for_developer(fwd.error()));

            AdamConfig cfg{.lr = 1e-2f, .beta1 = 0.9, .beta2 = 0.999, .eps = 1e-8, .sh_warmup = false};
            AdamOptimizer opt(model, cfg);
            opt.allocate_gradients();
            for (const auto type : AdamOptimizer::all_param_types())
                (void)opt.get_grad(type);
            opt.zero_grad(0);

            Tensor blend_peak = use_blend_peak ? Tensor::zeros({count}, Device::CUDA) : Tensor{};
            Tensor edge_score = use_edge ? Tensor::zeros({count}, Device::CUDA) : Tensor{};
            FastGSFusedExtraGradients extra;
            extra.blend_peak = use_blend_peak ? blend_peak.ptr<float>() : nullptr;
            extra.edge_weight_map = use_edge ? edge_map.ptr<float>() : nullptr;
            extra.edge_score_out = use_edge ? edge_score.ptr<float>() : nullptr;

            fast_lfs::rasterization::set_warp_cull_mode_for_testing(bwd_mode);
            fast_rasterize_backward(fwd->second, grad_image, model, opt, {},
                                    dens_type != DensificationType::None ? err_map : Tensor{},
                                    dens_type, 1, extra, {}, {}, /*defer_optimizer_step=*/true);
            fast_lfs::rasterization::set_warp_cull_mode_for_testing(0);
            if (cudaDeviceSynchronize() != cudaSuccess)
                throw std::runtime_error("cudaDeviceSynchronize failed");

            return Out{
                .means = opt.get_grad(ParamType::Means).to(Device::CPU).contiguous(),
                .sh0 = opt.get_grad(ParamType::Sh0).to(Device::CPU).contiguous(),
                .scaling = opt.get_grad(ParamType::Scaling).to(Device::CPU).contiguous(),
                .rotation = opt.get_grad(ParamType::Rotation).to(Device::CPU).contiguous(),
                .opacity = opt.get_grad(ParamType::Opacity).to(Device::CPU).contiguous(),
                .dens = model._densification_info.is_valid()
                            ? model._densification_info.to(Device::CPU).contiguous()
                            : Tensor{},
                .blend_peak = use_blend_peak ? blend_peak.to(Device::CPU).contiguous() : Tensor{},
                .edge_score = use_edge ? edge_score.to(Device::CPU).contiguous() : Tensor{},
            };
        };

        auto expect_norm_close = [](const Tensor& a, const Tensor& b, const float rel_tol, const char* tag) {
            ASSERT_EQ(a.numel(), b.numel()) << tag;
            float max_diff = 0.0f;
            float max_mag = 0.0f;
            for (size_t i = 0; i < a.numel(); ++i) {
                max_diff = std::max(max_diff, std::abs(a.ptr<float>()[i] - b.ptr<float>()[i]));
                max_mag = std::max(max_mag, std::max(std::abs(a.ptr<float>()[i]), std::abs(b.ptr<float>()[i])));
            }
            EXPECT_LE(max_diff, rel_tol * std::max(1.0f, max_mag)) << tag << " max_diff=" << max_diff << " max_mag=" << max_mag;
        };

        for (const auto dens_type : {DensificationType::MRNF, DensificationType::MCMC, DensificationType::None}) {
            for (const bool use_bp : {true, false}) {
                for (const bool use_edge : {true, false}) {
                    SCOPED_TRACE(::testing::Message()
                                 << "dens=" << static_cast<int>(dens_type) << " bp=" << use_bp << " edge=" << use_edge);
                    const auto sub = run_case(/*bwd_mode=*/0, dens_type, use_bp, use_edge);
                    const auto ref = run_case(/*bwd_mode=*/1, dens_type, use_bp, use_edge);
                    expect_norm_close(sub.means, ref.means, 1e-5f, "means");
                    expect_norm_close(sub.sh0, ref.sh0, 1e-5f, "sh0");
                    expect_norm_close(sub.scaling, ref.scaling, 1e-5f, "scaling");
                    expect_norm_close(sub.rotation, ref.rotation, 1e-5f, "rotation");
                    expect_norm_close(sub.opacity, ref.opacity, 1e-5f, "opacity");
                    if (dens_type != DensificationType::None) {
                        expect_norm_close(sub.dens, ref.dens, 5e-6f, "dens");
                    }
                    if (use_bp) {
                        expect_norm_close(sub.blend_peak, ref.blend_peak, 1e-6f, "blend_peak");
                    }
                    if (use_edge) {
                        expect_norm_close(sub.edge_score, ref.edge_score, 5e-6f, "edge_score");
                    }
                }
            }
        }
    }
} // namespace
