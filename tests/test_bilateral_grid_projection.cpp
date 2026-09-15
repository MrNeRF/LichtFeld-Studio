/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "components/bilateral_grid.hpp"
#include "core/tensor.hpp"
#include "tensor_hardening_test_utils.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <gtest/gtest.h>
#include <sstream>
#include <string>
#include <vector>

namespace {

    using lfs::core::Device;
    using lfs::core::Tensor;
    using lfs::training::BilateralGrid;
    using lfs::training::BilateralGridParameterization;

    class BilateralGridProjectionTest : public tensor_hardening::CudaTest {};

    std::vector<float> cpu_copy(const Tensor& tensor) {
        return tensor.cpu().contiguous().to_vector();
    }

    float affine_identity(const int channel) {
        return (channel == 0 || channel == 5 || channel == 10) ? 1.0f : 0.0f;
    }

    TEST_F(BilateralGridProjectionTest, AffineDatasetMeanEqualsIdentityAndPreservesResidual) {
        BilateralGrid grid(2, 3, 3, 2, 20);
        ASSERT_EQ(grid.channels(), 12);
        ASSERT_EQ(grid.grids().shape(), (lfs::core::TensorShape{2, 12, 2, 3, 3}));

        auto& grids = grid.grids();
        std::vector<float> host = cpu_copy(grids);
        const int n = 2, c = 12, l = 2, h = 3, w = 3;
        const int cells = l * h * w;
        for (int ni = 0; ni < n; ++ni) {
            for (int ci = 0; ci < c; ++ci) {
                for (int cell = 0; cell < cells; ++cell) {
                    const int idx = ((ni * c + ci) * cells) + cell;
                    host[static_cast<size_t>(idx)] += 0.25f * static_cast<float>(ci + 1) +
                                                      0.01f * static_cast<float>(cell);
                }
            }
        }
        grids.copy_from(Tensor::from_vector(host, grids.shape(), Device::CUDA));

        std::vector<float> means(static_cast<size_t>(c), 0.0f);
        for (int ni = 0; ni < n; ++ni) {
            for (int ci = 0; ci < c; ++ci) {
                for (int cell = 0; cell < cells; ++cell) {
                    const int idx = ((ni * c + ci) * cells) + cell;
                    means[static_cast<size_t>(ci)] += host[static_cast<size_t>(idx)];
                }
            }
        }
        const float denom = static_cast<float>(n * cells);
        std::vector<float> residual = host;
        for (int ni = 0; ni < n; ++ni) {
            for (int ci = 0; ci < c; ++ci) {
                const float mean = means[static_cast<size_t>(ci)] / denom;
                for (int cell = 0; cell < cells; ++cell) {
                    const int idx = ((ni * c + ci) * cells) + cell;
                    residual[static_cast<size_t>(idx)] -= mean;
                }
            }
        }

        grid.project_mean(false);
        const std::vector<float> after = cpu_copy(grids);
        const std::vector<float> offset = cpu_copy(grid.shared_offset());
        ASSERT_EQ(offset.size(), static_cast<size_t>(c));

        std::vector<float> after_means(static_cast<size_t>(c), 0.0f);
        for (int ni = 0; ni < n; ++ni) {
            for (int ci = 0; ci < c; ++ci) {
                for (int cell = 0; cell < cells; ++cell) {
                    const int idx = ((ni * c + ci) * cells) + cell;
                    after_means[static_cast<size_t>(ci)] += after[static_cast<size_t>(idx)];
                }
            }
        }
        for (int ci = 0; ci < c; ++ci) {
            EXPECT_NEAR(after_means[static_cast<size_t>(ci)] / denom + offset[static_cast<size_t>(ci)],
                        affine_identity(ci), 1e-5f)
                << "channel " << ci;
        }

        for (int ni = 0; ni < n; ++ni) {
            for (int ci = 0; ci < c; ++ci) {
                const float mean = after_means[static_cast<size_t>(ci)] / denom;
                for (int cell = 0; cell < cells; ++cell) {
                    const int idx = ((ni * c + ci) * cells) + cell;
                    EXPECT_NEAR(after[static_cast<size_t>(idx)] - mean,
                                residual[static_cast<size_t>(idx)], 1e-6f);
                }
            }
        }
    }

    TEST_F(BilateralGridProjectionTest, ExposureChromaPerImageMeansAreZero) {
        BilateralGrid grid(2, 2, 2, 2, 20, {}, BilateralGridParameterization::ExposureChroma);
        ASSERT_EQ(grid.channels(), 9);
        auto& grids = grid.grids();
        std::vector<float> host = cpu_copy(grids);
        const int n = 2, c = 9, cells = 8;
        for (int ni = 0; ni < n; ++ni) {
            for (int ci = 0; ci < c; ++ci) {
                for (int cell = 0; cell < cells; ++cell) {
                    const int idx = ((ni * c + ci) * cells) + cell;
                    host[static_cast<size_t>(idx)] = 0.1f * static_cast<float>(ni + 1) +
                                                     0.02f * static_cast<float>(ci) +
                                                     0.001f * static_cast<float>(cell);
                }
            }
        }
        grids.copy_from(Tensor::from_vector(host, grids.shape(), Device::CUDA));
        grid.project_mean(true);
        const std::vector<float> after = cpu_copy(grids);
        for (int ni = 0; ni < n; ++ni) {
            for (int ci = 0; ci < c; ++ci) {
                float mean = 0.0f;
                for (int cell = 0; cell < cells; ++cell) {
                    const int idx = ((ni * c + ci) * cells) + cell;
                    mean += after[static_cast<size_t>(idx)];
                }
                EXPECT_NEAR(mean / static_cast<float>(cells), 0.0f, 1e-6f)
                    << "image " << ni << " channel " << ci;
            }
        }
    }

    float cpu_tv(const std::vector<float>& grids, int n, int c, int l, int h, int w) {
        float sum = 0.0f;
        const int spatial = l * h * w;
        auto at = [&](int ni, int ci, int li, int hi, int wi) {
            return grids[static_cast<size_t>((((ni * c + ci) * l + li) * h + hi) * w + wi)];
        };
        for (int ni = 0; ni < n; ++ni) {
            for (int ci = 0; ci < c; ++ci) {
                for (int li = 0; li < l; ++li) {
                    for (int hi = 0; hi < h; ++hi) {
                        for (int wi = 0; wi < w; ++wi) {
                            const float val = at(ni, ci, li, hi, wi);
                            if (wi > 0) {
                                const float diff = val - at(ni, ci, li, hi, wi - 1);
                                sum += diff * diff / static_cast<float>(l * h * (w - 1));
                            }
                            if (hi > 0) {
                                const float diff = val - at(ni, ci, li, hi - 1, wi);
                                sum += diff * diff / static_cast<float>(l * (h - 1) * w);
                            }
                            if (li > 0) {
                                const float diff = val - at(ni, ci, li - 1, hi, wi);
                                sum += diff * diff / static_cast<float>((l - 1) * h * w);
                            }
                        }
                    }
                }
            }
        }
        (void)spatial;
        return sum / static_cast<float>(c * n);
    }

    std::vector<float> cpu_tv_grad(const std::vector<float>& grids, float grad, int n, int c, int l, int h, int w) {
        std::vector<float> out(grids.size(), 0.0f);
        const float s = 2.0f * grad / static_cast<float>(c * n);
        const float sx = s / static_cast<float>(l * h * (w - 1));
        const float sy = s / static_cast<float>(l * (h - 1) * w);
        const float sz = s / static_cast<float>((l - 1) * h * w);
        auto at = [&](int ni, int ci, int li, int hi, int wi) {
            return grids[static_cast<size_t>((((ni * c + ci) * l + li) * h + hi) * w + wi)];
        };
        auto ref = [&](int ni, int ci, int li, int hi, int wi) -> float& {
            return out[static_cast<size_t>((((ni * c + ci) * l + li) * h + hi) * w + wi)];
        };
        for (int ni = 0; ni < n; ++ni) {
            for (int ci = 0; ci < c; ++ci) {
                for (int li = 0; li < l; ++li) {
                    for (int hi = 0; hi < h; ++hi) {
                        for (int wi = 0; wi < w; ++wi) {
                            const float val = at(ni, ci, li, hi, wi);
                            float half_grad = 0.0f;
                            if (wi > 0)
                                half_grad += (val - at(ni, ci, li, hi, wi - 1)) * sx;
                            if (wi < w - 1)
                                half_grad += (val - at(ni, ci, li, hi, wi + 1)) * sx;
                            if (hi > 0)
                                half_grad += (val - at(ni, ci, li, hi - 1, wi)) * sy;
                            if (hi < h - 1)
                                half_grad += (val - at(ni, ci, li, hi + 1, wi)) * sy;
                            if (li > 0)
                                half_grad += (val - at(ni, ci, li - 1, hi, wi)) * sz;
                            if (li < l - 1)
                                half_grad += (val - at(ni, ci, li + 1, hi, wi)) * sz;
                            ref(ni, ci, li, hi, wi) = half_grad;
                        }
                    }
                }
            }
        }
        return out;
    }

    TEST_F(BilateralGridProjectionTest, TvC12MatchesCpuReference) {
        BilateralGrid grid(2, 3, 3, 3, 20);
        auto& grids = grid.grids();
        std::vector<float> host = cpu_copy(grids);
        for (size_t i = 0; i < host.size(); ++i) {
            host[i] += 0.05f * std::sin(0.17f * static_cast<float>(i));
        }
        grids.copy_from(Tensor::from_vector(host, grids.shape(), Device::CUDA));

        const float gpu_tv = grid.tv_loss_gpu().cpu().item<float>();
        const float cpu = cpu_tv(host, 2, 12, 3, 3, 3);
        EXPECT_NEAR(gpu_tv, cpu, 1e-5f);

        constexpr float kWeight = 1.0f;
        const auto cpu_g = cpu_tv_grad(host, kWeight, 2, 12, 3, 3, 3);
        const int slice = 12 * 3 * 3 * 3;
        for (int image = 0; image < 2; ++image) {
            grid.zero_grad();
            grid.tv_backward(kWeight, image);
            const std::vector<float> gpu_grad = cpu_copy(grid.grad_slice());
            ASSERT_EQ(gpu_grad.size(), static_cast<size_t>(slice));
            for (int i = 0; i < slice; ++i) {
                EXPECT_NEAR(gpu_grad[static_cast<size_t>(i)],
                            cpu_g[static_cast<size_t>(image * slice + i)], 1e-5f)
                    << "image " << image << " index " << i;
            }
        }
    }

    struct CpuAdamState {
        std::vector<float> param;
        std::vector<float> m;
        std::vector<float> v;
        int64_t last_step = 0;
    };

    void cpu_adam_update(CpuAdamState& state,
                         const std::vector<float>& grad,
                         const float lr,
                         const float beta1,
                         const float beta2,
                         const float bc1_rcp,
                         const float bc2_sqrt_rcp,
                         const float eps) {
        ASSERT_EQ(state.param.size(), grad.size());
        for (size_t i = 0; i < state.param.size(); ++i) {
            state.m[i] = beta1 * state.m[i] + (1.0f - beta1) * grad[i];
            state.v[i] = beta2 * state.v[i] + (1.0f - beta2) * grad[i] * grad[i];
            const float m_hat = state.m[i] * bc1_rcp;
            const float v_hat = state.v[i] * bc2_sqrt_rcp * bc2_sqrt_rcp;
            state.param[i] -= lr * m_hat / (std::sqrt(v_hat) + eps);
        }
    }

    void cpu_bias_corrections(const int64_t step,
                              const double beta1,
                              const double beta2,
                              float& bc1_rcp,
                              float& bc2_sqrt_rcp) {
        const double bc1 = 1.0 - std::pow(beta1, static_cast<double>(step + 1));
        const double bc2 = 1.0 - std::pow(beta2, static_cast<double>(step + 1));
        bc1_rcp = static_cast<float>(1.0 / bc1);
        bc2_sqrt_rcp = static_cast<float>(1.0 / std::sqrt(bc2));
    }

    void cpu_scheduler_step(int64_t& step,
                            double& current_lr,
                            const BilateralGrid::Config& config,
                            const double initial_lr,
                            const int total_iterations) {
        ++step;
        if (step <= config.warmup_steps) {
            const double progress = static_cast<double>(step) / config.warmup_steps;
            const double scale =
                config.warmup_start_factor + (1.0 - config.warmup_start_factor) * progress;
            current_lr = initial_lr * scale;
        } else {
            const double gamma = std::pow(config.final_lr_factor,
                                          1.0 / (total_iterations - config.warmup_steps));
            current_lr = initial_lr * std::pow(gamma, step - config.warmup_steps);
        }
    }

    void cpu_dense_adam_step(CpuAdamState& state,
                             const std::vector<float>& grad,
                             const int64_t step,
                             const double current_lr,
                             const BilateralGrid::Config& config) {
        const int64_t K = step - state.last_step;
        if (K > 1) {
            const double skipped = static_cast<double>(K - 1);
            const float s1 = static_cast<float>(std::pow(config.beta1, skipped));
            const float s2 = static_cast<float>(std::pow(config.beta2, skipped));
            for (size_t i = 0; i < state.m.size(); ++i) {
                state.m[i] *= s1;
                state.v[i] *= s2;
            }
        }
        float bc1_rcp = 0.0f;
        float bc2_sqrt_rcp = 0.0f;
        cpu_bias_corrections(step, config.beta1, config.beta2, bc1_rcp, bc2_sqrt_rcp);
        cpu_adam_update(state, grad, static_cast<float>(current_lr),
                        static_cast<float>(config.beta1), static_cast<float>(config.beta2),
                        bc1_rcp, bc2_sqrt_rcp, static_cast<float>(config.eps));
        state.last_step = step;
    }

    void expect_params_rel_near(const std::vector<float>& got,
                                const std::vector<float>& ref,
                                const float rel,
                                const std::string& ctx) {
        ASSERT_EQ(got.size(), ref.size()) << ctx;
        for (size_t i = 0; i < ref.size(); ++i) {
            const float denom = std::max(std::abs(ref[i]), 1.0e-8f);
            EXPECT_LE(std::abs(got[i] - ref[i]) / denom, rel)
                << ctx << " index " << i << " got=" << got[i] << " ref=" << ref[i];
        }
    }

    TEST_F(BilateralGridProjectionTest, SparseAdamCatchupMatchesDenseReference) {
        constexpr int kImages = 2;
        constexpr int kW = 2;
        constexpr int kH = 2;
        constexpr int kL = 2;
        constexpr int kIters = 20;
        constexpr int kSteps = 10;
        BilateralGrid::Config config;
        config.lr = 2e-3;
        config.beta1 = 0.9;
        config.beta2 = 0.999;
        config.eps = 1e-15;
        config.warmup_steps = 1000;
        config.warmup_start_factor = 0.01;
        config.final_lr_factor = 0.01;

        BilateralGrid grid(kImages, kW, kH, kL, kIters, config);
        auto& grids = grid.grids();
        std::vector<float> host = cpu_copy(grids);
        for (size_t i = 0; i < host.size(); ++i) {
            host[i] += 0.05f * std::sin(0.19f * static_cast<float>(i));
        }
        grids.copy_from(Tensor::from_vector(host, grids.shape(), Device::CUDA));

        const int slice = grid.channels() * kL * kH * kW;
        std::vector<float> grad0(static_cast<size_t>(slice));
        std::vector<float> grad1(static_cast<size_t>(slice));
        for (int i = 0; i < slice; ++i) {
            grad0[static_cast<size_t>(i)] = 0.01f * std::sin(0.11f * static_cast<float>(i));
            grad1[static_cast<size_t>(i)] = 0.02f * std::cos(0.13f * static_cast<float>(i));
        }
        const std::vector<float>* grads[2] = {&grad0, &grad1};

        CpuAdamState cpu[2];
        for (int img = 0; img < kImages; ++img) {
            cpu[img].param.assign(host.begin() + img * slice, host.begin() + (img + 1) * slice);
            cpu[img].m.assign(static_cast<size_t>(slice), 0.0f);
            cpu[img].v.assign(static_cast<size_t>(slice), 0.0f);
        }

        int64_t step = 0;
        double current_lr = config.warmup_steps > 0 ? config.lr * config.warmup_start_factor : config.lr;
        const auto visit1 = [](const int t) { return t % 3 == 0; };

        for (int t = 0; t < kSteps; ++t) {
            for (int img = 0; img < kImages; ++img) {
                const bool visited = (img == 0) || visit1(t);
                if (!visited)
                    continue;
                // Dense Adam with g=0 on skipped steps decays moments by beta
                // each visit; lazy catch-up scales by beta^(K-1) then applies the
                // current gradient. Parameter updates on those zero-grad steps
                // are omitted (standard lazy Adam).
                cpu_dense_adam_step(cpu[img], *grads[img], step, current_lr, config);
                grid.grad_slice().copy_from(
                    Tensor::from_vector(*grads[img], grid.grad_slice().shape(), Device::CUDA));
                grid.optimizer_step(img);
            }
            grid.scheduler_step();
            cpu_scheduler_step(step, current_lr, config, config.lr, kIters);
        }

        const std::vector<float> gpu = cpu_copy(grid.grids());
        std::vector<float> ref(host.size());
        for (int img = 0; img < kImages; ++img) {
            std::copy(cpu[img].param.begin(), cpu[img].param.end(),
                      ref.begin() + img * slice);
        }
        expect_params_rel_near(gpu, ref, 1e-6f, "sparse visits");

        // All-images optimizer_step must stay equivalent to dense Adam: every
        // image is visited each call, so last_step catch-up is a no-op.
        BilateralGrid dense_grid(kImages, kW, kH, kL, kIters, config);
        dense_grid.grids().copy_from(Tensor::from_vector(host, grids.shape(), Device::CUDA));
        CpuAdamState dense_cpu[2];
        for (int img = 0; img < kImages; ++img) {
            dense_cpu[img].param.assign(host.begin() + img * slice,
                                        host.begin() + (img + 1) * slice);
            dense_cpu[img].m.assign(static_cast<size_t>(slice), 0.0f);
            dense_cpu[img].v.assign(static_cast<size_t>(slice), 0.0f);
        }
        step = 0;
        current_lr = config.warmup_steps > 0 ? config.lr * config.warmup_start_factor : config.lr;
        for (int t = 0; t < kSteps; ++t) {
            for (int img = 0; img < kImages; ++img) {
                cpu_dense_adam_step(dense_cpu[img], *grads[img], step, current_lr, config);
                dense_grid.grad_slice().copy_from(
                    Tensor::from_vector(*grads[img], dense_grid.grad_slice().shape(), Device::CUDA));
                dense_grid.optimizer_step(img);
            }
            dense_grid.scheduler_step();
            cpu_scheduler_step(step, current_lr, config, config.lr, kIters);
        }
        const std::vector<float> dense_gpu = cpu_copy(dense_grid.grids());
        std::vector<float> dense_ref(host.size());
        for (int img = 0; img < kImages; ++img) {
            std::copy(dense_cpu[img].param.begin(), dense_cpu[img].param.end(),
                      dense_ref.begin() + img * slice);
        }
        expect_params_rel_near(dense_gpu, dense_ref, 1e-6f, "all-images dense");
    }

    TEST_F(BilateralGridProjectionTest, ChannelSumRebuildMatchesIncremental) {
        constexpr int kN = 2;
        constexpr int kW = 3;
        constexpr int kH = 3;
        constexpr int kL = 2;
        BilateralGrid grid(kN, kW, kH, kL, 20);
        auto& grids = grid.grids();
        std::vector<float> host = cpu_copy(grids);
        for (size_t i = 0; i < host.size(); ++i) {
            host[i] = 0.15f * std::sin(0.23f * static_cast<float>(i + 1)) +
                      0.07f * std::cos(0.09f * static_cast<float>(i));
        }
        grids.copy_from(Tensor::from_vector(host, grids.shape(), Device::CUDA));
        grid.project_mean(false);

        const int c = grid.channels();
        const int spatial = kL * kH * kW;
        std::vector<float> cpu_sum(static_cast<size_t>(c), 0.0f);
        for (int ni = 0; ni < kN; ++ni) {
            for (int ci = 0; ci < c; ++ci) {
                for (int cell = 0; cell < spatial; ++cell) {
                    const int idx = ((ni * c + ci) * spatial) + cell;
                    cpu_sum[static_cast<size_t>(ci)] += host[static_cast<size_t>(idx)];
                }
            }
        }
        const std::vector<float> rebuilt_sum = cpu_copy(grid.channel_sum());
        ASSERT_EQ(rebuilt_sum.size(), cpu_sum.size());
        for (int ci = 0; ci < c; ++ci) {
            EXPECT_NEAR(rebuilt_sum[static_cast<size_t>(ci)], cpu_sum[static_cast<size_t>(ci)], 1e-4f)
                << "rebuild factor is N*L*H*W; channel " << ci;
        }

        for (int t = 0; t < 5; ++t) {
            grid.step_image(t % kN, 0.05f);
        }

        const std::vector<float> incremental_sum = cpu_copy(grid.channel_sum());
        const std::vector<float> incremental_off = cpu_copy(grid.shared_offset());

        grid.project_mean(false);
        const std::vector<float> rebuild_sum = cpu_copy(grid.channel_sum());
        const std::vector<float> rebuild_off = cpu_copy(grid.shared_offset());
        ASSERT_EQ(incremental_sum.size(), rebuild_sum.size());
        ASSERT_EQ(incremental_off.size(), rebuild_off.size());
        for (size_t i = 0; i < rebuild_sum.size(); ++i) {
            EXPECT_NEAR(incremental_sum[i], rebuild_sum[i], 1e-5f) << "channel_sum " << i;
            EXPECT_NEAR(incremental_off[i], rebuild_off[i], 1e-5f) << "shared_offset " << i;
        }

        std::stringstream stream;
        grid.serialize(stream);
        BilateralGrid loaded(1, 1, 1, 1, 1);
        loaded.deserialize(stream);
        const std::vector<float> loaded_off = cpu_copy(loaded.shared_offset());
        const std::vector<float> loaded_sum = cpu_copy(loaded.channel_sum());
        for (size_t i = 0; i < rebuild_off.size(); ++i) {
            EXPECT_NEAR(loaded_off[i], rebuild_off[i], 1e-5f) << "deserialize offset " << i;
            EXPECT_NEAR(loaded_sum[i], rebuild_sum[i], 1e-5f) << "deserialize sum " << i;
        }

        BilateralGrid scratch(kN, kW, kH, kL, 20);
        scratch.grids().copy_from(grid.grids());
        scratch.project_mean(false);
        const std::vector<float> scratch_off = cpu_copy(scratch.shared_offset());
        for (size_t i = 0; i < rebuild_off.size(); ++i) {
            EXPECT_NEAR(loaded_off[i], scratch_off[i], 1e-5f) << "from-scratch offset " << i;
        }

        BilateralGrid live(kN, kW, kH, kL, 20);
        live.adopt_checkpoint_state(loaded);
        const std::vector<float> adopted_off = cpu_copy(live.shared_offset());
        for (size_t i = 0; i < rebuild_off.size(); ++i) {
            EXPECT_NEAR(adopted_off[i], scratch_off[i], 1e-5f) << "adopt offset " << i;
        }
    }

} // namespace
