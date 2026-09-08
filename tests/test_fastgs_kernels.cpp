/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "adam_api.h"
#include "core/camera.hpp"
#include "core/cuda/memory_arena.hpp"
#include "core/cuda/sh_layout.cuh"
#include "core/splat_data.hpp"
#include "core/tensor.hpp"
#include "core/tensor/internal/cuda_stream_context.hpp"
#include "core/tensor/internal/memory_pool.hpp"
#include "io/formats/ply.hpp"
#include "lfs/training/joint_adam_codec.hpp"
#include "lfs/training/morton_reorder.hpp"
#include "rasterization/fastgs/rasterization/include/forward.h"
#include "rasterization/fastgs/utils/utils.h"
#include "training/kernels/normal_consistency_loss.hpp"
#include "training/kernels/normal_loss.hpp"
#include "training/optimizer/adam_optimizer.hpp"
#include "training/rasterization/fast_rasterizer.hpp"
#include "training/strategies/mrnf.hpp"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <cuda_runtime.h>
#include <filesystem>
#include <gtest/gtest.h>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <torch/torch.h>
#include <unordered_set>
#include <vector>

using namespace lfs::training;
using namespace lfs::core;

cudaError_t fastgs_visibility_readback_delay(cudaStream_t stream, unsigned long long cycles);

namespace {
    constexpr const char* GARDEN_PATH = "data/garden";
    constexpr int W = 640, H = 480;
    constexpr float FX = 500.0f, FY = 500.0f;

    const AdamParamState& adam_state(const AdamOptimizer& opt, ParamType type) {
        const auto* state = opt.get_state(type);
        if (!state || !state->exp_avg.is_valid()) {
            throw std::runtime_error("Missing Adam moment state");
        }
        return *state;
    }

    // Recover gradients from the first Adam moment after one fused step from
    // zero moments: m = (1-beta1)*g, so g ≈ m/(1-beta1). Joint moments must
    // be decoded through their codec.
    Tensor adam_moment(const AdamOptimizer& opt, ParamType type) {
        const auto& state = adam_state(opt, type);
        if (state.exp_avg.dtype() == DataType::Float32) {
            return state.exp_avg;
        }

        // --- Joint (u, log_s) codec (default ON since 2.2) ---
        if (state.is_joint()) {
            if (!state.joint_bounds.is_valid()) {
                throw std::runtime_error("Joint Adam state missing joint_bounds");
            }
            const int bits = state.joint_bits;
            const int bpc = joint_adam::bytes_per_cell(bits);
            if (bpc <= 0) {
                throw std::runtime_error("Joint Adam: unsupported joint_bits");
            }

            auto packed_cpu = state.exp_avg.to(Device::CPU);
            auto bounds_cpu = state.joint_bounds.to(Device::CPU);
            const auto* packed = packed_cpu.ptr<std::uint8_t>();
            const auto* bounds = bounds_cpu.ptr<float>();
            const size_t n_bounds = bounds_cpu.shape()[0];

            // Contiguous params: packed [N, n_attr * bpc] → dequant [N, n_attr] (or [N] if n_attr==1
            // and original param was rank-1). Recover shape from packed layout.
            if (state.exp_avg.ndim() >= 2) {
                const size_t n_prim = state.exp_avg.shape()[0];
                const size_t packed_row = state.exp_avg.shape()[1];
                if (packed_row % static_cast<size_t>(bpc) != 0) {
                    throw std::runtime_error("Joint packed row not divisible by bytes_per_cell");
                }
                const size_t n_attr = packed_row / static_cast<size_t>(bpc);
                std::vector<float> dequant(n_prim * n_attr, 0.0f);

                auto decode_cell = [&](std::size_t cell, float umin, float umax, float smin, float smax,
                                       float& g1, float& g2) {
                    if (bits == 16) {
                        joint_adam::Codec16::decode_g1g2(packed, cell, umin, umax, smin, smax, g1, g2);
                    } else {
                        joint_adam::Codec8::decode_g1g2(packed, cell, umin, umax, smin, smax, g1, g2);
                    }
                };

                for (size_t p = 0; p < n_prim; ++p) {
                    const size_t bidx = p / static_cast<size_t>(joint_adam::kBlockSize);
                    if (bidx >= n_bounds) {
                        throw std::runtime_error("Joint bounds undersized for primitive index");
                    }
                    const float umin = bounds[bidx * 4 + 0];
                    const float umax = bounds[bidx * 4 + 1];
                    const float smin = bounds[bidx * 4 + 2];
                    const float smax = bounds[bidx * 4 + 3];
                    for (size_t a = 0; a < n_attr; ++a) {
                        const size_t cell = p * n_attr + a;
                        float g1 = 0.0f, g2 = 0.0f;
                        decode_cell(cell, umin, umax, smin, smax, g1, g2);
                        dequant[cell] = g1;
                    }
                }

                // Match param layouts used by numerical grads: sh0 [N,1,3], opacity [N], else [N,C].
                TensorShape out_shape;
                if (type == ParamType::Sh0 && n_attr == 3) {
                    out_shape = TensorShape({n_prim, size_t{1}, size_t{3}});
                } else if (n_attr == 1) {
                    out_shape = TensorShape({n_prim});
                } else {
                    out_shape = TensorShape({n_prim, n_attr});
                }
                return Tensor::from_blob(dequant.data(), out_shape, Device::CPU, DataType::Float32)
                    .clone()
                    .to(Device::CUDA);
            }

            // Swizzled shN: 1D packed cells (one cell per swizzled float).
            // Bounds are per 256-primitive block. When only one bounds row exists (N<=256),
            // every cell shares bounds[0] — sufficient for crop-damping N=1 and small fuzz.
            const size_t n_cells = state.exp_avg.numel() / static_cast<size_t>(bpc);
            if (n_bounds != 1) {
                throw std::runtime_error(
                    "adam_moment joint 1D (shN) multi-block decode needs swizzle→prim map");
            }
            std::vector<float> dequant(n_cells, 0.0f);
            const float umin = bounds[0], umax = bounds[1], smin = bounds[2], smax = bounds[3];
            for (size_t cell = 0; cell < n_cells; ++cell) {
                float g1 = 0.0f, g2 = 0.0f;
                if (bits == 16) {
                    joint_adam::Codec16::decode_g1g2(packed, cell, umin, umax, smin, smax, g1, g2);
                } else {
                    joint_adam::Codec8::decode_g1g2(packed, cell, umin, umax, smin, smax, g1, g2);
                }
                dequant[cell] = g1;
            }
            return Tensor::from_blob(dequant.data(), TensorShape({n_cells}), Device::CPU, DataType::Float32)
                .clone()
                .to(Device::CUDA);
        }

        throw std::runtime_error("Legacy Adam moment codec is unsupported");
    }

    void expect_adam_state_finite(const AdamOptimizer& opt, ParamType type) {
        const auto& state = adam_state(opt, type);
        if (state.is_joint()) {
            ASSERT_TRUE(state.joint_bounds.is_valid());
            auto bounds = state.joint_bounds.to(Device::CPU);
            auto* ptr = bounds.ptr<float>();
            for (size_t i = 0; i < bounds.numel(); ++i) {
                EXPECT_TRUE(std::isfinite(ptr[i]));
            }
            // Packed uint8 codes are always finite by construction.
            return;
        }
    }

    // L1 of decoded first moment m (joint or legacy). Proxy for "moments moved".
    float first_moment_l1(const AdamOptimizer& opt, ParamType type) {
        auto m = adam_moment(opt, type);
        return m.abs().sum().item<float>();
    }

    Tensor recovered_fused_grad(const AdamOptimizer& opt, ParamType type, float beta1 = 0.9f) {
        return adam_moment(opt, type).mul(1.0f / (1.0f - beta1));
    }

    SplatData make_adam_test_splat(const size_t count, const int sh_degree = 0) {
        std::vector<float> means_data(count * 3, 0.0f);
        std::vector<float> rotations(count * 4, 0.0f);
        for (size_t i = 0; i < count; ++i) {
            means_data[i * 3 + 2] = 1.0f;
            rotations[i * 4] = 1.0f;
        }
        const size_t sh_rest = sh_rest_coefficients_for_degree(sh_degree);
        return SplatData(
            sh_degree,
            Tensor::from_vector(means_data, {count, size_t{3}}, Device::CUDA),
            Tensor::full({count, size_t{1}, size_t{3}}, 0.25f, Device::CUDA),
            Tensor::full({count, sh_rest, size_t{3}}, 0.1f, Device::CUDA),
            Tensor::full({count, size_t{3}}, -1.5f, Device::CUDA),
            Tensor::from_vector(rotations, {count, size_t{4}}, Device::CUDA),
            Tensor::zeros({count, size_t{1}}, Device::CUDA),
            1.0f);
    }
} // namespace

TEST(FastGSOverflowGuards, RejectsInstanceCountsBeyondIntRange) {
    const uint64_t max_int = static_cast<uint64_t>(std::numeric_limits<int>::max());
    EXPECT_EQ(checked_fastgs_instance_count(max_int, 1, 1), std::numeric_limits<int>::max());
    EXPECT_THROW(
        checked_fastgs_instance_count(max_int + 1, 595037, 11907),
        std::overflow_error);
}

TEST(FastGSOverflowGuards, RejectsVisibleCountsBeyondPrimitiveCount) {
    const uint64_t max_int = static_cast<uint64_t>(std::numeric_limits<int>::max());
    EXPECT_EQ(checked_fastgs_visible_count(0, 1000), 0);
    EXPECT_EQ(checked_fastgs_visible_count(1, 1000), 1);
    EXPECT_EQ(checked_fastgs_visible_count(1000, 1000), 1000);
    EXPECT_EQ(checked_fastgs_visible_count(max_int, max_int), std::numeric_limits<int>::max());

    for (const uint64_t count : {uint64_t{1001}, max_int,
                                 static_cast<uint64_t>(std::numeric_limits<uint32_t>::max())}) {
        SCOPED_TRACE(count);
        try {
            (void)checked_fastgs_visible_count(count, 1000);
            FAIL() << "An impossible visible count must be rejected";
        } catch (const std::runtime_error& error) {
            EXPECT_EQ(std::string(error.what()),
                      "FastGS visible count exceeds primitive count: " + std::to_string(count) +
                          " visible primitives from 1000 primitives");
        }
    }
    EXPECT_THROW(checked_fastgs_visible_count(max_int + 1, max_int + 1), std::overflow_error);
}

class FastGSKernelTest : public ::testing::Test {
protected:
    void SetUp() override {
        if (!std::filesystem::exists(GARDEN_PATH)) {
            GTEST_SKIP() << "Garden dataset not found";
        }

        std::string ply = std::string(GARDEN_PATH) + "/point_cloud/iteration_7000/point_cloud.ply";
        if (std::filesystem::exists(ply)) {
            load_ply(ply);
        } else {
            create_synthetic_data();
        }

        auto R = Tensor::eye(3, Device::CUDA);
        std::vector<float> t_data{0, 0, 5};
        auto T = Tensor::from_blob(t_data.data(), {3}, Device::CPU, DataType::Float32).to(Device::CUDA);
        camera_ = std::make_unique<Camera>(R, T, FX, FY, W / 2.0f, H / 2.0f,
                                           Tensor(), Tensor(), CameraModelType::PINHOLE,
                                           "test", "", std::filesystem::path{}, W, H, 0);
        bg_ = Tensor::zeros({3}, Device::CUDA);
    }

    void TearDown() override {
        splat_.reset();
        camera_.reset();
        GlobalArenaManager::instance().get_arena().full_reset();
    }

    void load_ply(const std::string& path) {
        auto result = lfs::io::load_ply(path);
        if (!result) {
            create_synthetic_data();
            return;
        }
        n_ = std::min(result->value.means().shape()[0], size_t(10000));
        means_ = result->value.means().slice(0, 0, n_).contiguous().to(Device::CUDA);
        init_params();
    }

    void create_synthetic_data() {
        n_ = 10000;
        std::vector<float> data(n_ * 3);
        std::mt19937 gen(42);
        std::uniform_real_distribution<float> xy(-5, 5), z(-1, 3);
        for (size_t i = 0; i < n_; ++i) {
            data[i * 3] = xy(gen);
            data[i * 3 + 1] = xy(gen);
            data[i * 3 + 2] = z(gen);
        }
        means_ = Tensor::from_blob(data.data(), {n_, 3}, Device::CPU, DataType::Float32).to(Device::CUDA);
        init_params();
    }

    void init_params() {
        sh0_ = Tensor::randn({n_, 1, 3}, Device::CUDA).mul(0.5f);
        shN_ = Tensor::zeros({n_, 0, 3}, Device::CUDA);
        scaling_ = Tensor::randn({n_, 3}, Device::CUDA).mul(0.3f).sub(3.5f);
        rotation_ = Tensor::randn({n_, 4}, Device::CUDA);
        rotation_ = rotation_ / rotation_.pow(2.0f).sum(-1, true).sqrt();
        opacity_ = Tensor::randn({n_}, Device::CUDA).mul(2.0f);
        splat_ = std::make_unique<SplatData>(0, means_, sh0_, shN_, scaling_, rotation_, opacity_, 1.0f);
    }

    std::unique_ptr<AdamOptimizer> make_optimizer() {
        AdamConfig cfg{.lr = 0.001f, .beta1 = 0.9, .beta2 = 0.999, .eps = 1e-15};
        auto opt = std::make_unique<AdamOptimizer>(*splat_, cfg);
        opt->allocate_gradients();
        return opt;
    }

    auto forward() { return fast_rasterize_forward(*camera_, *splat_, bg_, 0, 0, 0, 0, false); }

    size_t n_ = 0;
    std::unique_ptr<SplatData> splat_;
    std::unique_ptr<Camera> camera_;
    Tensor means_, sh0_, shN_, scaling_, rotation_, opacity_, bg_;
};

// Forward kernels
TEST_F(FastGSKernelTest, Forward_Preprocess) {
    auto r = forward();
    ASSERT_TRUE(r.has_value()) << lfs::format_for_developer(r.error());
    EXPECT_GT(r->second.forward_ctx.n_instances, 0);
}

TEST_F(FastGSKernelTest, Forward_TileDepthOrdering) {
    auto r = forward();
    ASSERT_TRUE(r.has_value());
    EXPECT_TRUE(r->first.image.is_valid());
}

TEST_F(FastGSKernelTest, Forward_Instances) {
    auto r = forward();
    ASSERT_TRUE(r.has_value());
    EXPECT_GT(r->second.forward_ctx.n_instances, 0);
}

TEST_F(FastGSKernelTest, Forward_TileState) {
    auto r = forward();
    ASSERT_TRUE(r.has_value());
    EXPECT_GT(r->second.forward_ctx.per_tile_buffers_size, 0);
}

TEST_F(FastGSKernelTest, Forward_Blend) {
    auto r = forward();
    ASSERT_TRUE(r.has_value());
    auto& out = r->first;

    EXPECT_EQ(out.image.ndim(), 3);
    EXPECT_EQ(out.image.shape()[0], 3);
    EXPECT_EQ(out.image.shape()[1], static_cast<size_t>(H));
    EXPECT_EQ(out.image.shape()[2], static_cast<size_t>(W));

    float alpha_min = out.alpha.min().item<float>();
    float alpha_max = out.alpha.max().item<float>();
    EXPECT_GE(alpha_min, 0.0f);
    EXPECT_LE(alpha_max, 1.0f);
    EXPECT_GT(out.image.std().item<float>(), 0.0f);
}

TEST_F(FastGSKernelTest, Forward_Full) {
    auto r = forward();
    ASSERT_TRUE(r.has_value()) << lfs::format_for_developer(r.error());
    EXPECT_EQ(r->first.width, W);
    EXPECT_EQ(r->first.height, H);
}

// Backward kernels
TEST_F(FastGSKernelTest, Backward_Blend) {
    auto r = forward();
    ASSERT_TRUE(r.has_value());

    auto opt = make_optimizer();
    opt->zero_grad(0);
    fast_rasterize_backward(r->second, Tensor::ones_like(r->first.image),
                            *splat_, *opt, Tensor::zeros_like(r->first.alpha));

    EXPECT_GT(adam_moment(*opt, ParamType::Means).pow(2.0f).sum().item<float>(), 0.0f);
    EXPECT_GT(adam_moment(*opt, ParamType::Scaling).pow(2.0f).sum().item<float>(), 0.0f);
}

TEST_F(FastGSKernelTest, EdgeWeightedContributionUsesFloatMapInMainBackward) {
    auto r = forward();
    ASSERT_TRUE(r.has_value()) << lfs::format_for_developer(r.error());

    // This value lies halfway between the rejected u8/16 cache levels, so the
    // equality below also guards the float32 map contract against quantization.
    constexpr float edge_weight = 1.03125f;
    const float expected_total_contribution =
        r->first.alpha.sum().item<float>() * edge_weight;
    auto edge_weights = Tensor::full(
        {H, W}, edge_weight, Device::CUDA, DataType::Float32);
    auto edge_scores = Tensor::zeros({n_}, Device::CUDA, DataType::Float32);
    FastGSFusedExtraGradients fused;
    fused.edge_weight_map = edge_weights.ptr<float>();
    fused.edge_score_out = edge_scores.ptr<float>();

    auto opt = make_optimizer();
    opt->zero_grad(0);
    fast_rasterize_backward(
        r->second,
        Tensor::zeros_like(r->first.image),
        *splat_,
        *opt,
        Tensor::zeros_like(r->first.alpha),
        {},
        DensificationType::None,
        1,
        fused);

    const float actual_total_contribution = edge_scores.sum().item<float>();
    EXPECT_GT(actual_total_contribution, 0.0f);
    EXPECT_NEAR(actual_total_contribution, expected_total_contribution,
                std::max(1.0e-3f, expected_total_contribution * 1.0e-4f));
}

TEST(FastGSDepthGradientTest, BackwardDepthMatchesLibtorchAutogradForCenteredSplat) {
    if (!torch::cuda::is_available()) {
        GTEST_SKIP() << "CUDA not available";
    }

    std::vector<float> means_data{0.0f, 0.0f, 1.0f};
    auto means = Tensor::from_blob(means_data.data(), {1, 3}, Device::CPU, DataType::Float32).to(Device::CUDA);
    auto sh0 = Tensor::zeros({1, 1, 3}, Device::CUDA);
    auto shN = Tensor::zeros({1, 0, 3}, Device::CUDA);
    auto scaling = Tensor::full({1, 3}, -1.5f, Device::CUDA);
    std::vector<float> rotation_data{1.0f, 0.0f, 0.0f, 0.0f};
    auto rotation = Tensor::from_blob(rotation_data.data(), {1, 4}, Device::CPU, DataType::Float32).to(Device::CUDA);

    const float opacity_value = 0.3f;
    const float raw_opacity_value = std::log(opacity_value / (1.0f - opacity_value));
    auto opacity = Tensor::full({1}, raw_opacity_value, Device::CUDA);
    auto splat = SplatData(0, means, sh0, shN, scaling, rotation, opacity, 1.0f);

    auto R = Tensor::eye(3, Device::CUDA);
    std::vector<float> t_data{0.0f, 0.0f, 4.0f};
    auto T = Tensor::from_blob(t_data.data(), {3}, Device::CPU, DataType::Float32).to(Device::CUDA);
    auto camera = Camera(R, T, 1.0f, 1.0f, 0.5f, 0.5f,
                         Tensor(), Tensor(), CameraModelType::PINHOLE,
                         "depth_grad", "", std::filesystem::path{}, 1, 1, 0);
    auto bg = Tensor::zeros({3}, Device::CUDA);

    auto forward = fast_rasterize_forward(camera, splat, bg, 0, 0, 0, 0, false);
    ASSERT_TRUE(forward.has_value()) << lfs::format_for_developer(forward.error());
    ASSERT_EQ(forward->first.depth.numel(), 1);
    EXPECT_GT(forward->first.depth.item<float>(), 0.0f);

    AdamConfig cfg{.lr = 0.001f, .beta1 = 0.9, .beta2 = 0.999, .eps = 1e-15};
    AdamOptimizer opt(splat, cfg);
    opt.allocate_gradients();
    opt.zero_grad(0);

    const float upstream_depth_grad = 1.7f;
    auto grad_image = Tensor::zeros_like(forward->first.image);
    auto grad_depth = Tensor::full({1, 1}, upstream_depth_grad, Device::CUDA);
    fast_rasterize_backward(
        forward->second,
        grad_image,
        splat,
        opt,
        {},
        {},
        DensificationType::None,
        1,
        {},
        grad_depth);

    const auto mean_grad = recovered_fused_grad(opt, ParamType::Means).to(Device::CPU);
    const auto opacity_grad = recovered_fused_grad(opt, ParamType::Opacity).to(Device::CPU);

    auto raw_opacity_ag = torch::tensor({raw_opacity_value}, torch::dtype(torch::kFloat32).device(torch::kCUDA))
                              .set_requires_grad(true);
    auto depth_ag = torch::tensor({means_data[2] + t_data[2]}, torch::dtype(torch::kFloat32).device(torch::kCUDA))
                        .set_requires_grad(true);
    auto depth_out = torch::sigmoid(raw_opacity_ag) * depth_ag;
    auto loss = depth_out * upstream_depth_grad;
    loss.backward();

    const float expected_opacity_grad = raw_opacity_ag.grad().item<float>();
    const float expected_depth_grad = depth_ag.grad().item<float>();
    const float actual_opacity_grad = opacity_grad.ptr<float>()[0];
    const float actual_mean_z_grad = mean_grad.ptr<float>()[2];

    EXPECT_NEAR(actual_opacity_grad, expected_opacity_grad, 1.0e-4f)
        << "raw opacity depth gradient should match libtorch autograd";
    EXPECT_NEAR(actual_mean_z_grad, expected_depth_grad, 1.0e-4f)
        << "mean z depth gradient should match libtorch autograd";
    EXPECT_NEAR(mean_grad.ptr<float>()[0], 0.0f, 1.0e-5f);
    EXPECT_NEAR(mean_grad.ptr<float>()[1], 0.0f, 1.0e-5f);
}

TEST(FastGSDepthGradientTest, BackwardDepthMatchesLibtorchAutogradForOverlappingSplats) {
    if (!torch::cuda::is_available()) {
        GTEST_SKIP() << "CUDA not available";
    }

    std::vector<float> means_data{
        0.0f, 0.0f, 0.5f,
        0.0f, 0.0f, 1.5f};
    auto means = Tensor::from_blob(means_data.data(), {2, 3}, Device::CPU, DataType::Float32).to(Device::CUDA);
    auto sh0 = Tensor::zeros({2, 1, 3}, Device::CUDA);
    auto shN = Tensor::zeros({2, 0, 3}, Device::CUDA);
    auto scaling = Tensor::full({2, 3}, -1.5f, Device::CUDA);
    std::vector<float> rotation_data{
        1.0f, 0.0f, 0.0f, 0.0f,
        1.0f, 0.0f, 0.0f, 0.0f};
    auto rotation = Tensor::from_blob(rotation_data.data(), {2, 4}, Device::CPU, DataType::Float32).to(Device::CUDA);

    const std::vector<float> opacity_values{0.25f, 0.4f};
    std::vector<float> raw_opacity_values{
        std::log(opacity_values[0] / (1.0f - opacity_values[0])),
        std::log(opacity_values[1] / (1.0f - opacity_values[1]))};
    auto opacity = Tensor::from_blob(raw_opacity_values.data(), {2}, Device::CPU, DataType::Float32).to(Device::CUDA);
    auto splat = SplatData(0, means, sh0, shN, scaling, rotation, opacity, 1.0f);

    auto R = Tensor::eye(3, Device::CUDA);
    std::vector<float> t_data{0.0f, 0.0f, 4.0f};
    auto T = Tensor::from_blob(t_data.data(), {3}, Device::CPU, DataType::Float32).to(Device::CUDA);
    auto camera = Camera(R, T, 1.0f, 1.0f, 0.5f, 0.5f,
                         Tensor(), Tensor(), CameraModelType::PINHOLE,
                         "depth_grad_overlap", "", std::filesystem::path{}, 1, 1, 0);
    auto bg = Tensor::zeros({3}, Device::CUDA);

    auto forward = fast_rasterize_forward(camera, splat, bg, 0, 0, 0, 0, false);
    ASSERT_TRUE(forward.has_value()) << lfs::format_for_developer(forward.error());
    ASSERT_EQ(forward->first.depth.numel(), 1);

    const float depth0 = means_data[2] + t_data[2];
    const float depth1 = means_data[5] + t_data[2];
    const float expected_forward_depth =
        opacity_values[0] * depth0 +
        (1.0f - opacity_values[0]) * opacity_values[1] * depth1;
    EXPECT_NEAR(forward->first.depth.item<float>(), expected_forward_depth, 1.0e-4f)
        << "test setup should render the nearer splat first";

    AdamConfig cfg{.lr = 0.001f, .beta1 = 0.9, .beta2 = 0.999, .eps = 1e-15};
    AdamOptimizer opt(splat, cfg);
    opt.allocate_gradients();
    opt.zero_grad(0);

    const float upstream_depth_grad = 1.3f;
    auto grad_image = Tensor::zeros_like(forward->first.image);
    auto grad_depth = Tensor::full({1, 1}, upstream_depth_grad, Device::CUDA);
    fast_rasterize_backward(
        forward->second,
        grad_image,
        splat,
        opt,
        {},
        {},
        DensificationType::None,
        1,
        {},
        grad_depth);

    const auto mean_grad = recovered_fused_grad(opt, ParamType::Means).to(Device::CPU);
    const auto opacity_grad = recovered_fused_grad(opt, ParamType::Opacity).to(Device::CPU);

    const auto opts = torch::dtype(torch::kFloat32).device(torch::kCUDA);
    auto raw_opacity_ag = torch::tensor(raw_opacity_values, opts).clone().set_requires_grad(true);
    auto depth_ag = torch::tensor({depth0, depth1}, opts).clone().set_requires_grad(true);
    const auto alpha = torch::sigmoid(raw_opacity_ag);
    const auto depth_out =
        alpha.select(0, 0) * depth_ag.select(0, 0) +
        (1.0f - alpha.select(0, 0)) * alpha.select(0, 1) * depth_ag.select(0, 1);
    const auto loss = depth_out * upstream_depth_grad;
    loss.backward();

    const auto expected_opacity_grad = raw_opacity_ag.grad().to(torch::kCPU);
    const auto expected_depth_grad = depth_ag.grad().to(torch::kCPU);
    const float* actual_opacity_grad = opacity_grad.ptr<float>();
    const float* actual_mean_grad = mean_grad.ptr<float>();

    for (int i = 0; i < 2; ++i) {
        EXPECT_NEAR(actual_opacity_grad[i], expected_opacity_grad[i].item<float>(), 1.0e-4f)
            << "raw opacity depth gradient mismatch for splat " << i;
        EXPECT_NEAR(actual_mean_grad[i * 3 + 2], expected_depth_grad[i].item<float>(), 1.0e-4f)
            << "mean z depth gradient mismatch for splat " << i;
        EXPECT_NEAR(actual_mean_grad[i * 3], 0.0f, 1.0e-5f);
        EXPECT_NEAR(actual_mean_grad[i * 3 + 1], 0.0f, 1.0e-5f);
    }
}

namespace {
    struct NormalChannelScene {
        std::vector<float> means_data{0.0f, 0.0f, 1.0f};
        // Distinct raw log-scales with z clearly smallest so the normal axis
        // (argmin variance) is stable under the perturbations below.
        std::vector<float> scaling_data{-1.0f, -1.5f, -3.0f};
        float opacity_value = 0.3f;
        std::vector<float> t_data{0.0f, 0.0f, 4.0f};

        SplatData make_splat(const std::vector<float>& rotation_data) const {
            const size_t n = means_data.size() / 3;
            auto means = Tensor::from_blob(const_cast<float*>(means_data.data()), {n, 3}, Device::CPU, DataType::Float32).to(Device::CUDA);
            auto sh0 = Tensor::zeros({n, 1, 3}, Device::CUDA);
            auto shN = Tensor::zeros({n, 0, 3}, Device::CUDA);
            auto scaling = Tensor::from_blob(const_cast<float*>(scaling_data.data()), {n, 3}, Device::CPU, DataType::Float32).to(Device::CUDA);
            auto rotation = Tensor::from_blob(const_cast<float*>(rotation_data.data()), {n, 4}, Device::CPU, DataType::Float32).to(Device::CUDA);
            const float raw_opacity = std::log(opacity_value / (1.0f - opacity_value));
            auto opacity = Tensor::full({n}, raw_opacity, Device::CUDA);
            return SplatData(0, means, sh0, shN, scaling, rotation, opacity, 1.0f);
        }

        Camera make_camera() const {
            auto R = Tensor::eye(3, Device::CUDA);
            auto T = Tensor::from_blob(const_cast<float*>(t_data.data()), {3}, Device::CPU, DataType::Float32).to(Device::CUDA);
            return Camera(R, T, 1.0f, 1.0f, 0.5f, 0.5f,
                          Tensor(), Tensor(), CameraModelType::PINHOLE,
                          "normal_grad", "", std::filesystem::path{}, 1, 1, 0);
        }
    };
} // namespace

class FastGSVisibilityReadback : public ::testing::Test {
protected:
    static constexpr size_t scratch_bytes = 16 * 1024 * 1024;
    cudaStream_t blocking_stream_ = nullptr;
    cudaStream_t nonblocking_stream_ = nullptr;
    char* scratch_ = nullptr;
    unsigned long long delay_cycles_ = 0;

    static void check_cuda(cudaError_t error) {
        if (error != cudaSuccess) {
            throw std::runtime_error(cudaGetErrorString(error));
        }
    }

    void SetUp() override {
        if (!torch::cuda::is_available()) {
            GTEST_SKIP() << "CUDA not available";
        }
        ASSERT_EQ(cudaStreamCreateWithFlags(&blocking_stream_, cudaStreamDefault), cudaSuccess);
        ASSERT_EQ(cudaStreamCreateWithFlags(&nonblocking_stream_, cudaStreamNonBlocking), cudaSuccess);
        ASSERT_EQ(cudaMalloc(&scratch_, scratch_bytes), cudaSuccess);
        int device = 0;
        int clock_khz = 0;
        ASSERT_EQ(cudaGetDevice(&device), cudaSuccess);
        ASSERT_EQ(cudaDeviceGetAttribute(&clock_khz, cudaDevAttrClockRate, device), cudaSuccess);
        delay_cycles_ = static_cast<unsigned long long>(clock_khz) * 200;
        // Load the helper before the timed run, including with CUDA lazy loading.
        ASSERT_EQ(fastgs_visibility_readback_delay(nonblocking_stream_, 0), cudaSuccess);
        ASSERT_EQ(cudaStreamSynchronize(nonblocking_stream_), cudaSuccess);
    }

    void TearDown() override {
        if (blocking_stream_)
            EXPECT_EQ(cudaStreamSynchronize(blocking_stream_), cudaSuccess);
        if (nonblocking_stream_)
            EXPECT_EQ(cudaStreamSynchronize(nonblocking_stream_), cudaSuccess);
        if (scratch_)
            EXPECT_EQ(cudaFree(scratch_), cudaSuccess);
        if (nonblocking_stream_)
            EXPECT_EQ(cudaStreamDestroy(nonblocking_stream_), cudaSuccess);
        if (blocking_stream_)
            EXPECT_EQ(cudaStreamDestroy(blocking_stream_), cudaSuccess);
    }

    struct RenderResult {
        int n_visible;
        float image[3];
        std::vector<unsigned> primitive_work_indices;
    };

    RenderResult render(Camera& camera, SplatData& splat, cudaStream_t stream, bool delay = false,
                        int scratch_fill = 0) {
        CUDAStreamGuard guard(stream);
        // Preallocate every callback's storage. cudaMalloc/arena frame entry
        // after the delay could synchronize the device and conceal the bug.
        // Zero scratch makes an unordered read deterministically see zero.
        check_cuda(cudaMemsetAsync(scratch_, scratch_fill, scratch_bytes, stream));
        check_cuda(cudaDeviceSynchronize());
        size_t used = 256; // Reserve image, alpha and depth at the front.
        auto allocate = [&](size_t bytes) -> char* {
            const size_t offset = (used + 255) & ~size_t{255};
            if (offset > scratch_bytes || bytes > scratch_bytes - offset) {
                throw std::runtime_error("FastGS visibility test scratch exhausted");
            }
            used = offset + bytes;
            return scratch_ + offset;
        };
        auto* output = reinterpret_cast<float*>(scratch_);
        if (delay) {
            check_cuda(fastgs_visibility_readback_delay(stream, delay_cycles_));
        }
        const auto result = fast_lfs::rasterization::forward(
            allocate,
            [](size_t) {}, // Keep all phases alive until forward completes.
            allocate,
            [](const void* ptr, size_t) { return static_cast<char*>(const_cast<void*>(ptr)); },
            allocate,
            reinterpret_cast<const float3*>(splat.means().ptr<float>()),
            reinterpret_cast<const float3*>(splat.scaling_raw().ptr<float>()),
            reinterpret_cast<const float4*>(splat.rotation_raw().ptr<float>()),
            splat.opacity_raw().ptr<float>(),
            reinterpret_cast<const float3*>(splat.sh0().ptr<float>()),
            nullptr, nullptr, 0, 0,
            reinterpret_cast<const float4*>(camera.world_view_transform_ptr()),
            reinterpret_cast<const float3*>(camera.cam_position_ptr()),
            output, output + 3, output + 4,
            nullptr, nullptr, nullptr,
            static_cast<int>(splat.means().shape()[0]),
            1, 1, 1, 1, 1.0f, 1.0f, 0.5f, 0.5f, 0.01f, 1e10f,
            false, getCurrentCUDAStream());
        check_cuda(cudaStreamSynchronize(stream));
        RenderResult host{.n_visible = result.n_visible, .image = {}, .primitive_work_indices = std::vector<unsigned>(splat.means().shape()[0])};
        check_cuda(cudaMemcpy(host.image, output, sizeof(host.image), cudaMemcpyDeviceToHost));
        check_cuda(cudaMemcpy(host.primitive_work_indices.data(), result.primitive_work_indices,
                              host.primitive_work_indices.size() * sizeof(unsigned), cudaMemcpyDeviceToHost));
        return host;
    }
};

TEST_F(FastGSVisibilityReadback, CountIsStreamOrderedOnNonBlockingStream) {
    NormalChannelScene scene;
    auto camera = scene.make_camera();
    auto splat = scene.make_splat({1.0f, 0.0f, 0.0f, 0.0f});
    // Warm the forward kernels and CUB before the delayed call as well.
    const auto reference = render(camera, splat, blocking_stream_);
    ASSERT_EQ(reference.n_visible, 1);
    ASSERT_GT(reference.image[0], 0.0f);

    const auto actual = render(camera, splat, nonblocking_stream_, true);
    EXPECT_EQ(actual.n_visible, reference.n_visible);
    for (int channel = 0; channel < 3; ++channel) {
        EXPECT_NEAR(actual.image[channel], reference.image[channel], 1e-6f);
    }
}

TEST_F(FastGSVisibilityReadback, CountIsBoundedByPrimitiveCount) {
    NormalChannelScene scene;
    auto camera = scene.make_camera();
    for (const size_t count : {1, 31, 32, 33, 255, 256, 257, 1000}) {
        for (const bool visible : {true, false}) {
            SCOPED_TRACE(::testing::Message() << "N=" << count << ", visible=" << visible);
            auto splat = make_adam_test_splat(count);
            if (!visible) {
                // Camera is at z=-4; put every primitive behind it.
                std::vector<float> means(count * 3, 0.0f);
                for (size_t i = 0; i < count; ++i)
                    means[i * 3 + 2] = -10.0f;
                splat.means() = Tensor::from_vector(means, {count, size_t{3}}, Device::CUDA);
            }
            const auto result = render(camera, splat, nonblocking_stream_);
            EXPECT_GE(result.n_visible, 0);
            EXPECT_LE(result.n_visible, static_cast<int>(count));
            EXPECT_EQ(result.n_visible, visible ? static_cast<int>(count) : 0);
        }
    }
}

TEST_F(FastGSVisibilityReadback, MixedVisibilityCompactsIndicesWithDirtyScratch) {
    NormalChannelScene scene;
    auto camera = scene.make_camera();
    enum class Pattern { Alternating,
                         BlockEdges,
                         LastOnly,
                         None,
                         All };
    for (const size_t count : {1, 31, 32, 33, 255, 256, 257, 511, 512, 513,
                               4095, 4096, 4097, 65537}) {
        for (const auto pattern : {Pattern::Alternating, Pattern::BlockEdges,
                                   Pattern::LastOnly, Pattern::None, Pattern::All}) {
            SCOPED_TRACE(::testing::Message() << "N=" << count << ", pattern=" << static_cast<int>(pattern));
            auto splat = make_adam_test_splat(count);
            std::vector<float> means(count * 3, 0.0f);
            std::vector<unsigned> expected(count, 0xffffffffu);
            unsigned n_visible = 0;
            for (size_t i = 0; i < count; ++i) {
                bool visible = false;
                switch (pattern) {
                case Pattern::Alternating: visible = i % 2 == 1; break;
                case Pattern::BlockEdges: visible = i % 256 == 255 || i % 256 == 0; break;
                case Pattern::LastOnly: visible = i == count - 1; break;
                case Pattern::None: break;
                case Pattern::All: visible = true; break;
                }
                means[3 * i + 2] = visible ? 1.0f : -10.0f;
                if (visible)
                    expected[i] = n_visible++;
            }
            splat.means() = Tensor::from_vector(means, {count, size_t{3}}, Device::CUDA);
            const auto reference = render(camera, splat, blocking_stream_);
            ASSERT_EQ(reference.n_visible, n_visible);
            ASSERT_EQ(reference.primitive_work_indices, expected);

            // Dirty storage exposes missing map writes and dependence on zeros.
            // The separate delayed test exercises the readback ordering race.
            const auto actual = render(camera, splat, nonblocking_stream_, false, 0xa5);
            EXPECT_EQ(actual.n_visible, n_visible);
            EXPECT_EQ(actual.primitive_work_indices, expected);
            for (int channel = 0; channel < 3; ++channel)
                EXPECT_NEAR(actual.image[channel], reference.image[channel], 1e-6f);
        }
    }
}

TEST(FastGSNormalChannelTest, RendersCameraSpaceNormalForCenteredSplat) {
    if (!torch::cuda::is_available()) {
        GTEST_SKIP() << "CUDA not available";
    }

    NormalChannelScene scene;
    const std::vector<float> identity_quat{1.0f, 0.0f, 0.0f, 0.0f};
    auto splat = scene.make_splat(identity_quat);
    auto camera = scene.make_camera();
    auto bg = Tensor::zeros({3}, Device::CUDA);

    auto without_normal = fast_rasterize_forward(camera, splat, bg, 0, 0, 0, 0, false);
    ASSERT_TRUE(without_normal.has_value())
        << lfs::format_for_developer(without_normal.error());
    EXPECT_FALSE(without_normal->first.normal.is_valid())
        << "normal channel must stay off unless requested";
    without_normal->second.release_forward_context();

    auto forward = fast_rasterize_forward(camera, splat, bg, 0, 0, 0, 0, false, Tensor{}, true);
    ASSERT_TRUE(forward.has_value()) << lfs::format_for_developer(forward.error());
    ASSERT_TRUE(forward->first.normal.is_valid());
    ASSERT_EQ(forward->first.normal.numel(), 3);

    // Identity rotation, z the smallest axis: world normal is -z (oriented
    // toward the camera at -4z), identity w2c keeps it in place, and the
    // pixel-centered splat blends with weight alpha.
    const auto normal_cpu = forward->first.normal.to(Device::CPU);
    const float* n = normal_cpu.ptr<float>();
    EXPECT_NEAR(n[0], 0.0f, 1.0e-5f);
    EXPECT_NEAR(n[1], 0.0f, 1.0e-5f);
    EXPECT_NEAR(n[2], -scene.opacity_value, 1.0e-4f);
    forward->second.release_forward_context();
}

TEST(FastGSNormalChannelTest, BackwardNormalRotationGradientMatchesFiniteDifferences) {
    if (!torch::cuda::is_available()) {
        GTEST_SKIP() << "CUDA not available";
    }

    NormalChannelScene scene;
    // Generic quaternion away from symmetry; keeps the smallest-axis column
    // pointed toward the camera so the orientation sign stays fixed.
    const std::vector<float> base_quat{0.95f, 0.15f, -0.1f, 0.05f};
    const std::vector<float> upstream{0.7f, -0.4f, 1.1f};
    auto camera = scene.make_camera();
    auto bg = Tensor::zeros({3}, Device::CUDA);

    const auto render_loss = [&](const std::vector<float>& quat) {
        auto splat = scene.make_splat(quat);
        auto forward = fast_rasterize_forward(camera, splat, bg, 0, 0, 0, 0, false, Tensor{}, true);
        if (!forward.has_value()) {
            throw lfs::Exception(std::move(forward.error()));
        }
        const auto normal_cpu = forward->first.normal.to(Device::CPU);
        const float* n = normal_cpu.ptr<float>();
        const float loss = upstream[0] * n[0] + upstream[1] * n[1] + upstream[2] * n[2];
        forward->second.release_forward_context();
        return loss;
    };

    auto splat = scene.make_splat(base_quat);
    auto forward = fast_rasterize_forward(camera, splat, bg, 0, 0, 0, 0, false, Tensor{}, true);
    ASSERT_TRUE(forward.has_value()) << lfs::format_for_developer(forward.error());

    AdamConfig cfg{.lr = 0.001f, .beta1 = 0.9, .beta2 = 0.999, .eps = 1e-15};
    AdamOptimizer opt(splat, cfg);
    opt.allocate_gradients();
    opt.zero_grad(0);

    std::vector<float> upstream_data = upstream;
    auto grad_image = Tensor::zeros_like(forward->first.image);
    auto grad_normal = Tensor::from_blob(upstream_data.data(), {3, 1, 1}, Device::CPU, DataType::Float32).to(Device::CUDA);
    fast_rasterize_backward(
        forward->second,
        grad_image,
        splat,
        opt,
        {},
        {},
        DensificationType::None,
        1,
        {},
        {},
        grad_normal);

    const auto rotation_grad = recovered_fused_grad(opt, ParamType::Rotation).to(Device::CPU);
    const float* actual = rotation_grad.ptr<float>();

    // The splat sits exactly on the pixel center, so the blend weight has zero
    // derivative w.r.t. rotation there and central differences through the full
    // forward isolate exactly the detached-weight value path the kernel emits.
    const float h = 2.0e-2f;
    for (int c = 0; c < 4; ++c) {
        std::vector<float> plus = base_quat;
        std::vector<float> minus = base_quat;
        plus[c] += h;
        minus[c] -= h;
        const float expected = (render_loss(plus) - render_loss(minus)) / (2.0f * h);
        EXPECT_NEAR(actual[c], expected, std::max(2.0e-3f, std::abs(expected) * 2.0e-2f))
            << "rotation gradient mismatch for quaternion component " << c;
    }
}

TEST(FastGSNormalChannelTest, BackwardNormalRotationGradientUsesCompactVisibleIndex) {
    if (!torch::cuda::is_available()) {
        GTEST_SKIP() << "CUDA not available";
    }

    NormalChannelScene single_scene;
    NormalChannelScene scene;
    // R is identity and T is world-to-camera translation: depth = world_z + 4.
    // Rows 0/1 have depth -46 and fail preprocess forward's near-plane culling.
    scene.means_data = {0.0f, 0.0f, -50.0f,
                        0.0f, 0.0f, -50.0f,
                        0.0f, 0.0f, 1.0f};
    scene.scaling_data = {-1.0f, -1.5f, -3.0f,
                          -1.0f, -1.5f, -3.0f,
                          -1.0f, -1.5f, -3.0f};
    const std::vector<float> base_quat{0.95f, 0.15f, -0.1f, 0.05f};
    const std::vector<float> upstream{0.7f, -0.4f, 1.1f};
    std::vector<float> rotations;
    for (int row = 0; row < 3; ++row) {
        rotations.insert(rotations.end(), base_quat.begin(), base_quat.end());
    }
    auto camera = scene.make_camera();
    auto bg = Tensor::zeros({3}, Device::CUDA);

    auto single_splat = single_scene.make_splat(base_quat);
    auto single_forward = fast_rasterize_forward(camera, single_splat, bg, 0, 0, 0, 0, false, Tensor{}, true);
    ASSERT_TRUE(single_forward.has_value()) << lfs::format_for_developer(single_forward.error());
    ASSERT_TRUE(single_forward->first.normal.is_valid());
    ASSERT_EQ(single_forward->first.normal.numel(), 3);
    const auto single_normal_cpu = single_forward->first.normal.to(Device::CPU);
    single_forward->second.release_forward_context();

    const auto render_loss = [&](const std::vector<float>& rotation_data) {
        auto splat = scene.make_splat(rotation_data);
        auto forward = fast_rasterize_forward(camera, splat, bg, 0, 0, 0, 0, false, Tensor{}, true);
        if (!forward.has_value()) {
            throw lfs::Exception(std::move(forward.error()));
        }
        const auto normal_cpu = forward->first.normal.to(Device::CPU);
        const float* n = normal_cpu.ptr<float>();
        const float loss = upstream[0] * n[0] + upstream[1] * n[1] + upstream[2] * n[2];
        forward->second.release_forward_context();
        return loss;
    };

    auto splat = scene.make_splat(rotations);
    auto forward = fast_rasterize_forward(camera, splat, bg, 0, 0, 0, 0, false, Tensor{}, true);
    ASSERT_TRUE(forward.has_value()) << lfs::format_for_developer(forward.error());
    ASSERT_EQ(forward->second.forward_ctx.n_visible, 1);
    ASSERT_TRUE(forward->first.normal.is_valid());
    ASSERT_EQ(forward->first.normal.numel(), 3);
    const auto normal_cpu = forward->first.normal.to(Device::CPU);
    for (int c = 0; c < 3; ++c) {
        EXPECT_FLOAT_EQ(normal_cpu.ptr<float>()[c], single_normal_cpu.ptr<float>()[c])
            << "invisible rows must not change normal channel " << c;
    }

    AdamConfig cfg{.lr = 0.001f, .beta1 = 0.9, .beta2 = 0.999, .eps = 1e-15};
    AdamOptimizer opt(splat, cfg);
    opt.allocate_gradients();
    opt.zero_grad(0);

    std::vector<float> upstream_data = upstream;
    auto grad_image = Tensor::zeros_like(forward->first.image);
    auto grad_normal = Tensor::from_blob(upstream_data.data(), {3, 1, 1}, Device::CPU, DataType::Float32).to(Device::CUDA);
    fast_rasterize_backward(
        forward->second,
        grad_image,
        splat,
        opt,
        {},
        {},
        DensificationType::None,
        1,
        {},
        {},
        grad_normal);

    const auto rotation_grad = recovered_fused_grad(opt, ParamType::Rotation).to(Device::CPU);
    const float* actual = rotation_grad.ptr<float>();
    for (int row = 0; row < 2; ++row) {
        for (int c = 0; c < 4; ++c) {
            EXPECT_EQ(actual[row * 4 + c], 0.0f)
                << "invisible row " << row << " has rotation gradient for component " << c;
        }
    }

    // Only primitive row 2 is visible, so its normal helper is at work_idx 0.
    // As in the single-splat test, pixel centering isolates the normal value path.
    const float h = 2.0e-2f;
    for (int c = 0; c < 4; ++c) {
        std::vector<float> plus = rotations;
        std::vector<float> minus = rotations;
        plus[2 * 4 + c] += h;
        minus[2 * 4 + c] -= h;
        const float expected = (render_loss(plus) - render_loss(minus)) / (2.0f * h);
        EXPECT_NEAR(actual[2 * 4 + c], expected, std::max(2.0e-3f, std::abs(expected) * 2.0e-2f))
            << "rotation gradient mismatch for visible row 2, quaternion component " << c;
    }
}

TEST_F(FastGSKernelTest, Backward_Preprocess) {
    auto r = forward();
    ASSERT_TRUE(r.has_value());

    auto opt = make_optimizer();
    opt->zero_grad(0);
    fast_rasterize_backward(r->second, Tensor::randn_like(r->first.image).mul(0.1f),
                            *splat_, *opt, Tensor::randn_like(r->first.alpha).mul(0.1f));

    EXPECT_TRUE(adam_moment(*opt, ParamType::Means).is_valid());
    EXPECT_TRUE(adam_moment(*opt, ParamType::Rotation).is_valid());
}

TEST_F(FastGSKernelTest, Backward_Full) {
    auto r = forward();
    ASSERT_TRUE(r.has_value());

    auto target = Tensor::randn_like(r->first.image).mul(0.5f).add(0.5f);
    auto grad = (r->first.image - target).mul(2.0f / static_cast<float>(r->first.image.numel()));

    auto opt = make_optimizer();
    opt->zero_grad(0);
    ASSERT_NO_THROW(fast_rasterize_backward(r->second, grad, *splat_, *opt, {}));
}

// Optimizer kernels
TEST_F(FastGSKernelTest, Optimizer_AdamStep) {
    auto opt = make_optimizer();
    opt->zero_grad(0);
    opt->get_grad(ParamType::Scaling).fill_(0.01f);

    auto before = splat_->scaling_raw().clone();
    opt->step(1);
    auto diff = (splat_->scaling_raw() - before).abs().sum().item<float>();

    EXPECT_GT(diff, 0.0f);
}

TEST(AdamCropDampingTest, SetterRequiresExactBooleanRowMaskAndCanClearIt) {
    if (!torch::cuda::is_available()) {
        GTEST_SKIP() << "CUDA not available";
    }

    auto splat = make_adam_test_splat(3);
    AdamOptimizer optimizer(splat, AdamConfig{});

    EXPECT_THROW(
        optimizer.set_crop_damping_mask(Tensor::zeros_bool({2}, Device::CPU)),
        std::runtime_error);
    EXPECT_THROW(
        optimizer.set_crop_damping_mask(Tensor::zeros({3}, Device::CPU)),
        std::runtime_error);
    EXPECT_THROW(
        optimizer.set_crop_damping_mask(Tensor::zeros_bool({3, 1}, Device::CPU)),
        std::runtime_error);

    optimizer.set_crop_damping_mask(Tensor::zeros_bool({3}, Device::CPU));
    EXPECT_TRUE(optimizer.crop_damping_mask().is_valid());
    EXPECT_EQ(optimizer.crop_damping_mask().device(), Device::CUDA);
    EXPECT_TRUE(optimizer.crop_damping_mask().is_contiguous());

    optimizer.set_crop_damping_mask({});
    EXPECT_FALSE(optimizer.crop_damping_mask().is_valid());
}

TEST(AdamCropDampingTest, RepeatedMaskReplacementIsSafeAcrossStreams) {
    if (!torch::cuda::is_available()) {
        GTEST_SKIP() << "CUDA not available";
    }

    cudaStream_t producer_stream = nullptr;
    cudaStream_t execution_stream = nullptr;
    ASSERT_EQ(cudaStreamCreateWithFlags(&producer_stream, cudaStreamNonBlocking), cudaSuccess);
    ASSERT_EQ(cudaStreamCreateWithFlags(&execution_stream, cudaStreamNonBlocking), cudaSuccess);

    {
        auto splat = make_adam_test_splat(4);
        AdamOptimizer optimizer(splat, AdamConfig{});
        optimizer.allocate_gradients();
        optimizer.set_cropbox_lr_scale(0.1f);

        for (int iteration = 1; iteration <= 64; ++iteration) {
            Tensor mask;
            {
                const CUDAStreamGuard producer_guard(producer_stream);
                mask = iteration % 2 == 0
                           ? Tensor::zeros_bool({4}, Device::CUDA)
                           : Tensor::ones_bool({4}, Device::CUDA);
            }
            optimizer.set_crop_damping_mask(std::move(mask));

            {
                const CUDAStreamGuard execution_guard(execution_stream);
                optimizer.zero_grad(iteration - 1);
                optimizer.get_grad(ParamType::Means).fill_(1.0f);
                optimizer.step(iteration);
            }
        }

        ASSERT_EQ(cudaStreamSynchronize(producer_stream), cudaSuccess);
        ASSERT_EQ(cudaStreamSynchronize(execution_stream), cudaSuccess);
        EXPECT_TRUE(splat.means().isfinite().all().item<bool>());
    }

    CudaMemoryPool::instance().release_stream(producer_stream);
    CudaMemoryPool::instance().release_stream(execution_stream);
    ASSERT_EQ(cudaStreamDestroy(producer_stream), cudaSuccess);
    ASSERT_EQ(cudaStreamDestroy(execution_stream), cudaSuccess);
}

TEST(FastGSCropDampingTest, FusedBackwardZeroScaleSkipsContiguousAndSwizzledWrites) {
    if (!torch::cuda::is_available()) {
        GTEST_SKIP() << "CUDA not available";
    }

    struct UpdateResult {
        float opacity_delta = 0.0f;
        float shn_delta = 0.0f;
        float opacity_moment_scale = 0.0f;
        float shn_moment_scale = 0.0f;
    };

    const auto run_backward = [](const bool damp) {
        auto splat = make_adam_test_splat(1, 1);
        auto camera_rotation = Tensor::eye(3, Device::CUDA);
        auto camera_translation = Tensor::from_vector(
            std::vector<float>{0.0f, 0.0f, 4.0f},
            {3},
            Device::CUDA);
        Camera camera(
            camera_rotation,
            camera_translation,
            8.0f,
            8.0f,
            3.5f,
            3.5f,
            {},
            {},
            CameraModelType::PINHOLE,
            "crop_damping",
            "",
            {},
            8,
            8,
            0);
        auto background = Tensor::zeros({3}, Device::CUDA);
        auto forward = fast_rasterize_forward(
            camera, splat, background, 0, 0, 0, 0, false);
        if (!forward) {
            throw lfs::Exception(std::move(forward.error()));
        }

        AdamOptimizer optimizer(splat, AdamConfig{});
        optimizer.allocate_gradients();
        optimizer.zero_grad(1000);
        if (damp) {
            optimizer.set_crop_damping_mask(Tensor::ones_bool({1}, Device::CUDA));
            optimizer.set_cropbox_lr_scale(0.0f);
        }

        const auto opacity_before = splat.opacity_raw().clone();
        const auto shn_before = splat.shN().clone();
        fast_rasterize_backward(
            forward->second,
            Tensor::ones_like(forward->first.image),
            splat,
            optimizer,
            {},
            {},
            DensificationType::None,
            1001);

        UpdateResult result;
        result.opacity_delta =
            (splat.opacity_raw() - opacity_before).abs().sum().item<float>();
        result.shn_delta = (splat.shN() - shn_before).abs().sum().item<float>();
        // Joint codec has no per-primitive moment scales — use decoded |m| L1.
        result.opacity_moment_scale = first_moment_l1(optimizer, ParamType::Opacity);
        result.shn_moment_scale = first_moment_l1(optimizer, ParamType::ShN);
        return result;
    };

    const auto baseline = run_backward(false);
    const auto damped = run_backward(true);
    EXPECT_GT(baseline.opacity_delta, 0.0f);
    EXPECT_GT(baseline.shn_delta, 0.0f);
    EXPECT_GT(baseline.opacity_moment_scale, 0.0f);
    EXPECT_GT(baseline.shn_moment_scale, 0.0f);
    EXPECT_FLOAT_EQ(damped.opacity_delta, 0.0f);
    EXPECT_FLOAT_EQ(damped.shn_delta, 0.0f);
    EXPECT_FLOAT_EQ(damped.opacity_moment_scale, 0.0f);
    EXPECT_FLOAT_EQ(damped.shn_moment_scale, 0.0f);
}

TEST_F(FastGSKernelTest, Optimizer_ZeroRows) {
    auto opt = make_optimizer();
    opt->get_grad(ParamType::Means).fill_(1.0f);
    opt->step(1);

    std::vector<int64_t> indices = {0, 1, 2, 3, 4};
    ASSERT_NO_THROW(opt->reset_state_at_indices(ParamType::Means, indices));
}

// Numerical tests
TEST_F(FastGSKernelTest, Numerical_Deterministic) {
    auto r1 = forward();
    ASSERT_TRUE(r1.has_value());
    auto image1 = r1->first.image.clone();
    r1->second.release_forward_context();

    auto r2 = forward();
    ASSERT_TRUE(r2.has_value());

    float diff = (image1 - r2->first.image).abs().max().item<float>();
    EXPECT_LT(diff, 1e-5f);
}

TEST_F(FastGSKernelTest, Numerical_GradientFinite) {
    auto r = forward();
    ASSERT_TRUE(r.has_value());

    auto opt = make_optimizer();
    opt->zero_grad(0);
    fast_rasterize_backward(r->second, Tensor::randn_like(r->first.image), *splat_, *opt, {});

    auto check = [](const Tensor& t) {
        auto cpu = t.to(Device::CPU);
        auto p = cpu.ptr<float>();
        for (size_t i = 0; i < std::min(t.numel(), size_t(1000)); ++i) {
            EXPECT_TRUE(std::isfinite(p[i]));
        }
    };

    check(adam_moment(*opt, ParamType::Means));
    check(adam_moment(*opt, ParamType::Scaling));
    check(adam_moment(*opt, ParamType::Rotation));
    check(adam_moment(*opt, ParamType::Opacity));
    check(adam_moment(*opt, ParamType::Sh0));
}

// Edge cases
TEST_F(FastGSKernelTest, EdgeCase_SingleGaussian) {
    n_ = 1;
    means_ = Tensor::zeros({1, 3}, Device::CUDA);
    sh0_ = Tensor::zeros({1, 1, 3}, Device::CUDA);
    shN_ = Tensor::zeros({1, 0, 3}, Device::CUDA);
    scaling_ = Tensor::full({1, 3}, -5.0f, Device::CUDA);
    rotation_ = Tensor::zeros({1, 4}, Device::CUDA);
    rotation_.slice(1, 0, 1).fill_(1.0f);
    opacity_ = Tensor::zeros({1}, Device::CUDA);
    splat_ = std::make_unique<SplatData>(0, means_, sh0_, shN_, scaling_, rotation_, opacity_, 1.0f);

    auto r = forward();
    ASSERT_TRUE(r.has_value());
}

TEST_F(FastGSKernelTest, EdgeCase_LargeGaussians) {
    scaling_.fill_(2.0f);
    splat_ = std::make_unique<SplatData>(0, means_, sh0_, shN_, scaling_, rotation_, opacity_, 1.0f);

    auto r = forward();
    ASSERT_TRUE(r.has_value());
    EXPECT_GT(r->second.forward_ctx.n_instances, static_cast<int>(n_));
}

TEST_F(FastGSKernelTest, EdgeCase_CameraBehind) {
    std::vector<float> t_data{0, 0, -10};
    auto T = Tensor::from_blob(t_data.data(), {3}, Device::CPU, DataType::Float32).to(Device::CUDA);
    camera_ = std::make_unique<Camera>(Tensor::eye(3, Device::CUDA), T, FX, FY, W / 2.0f, H / 2.0f,
                                       Tensor(), Tensor(), CameraModelType::PINHOLE,
                                       "behind", "", std::filesystem::path{}, W, H, 0);

    auto r = forward();
    ASSERT_TRUE(r.has_value());
}

// Tiled rendering
TEST_F(FastGSKernelTest, TiledRendering_Single) {
    auto r = fast_rasterize_forward(*camera_, *splat_, bg_, 100, 100, 256, 256, false);
    ASSERT_TRUE(r.has_value());
    EXPECT_EQ(r->first.width, 256);
    EXPECT_EQ(r->first.height, 256);
}

TEST_F(FastGSKernelTest, TiledRendering_Consistency) {
    auto full = forward();
    ASSERT_TRUE(full.has_value());
    auto region = full->first.image.slice(1, 50, 200).slice(2, 100, 300).clone();
    full->second.release_forward_context();

    auto tile = fast_rasterize_forward(*camera_, *splat_, bg_, 100, 50, 200, 150, false);
    ASSERT_TRUE(tile.has_value());

    float diff = (tile->first.image - region).abs().max().item<float>();
    EXPECT_LT(diff, 0.01f);
}

// =============================================================================
// Numerical gradient verification using finite differences
// =============================================================================

namespace {

    torch::Tensor to_torch(const Tensor& t) {
        auto cpu = t.to(Device::CPU);
        std::vector<int64_t> shape;
        for (size_t i = 0; i < t.ndim(); ++i)
            shape.push_back(t.shape()[i]);
        return torch::from_blob(cpu.ptr<float>(), shape, torch::kFloat32).clone().to(torch::kCUDA);
    }

} // namespace

class FastGSGradientTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Small scene for numerical gradient verification
        n_ = 32;
        std::mt19937 gen(123);
        std::uniform_real_distribution<float> pos(-2, 2);

        std::vector<float> means_data(n_ * 3);
        for (size_t i = 0; i < n_ * 3; ++i)
            means_data[i] = pos(gen);
        means_ = Tensor::from_blob(means_data.data(), {n_, 3}, Device::CPU, DataType::Float32).to(Device::CUDA);

        sh0_ = Tensor::randn({n_, 1, 3}, Device::CUDA).mul(0.3f);
        shN_ = Tensor::zeros({n_, 0, 3}, Device::CUDA);
        scaling_ = Tensor::randn({n_, 3}, Device::CUDA).mul(0.2f).sub(3.0f);
        rotation_ = Tensor::randn({n_, 4}, Device::CUDA);
        rotation_ = rotation_ / rotation_.pow(2.0f).sum(-1, true).sqrt();
        opacity_ = Tensor::randn({n_}, Device::CUDA);

        auto R = Tensor::eye(3, Device::CUDA);
        std::vector<float> t_data{0, 0, 4};
        auto T = Tensor::from_blob(t_data.data(), {3}, Device::CPU, DataType::Float32).to(Device::CUDA);
        camera_ = std::make_unique<Camera>(R, T, 200.0f, 200.0f, 64.0f, 64.0f,
                                           Tensor(), Tensor(), CameraModelType::PINHOLE,
                                           "test", "", std::filesystem::path{}, 128, 128, 0);
        bg_ = Tensor::zeros({3}, Device::CUDA);
    }

    void TearDown() override {
        GlobalArenaManager::instance().get_arena().full_reset();
    }

    float compute_loss(const Tensor& means, const Tensor& scaling, const Tensor& rotation,
                       const Tensor& opacity, const Tensor& sh0) {
        auto splat = std::make_unique<SplatData>(0, means, sh0, shN_, scaling, rotation, opacity, 1.0f);
        auto r = fast_rasterize_forward(*camera_, *splat, bg_, 0, 0, 0, 0, false);
        if (!r)
            return 0.0f;
        return r->first.image.pow(2.0f).sum().item<float>();
    }

    Tensor numerical_grad(ParamType param, float eps = 1e-3f) {
        Tensor orig;
        switch (param) {
        case ParamType::Means: orig = means_.clone(); break;
        case ParamType::Scaling: orig = scaling_.clone(); break;
        case ParamType::Rotation: orig = rotation_.clone(); break;
        case ParamType::Opacity: orig = opacity_.clone(); break;
        case ParamType::Sh0: orig = sh0_.clone(); break;
        default: return {};
        }

        Tensor grad = Tensor::zeros_like(orig);
        auto orig_cpu = orig.to(Device::CPU);
        auto grad_cpu = grad.to(Device::CPU);
        float* o_ptr = orig_cpu.ptr<float>();
        float* g_ptr = grad_cpu.ptr<float>();

        for (size_t i = 0; i < orig.numel(); ++i) {
            // Perturb +eps
            auto perturbed = orig_cpu.clone();
            perturbed.ptr<float>()[i] += eps;
            set_param(param, perturbed.to(Device::CUDA));
            float loss_plus = compute_loss(means_, scaling_, rotation_, opacity_, sh0_);

            // Perturb -eps
            perturbed.ptr<float>()[i] = o_ptr[i] - eps;
            set_param(param, perturbed.to(Device::CUDA));
            float loss_minus = compute_loss(means_, scaling_, rotation_, opacity_, sh0_);

            g_ptr[i] = (loss_plus - loss_minus) / (2.0f * eps);
        }

        set_param(param, orig);
        return grad_cpu.to(Device::CUDA);
    }

    void set_param(ParamType param, const Tensor& val) {
        switch (param) {
        case ParamType::Means: means_ = val; break;
        case ParamType::Scaling: scaling_ = val; break;
        case ParamType::Rotation: rotation_ = val; break;
        case ParamType::Opacity: opacity_ = val; break;
        case ParamType::Sh0: sh0_ = val; break;
        default: break;
        }
    }

    Tensor analytical_grad(ParamType param) {
        auto splat = std::make_unique<SplatData>(0, means_, sh0_, shN_, scaling_, rotation_, opacity_, 1.0f);
        auto r = fast_rasterize_forward(*camera_, *splat, bg_, 0, 0, 0, 0, false);
        if (!r)
            return {};

        AdamConfig cfg{.lr = 0.001f, .beta1 = 0.9, .beta2 = 0.999, .eps = 1e-15};
        auto opt = std::make_unique<AdamOptimizer>(*splat, cfg);
        opt->allocate_gradients();
        opt->zero_grad(0);

        auto grad_out = r->first.image.mul(2.0f);
        fast_rasterize_backward(r->second, grad_out, *splat, *opt, {}, {}, DensificationType::None, 1);

        return recovered_fused_grad(*opt, param).clone();
    }

    size_t n_;
    Tensor means_, sh0_, shN_, scaling_, rotation_, opacity_, bg_;
    std::unique_ptr<Camera> camera_;
};

namespace {
    void print_grad_stats(const char* name, const Tensor& num, const Tensor& ana) {
        auto n_cpu = num.to(Device::CPU);
        auto a_cpu = ana.to(Device::CPU);
        float* n = n_cpu.ptr<float>();
        float* a = a_cpu.ptr<float>();

        float max_err = 0, sum_err = 0, num_norm = 0, ana_norm = 0, dot = 0;
        for (size_t i = 0; i < num.numel(); ++i) {
            max_err = std::max(max_err, std::abs(n[i] - a[i]));
            sum_err += std::abs(n[i] - a[i]);
            num_norm += n[i] * n[i];
            ana_norm += a[i] * a[i];
            dot += n[i] * a[i];
        }
        float cos_sim = dot / (std::sqrt(num_norm) * std::sqrt(ana_norm) + 1e-8f);
        printf("  %-10s num_norm=%.4f ana_norm=%.4f max_err=%.5f mean_err=%.5f cos_sim=%.4f\n",
               name, std::sqrt(num_norm), std::sqrt(ana_norm), max_err, sum_err / num.numel(), cos_sim);
    }
} // namespace

TEST_F(FastGSGradientTest, Numerical_Means) {
    auto num = numerical_grad(ParamType::Means);
    auto ana = analytical_grad(ParamType::Means);
    ASSERT_TRUE(num.is_valid() && ana.is_valid());
    print_grad_stats("Means", num, ana);

    auto num_cpu = num.to(Device::CPU);
    auto ana_cpu = ana.to(Device::CPU);
    float* n_ptr = num_cpu.ptr<float>();
    float* a_ptr = ana_cpu.ptr<float>();

    int mismatches = 0;
    for (size_t i = 0; i < num.numel(); ++i) {
        float err = std::abs(n_ptr[i] - a_ptr[i]);
        float rel = err / (std::abs(n_ptr[i]) + 1e-6f);
        if (rel > 0.2f && err > 1e-3f)
            ++mismatches;
    }
    EXPECT_LT(mismatches, static_cast<int>(num.numel() * 0.15f));
}

TEST_F(FastGSGradientTest, Numerical_Scaling) {
    auto num = numerical_grad(ParamType::Scaling);
    auto ana = analytical_grad(ParamType::Scaling);
    ASSERT_TRUE(num.is_valid() && ana.is_valid());
    print_grad_stats("Scaling", num, ana);

    auto diff = (num - ana).abs();
    float mean_err = diff.mean().item<float>();
    EXPECT_LT(mean_err, 1.0f);
}

TEST_F(FastGSGradientTest, Numerical_Opacity) {
    auto num = numerical_grad(ParamType::Opacity);
    auto ana = analytical_grad(ParamType::Opacity);
    ASSERT_TRUE(num.is_valid() && ana.is_valid());
    print_grad_stats("Opacity", num, ana);

    auto num_cpu = num.to(Device::CPU);
    auto ana_cpu = ana.to(Device::CPU);
    float* n_ptr = num_cpu.ptr<float>();
    float* a_ptr = ana_cpu.ptr<float>();

    int mismatches = 0;
    for (size_t i = 0; i < num.numel(); ++i) {
        float err = std::abs(n_ptr[i] - a_ptr[i]);
        float rel = err / (std::abs(n_ptr[i]) + 1e-6f);
        if (rel > 0.1f && err > 1e-4f)
            ++mismatches;
    }
    EXPECT_LT(mismatches, static_cast<int>(num.numel() * 0.1f));
}

TEST_F(FastGSGradientTest, Numerical_Sh0) {
    auto num = numerical_grad(ParamType::Sh0);
    auto ana = analytical_grad(ParamType::Sh0);
    ASSERT_TRUE(num.is_valid() && ana.is_valid());
    print_grad_stats("Sh0", num, ana);

    auto diff = (num - ana).abs();
    float mean_err = diff.mean().item<float>();
    EXPECT_LT(mean_err, 1.0f);
}

TEST_F(FastGSGradientTest, GradientDirection) {
    // Verify gradient descent decreases loss
    auto splat = std::make_unique<SplatData>(0, means_, sh0_, shN_, scaling_, rotation_, opacity_, 1.0f);
    auto r = fast_rasterize_forward(*camera_, *splat, bg_, 0, 0, 0, 0, false);
    ASSERT_TRUE(r.has_value());

    float loss_before = r->first.image.pow(2.0f).sum().item<float>();

    AdamConfig cfg{.lr = 0.01f, .beta1 = 0.9, .beta2 = 0.999, .eps = 1e-15};
    auto opt = std::make_unique<AdamOptimizer>(*splat, cfg);
    opt->allocate_gradients();
    opt->zero_grad(0);

    auto grad_out = r->first.image.mul(2.0f);
    fast_rasterize_backward(r->second, grad_out, *splat, *opt, {}, {}, DensificationType::None, 1);

    r = fast_rasterize_forward(*camera_, *splat, bg_, 0, 0, 0, 0, false);
    ASSERT_TRUE(r.has_value());
    float loss_after = r->first.image.pow(2.0f).sum().item<float>();

    EXPECT_LT(loss_after, loss_before);
}

// =============================================================================
// Dense single-tile gradient test. This exercises the tile backward path with
// many splats contributing to the same pixels.
// =============================================================================

class FastGSDenseTileGradientTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create 128 gaussians concentrated in a SINGLE 16x16 tile
        // This ensures many gaussians end up in the same tile.
        n_ = 128;
        std::mt19937 gen(456);
        // Very small spread - all gaussians within ~1 pixel of each other
        std::uniform_real_distribution<float> tiny_offset(-0.01f, 0.01f);

        // Gaussians at (0, 0, z_i) with z spaced to give stable depth ordering
        // With camera at z=5, focal=100, this projects to pixel ~(32, 32) center of 64x64 image
        std::vector<float> means_data(n_ * 3);
        for (size_t i = 0; i < n_; ++i) {
            means_data[i * 3] = tiny_offset(gen);                  // x: tiny spread
            means_data[i * 3 + 1] = tiny_offset(gen);              // y: tiny spread
            means_data[i * 3 + 2] = static_cast<float>(i) * 0.02f; // z: stable ordering
        }
        means_ = Tensor::from_blob(means_data.data(), {n_, 3}, Device::CPU, DataType::Float32).to(Device::CUDA);

        sh0_ = Tensor::randn({n_, 1, 3}, Device::CUDA).mul(0.3f);
        shN_ = Tensor::zeros({n_, 0, 3}, Device::CUDA);
        // Very small gaussians so they all project to the same tile (scale exp(-5) ≈ 0.007)
        scaling_ = Tensor::full({n_, 3}, -5.0f, Device::CUDA);
        rotation_ = Tensor::zeros({n_, 4}, Device::CUDA);
        rotation_.slice(1, 0, 1).fill_(1.0f);               // Identity rotation (w=1, x=y=z=0)
        opacity_ = Tensor::full({n_}, -3.0f, Device::CUDA); // sigmoid(-3) ≈ 0.047, all contribute

        // Camera looking at origin from z=5
        auto R = Tensor::eye(3, Device::CUDA);
        std::vector<float> t_data{0, 0, 5};
        auto T = Tensor::from_blob(t_data.data(), {3}, Device::CPU, DataType::Float32).to(Device::CUDA);
        // 64x64 image, focal length 100 -> gaussians at (0,0) project to center (32,32)
        // which is in tile (32/16, 32/16) = tile (2, 2)
        camera_ = std::make_unique<Camera>(R, T, 100.0f, 100.0f, 32.0f, 32.0f,
                                           Tensor(), Tensor(), CameraModelType::PINHOLE,
                                           "test", "", std::filesystem::path{}, 64, 64, 0);
        bg_ = Tensor::zeros({3}, Device::CUDA);
    }

    void TearDown() override {
        GlobalArenaManager::instance().get_arena().full_reset();
    }

    float compute_loss(const Tensor& means, const Tensor& scaling, const Tensor& rotation,
                       const Tensor& opacity, const Tensor& sh0) {
        auto splat = std::make_unique<SplatData>(0, means, sh0, shN_, scaling, rotation, opacity, 1.0f);
        auto r = fast_rasterize_forward(*camera_, *splat, bg_, 0, 0, 0, 0, false);
        if (!r)
            return 0.0f;
        return r->first.image.pow(2.0f).sum().item<float>();
    }

    Tensor numerical_grad(ParamType param, float eps = 1e-3f) {
        Tensor orig;
        switch (param) {
        case ParamType::Means: orig = means_.clone(); break;
        case ParamType::Scaling: orig = scaling_.clone(); break;
        case ParamType::Opacity: orig = opacity_.clone(); break;
        case ParamType::Sh0: orig = sh0_.clone(); break;
        default: return {};
        }

        Tensor grad = Tensor::zeros_like(orig);
        auto orig_cpu = orig.to(Device::CPU);
        auto grad_cpu = grad.to(Device::CPU);
        float* o_ptr = orig_cpu.ptr<float>();
        float* g_ptr = grad_cpu.ptr<float>();

        for (size_t i = 0; i < orig.numel(); ++i) {
            auto perturbed = orig_cpu.clone();
            perturbed.ptr<float>()[i] += eps;
            set_param(param, perturbed.to(Device::CUDA));
            float loss_plus = compute_loss(means_, scaling_, rotation_, opacity_, sh0_);

            perturbed.ptr<float>()[i] = o_ptr[i] - eps;
            set_param(param, perturbed.to(Device::CUDA));
            float loss_minus = compute_loss(means_, scaling_, rotation_, opacity_, sh0_);

            g_ptr[i] = (loss_plus - loss_minus) / (2.0f * eps);
        }

        set_param(param, orig);
        return grad_cpu.to(Device::CUDA);
    }

    void set_param(ParamType param, const Tensor& val) {
        switch (param) {
        case ParamType::Means: means_ = val; break;
        case ParamType::Scaling: scaling_ = val; break;
        case ParamType::Opacity: opacity_ = val; break;
        case ParamType::Sh0: sh0_ = val; break;
        default: break;
        }
    }

    Tensor analytical_grad(ParamType param) {
        auto splat = std::make_unique<SplatData>(0, means_, sh0_, shN_, scaling_, rotation_, opacity_, 1.0f);
        auto r = fast_rasterize_forward(*camera_, *splat, bg_, 0, 0, 0, 0, false);
        if (!r)
            return {};

        AdamConfig cfg{.lr = 0.001f, .beta1 = 0.9, .beta2 = 0.999, .eps = 1e-15};
        auto opt = std::make_unique<AdamOptimizer>(*splat, cfg);
        opt->allocate_gradients();
        opt->zero_grad(0);

        auto grad_out = r->first.image.mul(2.0f);
        fast_rasterize_backward(r->second, grad_out, *splat, *opt, {}, {}, DensificationType::None, 1);

        return recovered_fused_grad(*opt, param).clone();
    }

    size_t n_;
    Tensor means_, sh0_, shN_, scaling_, rotation_, opacity_, bg_;
    std::unique_ptr<Camera> camera_;
};

TEST_F(FastGSDenseTileGradientTest, VerifyDenseTileInstances) {
    // Verify this setup actually produces a dense tile workload.
    auto splat = std::make_unique<SplatData>(0, means_, sh0_, shN_, scaling_, rotation_, opacity_, 1.0f);
    auto r = fast_rasterize_forward(*camera_, *splat, bg_, 0, 0, 0, 0, false);
    ASSERT_TRUE(r.has_value());

    printf("  n_instances=%d\n", r->second.forward_ctx.n_instances);

    EXPECT_GT(r->second.forward_ctx.n_instances, 100);
}

TEST_F(FastGSDenseTileGradientTest, Numerical_Means_DenseTile) {
    auto num = numerical_grad(ParamType::Means);
    auto ana = analytical_grad(ParamType::Means);
    ASSERT_TRUE(num.is_valid() && ana.is_valid());

    auto n_cpu = num.to(Device::CPU);
    auto a_cpu = ana.to(Device::CPU);
    float* n = n_cpu.ptr<float>();
    float* a = a_cpu.ptr<float>();

    float max_err = 0, sum_err = 0, num_norm = 0, ana_norm = 0, dot = 0;
    for (size_t i = 0; i < num.numel(); ++i) {
        max_err = std::max(max_err, std::abs(n[i] - a[i]));
        sum_err += std::abs(n[i] - a[i]);
        num_norm += n[i] * n[i];
        ana_norm += a[i] * a[i];
        dot += n[i] * a[i];
    }
    float cos_sim = dot / (std::sqrt(num_norm) * std::sqrt(ana_norm) + 1e-8f);
    printf("  DenseTile Means: num_norm=%.4f ana_norm=%.4f max_err=%.5f mean_err=%.5f cos_sim=%.4f\n",
           std::sqrt(num_norm), std::sqrt(ana_norm), max_err, sum_err / num.numel(), cos_sim);

    EXPECT_GT(cos_sim, 0.80f) << "Gradient direction mismatch in dense tile backward";

    float mean_err = sum_err / num.numel();
    EXPECT_LT(mean_err, 2.0f) << "Mean gradient error too high";
}

TEST_F(FastGSDenseTileGradientTest, Numerical_Opacity_DenseTile) {
    auto num = numerical_grad(ParamType::Opacity);
    auto ana = analytical_grad(ParamType::Opacity);
    ASSERT_TRUE(num.is_valid() && ana.is_valid());

    auto n_cpu = num.to(Device::CPU);
    auto a_cpu = ana.to(Device::CPU);
    float* n = n_cpu.ptr<float>();
    float* a = a_cpu.ptr<float>();

    float max_err = 0, sum_err = 0, num_norm = 0, ana_norm = 0, dot = 0;
    for (size_t i = 0; i < num.numel(); ++i) {
        max_err = std::max(max_err, std::abs(n[i] - a[i]));
        sum_err += std::abs(n[i] - a[i]);
        num_norm += n[i] * n[i];
        ana_norm += a[i] * a[i];
        dot += n[i] * a[i];
    }
    float cos_sim = dot / (std::sqrt(num_norm) * std::sqrt(ana_norm) + 1e-8f);
    printf("  DenseTile Opacity: num_norm=%.4f ana_norm=%.4f max_err=%.5f mean_err=%.5f cos_sim=%.4f\n",
           std::sqrt(num_norm), std::sqrt(ana_norm), max_err, sum_err / num.numel(), cos_sim);

    EXPECT_GT(cos_sim, 0.95f) << "Gradient direction mismatch in dense tile backward";
}

TEST_F(FastGSDenseTileGradientTest, GradientDescent_DenseTile) {
    // Verify gradient descent actually reduces loss with many gaussians per tile
    auto splat = std::make_unique<SplatData>(0, means_, sh0_, shN_, scaling_, rotation_, opacity_, 1.0f);
    auto r = fast_rasterize_forward(*camera_, *splat, bg_, 0, 0, 0, 0, false);
    ASSERT_TRUE(r.has_value());

    float loss_before = r->first.image.pow(2.0f).sum().item<float>();
    printf("  Loss before: %.4f\n", loss_before);
    r->second.release_forward_context();

    AdamConfig cfg{.lr = 0.01f, .beta1 = 0.9, .beta2 = 0.999, .eps = 1e-15};
    auto opt = std::make_unique<AdamOptimizer>(*splat, cfg);
    opt->allocate_gradients();

    // Do several gradient descent steps
    for (int step = 0; step < 10; ++step) {
        r = fast_rasterize_forward(*camera_, *splat, bg_, 0, 0, 0, 0, false);
        ASSERT_TRUE(r.has_value());

        opt->zero_grad(0);
        auto grad_out = r->first.image.mul(2.0f);
        fast_rasterize_backward(r->second, grad_out, *splat, *opt, {}, {}, DensificationType::None, step + 1);
    }

    r = fast_rasterize_forward(*camera_, *splat, bg_, 0, 0, 0, 0, false);
    ASSERT_TRUE(r.has_value());
    float loss_after = r->first.image.pow(2.0f).sum().item<float>();
    printf("  Loss after 10 steps: %.4f (reduction: %.2f%%)\n",
           loss_after, (loss_before - loss_after) / loss_before * 100.0f);

    EXPECT_LT(loss_after, loss_before) << "Gradient descent should reduce loss";
    // Expect at least 10% reduction with 10 steps
    EXPECT_LT(loss_after, loss_before * 0.9f) << "Loss reduction too small - gradients may be wrong";
}

namespace {

    // 1 LSB of (u, log_s) at the live block bounds, mapped through us_to_g1g2 at (0,0)+1LSB.
    void joint_zero_decode_tol(const float* bb, const int bits, float& m_tol, float& v_tol) {
        const float qmax = bits == 16 ? joint_adam::Codec16::kQMax : joint_adam::Codec8::kQMax;
        const float du = (bb[1] - bb[0]) / qmax;
        const float ds = (bb[3] - bb[2]) / qmax;
        float m = 0.0f;
        float v = 0.0f;
        joint_adam::Codec16::us_to_g1g2(du, ds, m, v);
        m_tol = std::max(std::abs(m) * 2.0f, 1e-6f);
        v_tol = std::max(std::abs(v) * 2.0f, 1e-12f);
    }

    int64_t joint_sh_cell(const int p, const int k, const int c, const int slots) {
        constexpr int R = 32;
        const int slot = (p / R) * (slots * R) + k * R + (p % R);
        return static_cast<int64_t>(slot) * 4 + c;
    }

    void fill_realistic_moments(std::vector<float>& m, std::vector<float>& v, const uint32_t seed) {
        std::mt19937 rng(seed);
        std::uniform_real_distribution<float> dm(-1e-3f, 1e-3f);
        std::uniform_real_distribution<float> dv(1e-12f, 1e-4f);
        for (size_t i = 0; i < m.size(); ++i) {
            m[i] = dm(rng);
            v[i] = dv(rng);
        }
    }

    Tensor upload_u8(const std::vector<uint8_t>& host) {
        auto t = Tensor::empty({host.size()}, Device::CPU, DataType::UInt8);
        std::memcpy(t.ptr<uint8_t>(), host.data(), host.size());
        return t.to(Device::CUDA);
    }

    Tensor upload_i64(const std::vector<int64_t>& host) {
        auto t = Tensor::empty({host.size()}, Device::CPU, DataType::Int64);
        std::memcpy(t.ptr<int64_t>(), host.data(), host.size() * sizeof(int64_t));
        return t.to(Device::CUDA);
    }

    void expect_bounds_include_zero(const float* bb, const char* label) {
        EXPECT_LE(bb[0], 0.0f) << label << " umin";
        EXPECT_GE(bb[1], 0.0f) << label << " umax";
        EXPECT_LE(bb[2], 0.0f) << label << " smin";
        EXPECT_GE(bb[3], 0.0f) << label << " smax";
    }

    template <int BITS>
    void expect_reset_cell_zero(const uint8_t* packed, const size_t cell, const float* bb,
                                const char* label) {
        float m = 0.0f;
        float v = 0.0f;
        joint_adam::Codec<BITS>::decode_g1g2(packed, cell, bb[0], bb[1], bb[2], bb[3], m, v);
        float m_tol = 0.0f;
        float v_tol = 0.0f;
        joint_zero_decode_tol(bb, BITS, m_tol, v_tol);
        EXPECT_NEAR(m, 0.0f, m_tol) << label << " cell=" << cell;
        EXPECT_NEAR(v, 0.0f, v_tol) << label << " cell=" << cell;
    }

    template <int BITS>
    void expect_live_cell_preserved(const uint8_t* packed, const size_t cell, const float* bb,
                                    const float m0, const float v0, const char* label) {
        float u0 = 0.0f;
        float s0 = 0.0f;
        joint_adam::Codec<BITS>::g1g2_to_us(m0, v0, u0, s0);
        float u = 0.0f;
        float s = 0.0f;
        joint_adam::Codec<BITS>::decode_us(packed, cell, bb[0], bb[1], bb[2], bb[3], u, s);
        const float qmax = joint_adam::Codec<BITS>::kQMax;
        const float u_tol = 2.0f * (bb[1] - bb[0]) / qmax;
        const float s_tol = 2.0f * (bb[3] - bb[2]) / qmax;
        EXPECT_NEAR(u, u0, u_tol) << label << " cell=" << cell << " u";
        EXPECT_NEAR(s, s0, s_tol) << label << " cell=" << cell << " log_s";
    }

} // namespace

// fails when several threads re-encode the same 256-row block concurrently
TEST(JointEncodeZero, Contiguous16BitMultiIndexSameBlock) {
    constexpr int n_prims = 1024;
    constexpr int n_attr = 3;
    constexpr int bits = 16;
    constexpr int kBS = joint_adam::kBlockSize;
    const int n_blocks = static_cast<int>(joint_adam::n_bounds_for_prims(n_prims));
    const int bpc = joint_adam::Codec16::kBytesPerCell;
    const size_t n_cells = static_cast<size_t>(n_prims) * static_cast<size_t>(n_attr);

    std::vector<float> m(n_cells);
    std::vector<float> v(n_cells);
    fill_realistic_moments(m, v, 20260816u);

    std::vector<uint8_t> packed_h(n_cells * static_cast<size_t>(bpc), 0);
    std::vector<float> bounds_h(static_cast<size_t>(n_blocks) * 4, 0.0f);
    for (int b = 0; b < n_blocks; ++b) {
        const int begin = b * kBS;
        const int end = std::min(begin + kBS, n_prims);
        const size_t n_block_cells = static_cast<size_t>(end - begin) * static_cast<size_t>(n_attr);
        std::vector<float> g1(n_block_cells);
        std::vector<float> g2(n_block_cells);
        size_t t = 0;
        for (int p = begin; p < end; ++p) {
            for (int a = 0; a < n_attr; ++a) {
                const size_t cell = static_cast<size_t>(p) * n_attr + static_cast<size_t>(a);
                g1[t] = m[cell];
                g2[t] = v[cell];
                ++t;
            }
        }
        float bb[4];
        joint_adam::Codec16::reduce_bounds(g1.data(), g2.data(), n_block_cells, bb);
        bounds_h[static_cast<size_t>(b) * 4 + 0] = bb[0];
        bounds_h[static_cast<size_t>(b) * 4 + 1] = bb[1];
        bounds_h[static_cast<size_t>(b) * 4 + 2] = bb[2];
        bounds_h[static_cast<size_t>(b) * 4 + 3] = bb[3];
        for (int p = begin; p < end; ++p) {
            for (int a = 0; a < n_attr; ++a) {
                const size_t cell = static_cast<size_t>(p) * n_attr + static_cast<size_t>(a);
                joint_adam::Codec16::encode_g1g2(packed_h.data(), cell, m[cell], v[cell],
                                                 bb[0], bb[1], bb[2], bb[3]);
            }
        }
    }
    const std::vector<uint8_t> packed_before = packed_h;
    const std::vector<float> bounds_before = bounds_h;

    std::vector<int64_t> indices;
    indices.reserve(180);
    for (int i = 0; i < 120; ++i)
        indices.push_back(i);
    for (int i = 0; i < 60; ++i)
        indices.push_back(2 * kBS + i);
    std::mt19937 shuf(424242u);
    std::shuffle(indices.begin(), indices.end(), shuf);

    auto packed = upload_u8(packed_h);
    auto bounds = Tensor::from_vector(bounds_h,
                                      {static_cast<size_t>(n_blocks), size_t{4}},
                                      Device::CUDA);
    auto idx = upload_i64(indices);

    fast_lfs::optimizer::joint_encode_zero_rows_at_indices(
        packed.ptr<uint8_t>(),
        bounds.ptr<float>(),
        idx.ptr<int64_t>(),
        static_cast<int>(indices.size()),
        n_attr,
        bits,
        n_prims,
        nullptr);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    auto packed_cpu = packed.cpu();
    auto bounds_cpu = bounds.cpu();
    const auto* bytes = packed_cpu.ptr<uint8_t>();
    const float* bb_all = bounds_cpu.ptr<float>();

    const std::unordered_set<int64_t> reset(indices.begin(), indices.end());
    for (int b : {0, 2}) {
        const float* bb = bb_all + b * 4;
        expect_bounds_include_zero(bb, "contiguous touched block");
        const int begin = b * kBS;
        const int end = begin + kBS;
        for (int p = begin; p < end; ++p) {
            for (int a = 0; a < n_attr; ++a) {
                const size_t cell = static_cast<size_t>(p) * n_attr + static_cast<size_t>(a);
                if (reset.count(p) != 0) {
                    expect_reset_cell_zero<16>(bytes, cell, bb, "contiguous reset");
                } else {
                    expect_live_cell_preserved<16>(bytes, cell, bb, m[cell], v[cell],
                                                   "contiguous live");
                }
            }
        }
    }

    for (int b : {1, 3}) {
        EXPECT_EQ(std::memcmp(bb_all + b * 4, bounds_before.data() + static_cast<size_t>(b) * 4,
                              4 * sizeof(float)),
                  0)
            << "untouched bounds block " << b;
        const size_t byte0 = static_cast<size_t>(b) * kBS * n_attr * static_cast<size_t>(bpc);
        const size_t nbytes = static_cast<size_t>(kBS) * n_attr * static_cast<size_t>(bpc);
        EXPECT_EQ(std::memcmp(bytes + byte0, packed_before.data() + byte0, nbytes), 0)
            << "untouched packed block " << b;
    }
}

// fails when several threads re-encode the same 256-row block concurrently
TEST(JointEncodeZero, SwizzledShN8BitMultiIndexSameBlock) {
    constexpr int n_prims = 512;
    constexpr int slots = 2;
    constexpr int bits = 8;
    constexpr int kBS = joint_adam::kBlockSize;
    const int n_blocks = static_cast<int>(joint_adam::n_bounds_for_prims(n_prims));
    const int bpc = joint_adam::Codec8::kBytesPerCell;
    const size_t n_cells = static_cast<size_t>(n_prims) * static_cast<size_t>(slots) * 4u;

    std::vector<float> m(n_cells);
    std::vector<float> v(n_cells);
    fill_realistic_moments(m, v, 20260817u);

    std::vector<uint8_t> packed_h(n_cells * static_cast<size_t>(bpc), 0);
    std::vector<float> bounds_h(static_cast<size_t>(n_blocks) * 4, 0.0f);
    for (int b = 0; b < n_blocks; ++b) {
        const int begin = b * kBS;
        const int end = std::min(begin + kBS, n_prims);
        std::vector<float> g1;
        std::vector<float> g2;
        g1.reserve(static_cast<size_t>(end - begin) * slots * 4);
        g2.reserve(g1.capacity());
        for (int p = begin; p < end; ++p) {
            for (int k = 0; k < slots; ++k) {
                for (int c = 0; c < 4; ++c) {
                    const size_t cell = static_cast<size_t>(joint_sh_cell(p, k, c, slots));
                    g1.push_back(m[cell]);
                    g2.push_back(v[cell]);
                }
            }
        }
        float bb[4];
        joint_adam::Codec8::reduce_bounds(g1.data(), g2.data(), g1.size(), bb);
        bounds_h[static_cast<size_t>(b) * 4 + 0] = bb[0];
        bounds_h[static_cast<size_t>(b) * 4 + 1] = bb[1];
        bounds_h[static_cast<size_t>(b) * 4 + 2] = bb[2];
        bounds_h[static_cast<size_t>(b) * 4 + 3] = bb[3];
        for (int p = begin; p < end; ++p) {
            for (int k = 0; k < slots; ++k) {
                for (int c = 0; c < 4; ++c) {
                    const size_t cell = static_cast<size_t>(joint_sh_cell(p, k, c, slots));
                    joint_adam::Codec8::encode_g1g2(packed_h.data(), cell, m[cell], v[cell],
                                                    bb[0], bb[1], bb[2], bb[3]);
                }
            }
        }
    }

    std::vector<int64_t> indices;
    indices.reserve(120);
    for (int i = 0; i < 100; ++i)
        indices.push_back(i);
    for (int i = 0; i < 20; ++i)
        indices.push_back(kBS + i);
    std::mt19937 shuf(434343u);
    std::shuffle(indices.begin(), indices.end(), shuf);

    auto packed = upload_u8(packed_h);
    auto bounds = Tensor::from_vector(bounds_h,
                                      {static_cast<size_t>(n_blocks), size_t{4}},
                                      Device::CUDA);
    auto idx = upload_i64(indices);

    fast_lfs::optimizer::joint_encode_zero_shN_at_indices(
        packed.ptr<uint8_t>(),
        bounds.ptr<float>(),
        idx.ptr<int64_t>(),
        static_cast<int>(indices.size()),
        slots,
        bits,
        n_prims,
        nullptr);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    auto packed_cpu = packed.cpu();
    auto bounds_cpu = bounds.cpu();
    const auto* bytes = packed_cpu.ptr<uint8_t>();
    const float* bb_all = bounds_cpu.ptr<float>();

    const std::unordered_set<int64_t> reset(indices.begin(), indices.end());
    for (int b : {0, 1}) {
        const float* bb = bb_all + b * 4;
        expect_bounds_include_zero(bb, "shN touched block");
        const int begin = b * kBS;
        const int end = begin + kBS;
        for (int p = begin; p < end; ++p) {
            for (int k = 0; k < slots; ++k) {
                for (int c = 0; c < 4; ++c) {
                    const size_t cell = static_cast<size_t>(joint_sh_cell(p, k, c, slots));
                    if (reset.count(p) != 0) {
                        expect_reset_cell_zero<8>(bytes, cell, bb, "shN reset");
                    } else {
                        expect_live_cell_preserved<8>(bytes, cell, bb, m[cell], v[cell], "shN live");
                    }
                }
            }
        }
    }
}

namespace {
    torch::Tensor normal_regression_cuda(const torch::Tensor& t) { return t.detach().contiguous().to(torch::kCUDA); }
    double normal_regression_error(const torch::Tensor& actual, const torch::Tensor& expected) {
        return (actual.cpu() - expected.detach()).abs().max().item<double>();
    }
} // namespace

// Guards direct normal-prior loss and gradients against CPU autograd with optional pixel weights.
// Invalid pixels and batches below the participation threshold must have zero gradients.
TEST(NormalLossRegression, PriorAutogradAndInactivePixels) {
    torch::manual_seed(2309);
    constexpr int h = 17, w = 19, p = h * w;
    for (bool weighted : {false, true}) {
        SCOPED_TRACE(::testing::Message() << "weighted=" << weighted);
        auto n = torch::randn({3, p}) * .4;
        auto a = torch::rand({p}) * .7 + .3;
        auto t = torch::randn({3, p});
        auto weights = torch::rand({p});
        n.index_put_({torch::indexing::Slice(), 0}, 1e-8);
        a.index_put_({1}, 1e-8);
        t.index_put_({torch::indexing::Slice(), 2}, 0);
        n.index_put_({torch::indexing::Slice(), 3}, .059); // just above norm threshold
        weights.index_put_({4}, 0);
        n.set_requires_grad(true);
        auto nn = n.norm(2, 0), tn = t.norm(2, 0);
        auto valid = (nn >= .1) & (tn >= .5) & (a > .001);
        auto aw = a * (weighted ? weights : torch::ones_like(a)) * valid;
        auto cos = (n * t).sum(0) / (nn.clamp_min(1e-12) * tn.clamp_min(1e-12));
        auto loss = .005 * (aw * (1 - cos)).sum() / aw.sum().clamp_min(1);
        loss.backward();
        auto nc = normal_regression_cuda(n), ac = normal_regression_cuda(a), tc = normal_regression_cuda(t), wc = normal_regression_cuda(weights);
        auto gn = torch::full_like(nc, 123), out = torch::zeros({1}, nc.options());
        auto part = torch::zeros({(long)kernels::normal_loss_partial_count(p)}, nc.options());
        kernels::launch_normal_loss(nc.data_ptr<float>(), ac.data_ptr<float>(), tc.data_ptr<float>(),
                                    gn.data_ptr<float>(), out.data_ptr<float>(), part.data_ptr<float>(), w, h, .005, nullptr,
                                    weighted ? wc.data_ptr<float>() : nullptr);
        ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
        double err = normal_regression_error(gn, n.grad());
        EXPECT_LT(err, 2e-8);
        EXPECT_NEAR(out.item<float>(), loss.item<float>(), 2e-8);
        // Fewer than 64 participating pixels must overwrite every gradient with zero.
        ac.zero_();
        ac.slice(0, 0, 63).fill_(.8);
        kernels::launch_normal_loss(nc.data_ptr<float>(), ac.data_ptr<float>(), tc.data_ptr<float>(),
                                    gn.data_ptr<float>(), out.data_ptr<float>(), part.data_ptr<float>(), w, h, .005);
        ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
        EXPECT_EQ(gn.abs().max().item<float>(), 0);
        EXPECT_EQ(out.item<float>(), 0);
    }
}

// Guards consistency and prior-depth geometry gradients against CPU autograd across depth scales.
// The reference includes invalid stencils and detached coverage weights and facing decisions.
TEST(NormalLossRegression, DepthConsistencyAndPriorDepthAutograd) {
    using torch::indexing::Slice;
    torch::manual_seed(2310);
    constexpr int h = 17, w = 19, p = h * w;
    for (bool prior : {false, true})
        for (float z : {0.01f, 5.f, 500.f}) {
            SCOPED_TRACE(::testing::Message() << "prior=" << prior << " depth=" << z);
            auto a = torch::rand({h, w}) * .3 + .65;
            auto d = (torch::rand({h, w}) * .004 + .998) * z * a;
            auto n = torch::randn({3, h, w}) * .4;
            a.index_put_({4, 4}, .1);                // alpha gate, including neighbor propagation
            d.index_put_({8, 8}, z * 1.3 * a[8][8]); // jump gate
            n.index_put_({Slice(), 10, 10}, 0);      // invalid normal/prior
            a.set_requires_grad(true);
            d.set_requires_grad(true);
            n.set_requires_grad(!prior);
            auto xs = torch::arange(w).view({1, w}).expand({h, w});
            auto ys = torch::arange(h).view({h, 1}).expand({h, w});
            constexpr float fx = 300, fy = 280, cx = 9.5, cy = 8.5;
            auto ray = torch::stack({(xs + .5 - cx) / fx, (ys + .5 - cy) / fy, torch::ones({h, w})});
            auto e = d.clamp_min(0) / a;
            auto point = e.unsqueeze(0) * ray;
            auto tx = point.index({Slice(), Slice(1, h - 1), Slice(2, w)}) - point.index({Slice(), Slice(1, h - 1), Slice(0, w - 2)});
            auto ty = point.index({Slice(), Slice(2, h), Slice(1, w - 1)}) - point.index({Slice(), Slice(0, h - 2), Slice(1, w - 1)});
            auto raw = torch::cross(tx, ty, 0);
            auto sign = torch::where((raw * ray.index({Slice(), Slice(1, h - 1), Slice(1, w - 1)})).sum(0) > 0, -1., 1.);
            auto nd = raw / raw.norm(2, 0).clamp_min(1e-30) * sign;
            auto mid = n.index({Slice(), Slice(1, h - 1), Slice(1, w - 1)});
            auto nn = mid.norm(2, 0);
            auto cos = (nd * mid).sum(0) / nn.clamp_min(1e-12);
            auto ec = e.index({Slice(1, h - 1), Slice(1, w - 1)});
            auto ac = a.index({Slice(1, h - 1), Slice(1, w - 1)});
            auto valid = (ac >= .5) & (ec >= 1e-6) & (nn >= (prior ? .5 : .1)) & (raw.square().sum(0) >= 1e-24);
            for (auto [dy, dx] : std::vector<std::pair<int, int>>{{0, 1}, {0, -1}, {1, 0}, {-1, 0}}) {
                auto en = e.index({Slice(1 + dy, h - 1 + dy), Slice(1 + dx, w - 1 + dx)});
                auto an = a.index({Slice(1 + dy, h - 1 + dy), Slice(1 + dx, w - 1 + dx)});
                valid = valid & (an >= .5) & (en >= 1e-6) & ((en - ec).abs() <= .05 * ec);
            }
            // Coverage weights and validity decisions are intentionally detached in the kernels.
            auto aw = ac.detach() * valid.detach();
            auto loss = .005 * (aw * (1 - cos)).sum() / aw.sum().clamp_min(1);
            loss.backward();
            auto nc = normal_regression_cuda(n), dc = normal_regression_cuda(d), alc = normal_regression_cuda(a);
            auto gn = torch::zeros_like(nc), gd = torch::zeros_like(dc), ga = torch::zeros_like(alc);
            auto out = torch::zeros({1}, nc.options());
            auto part = torch::zeros({(long)kernels::normal_consistency_partial_count(p)}, nc.options());
            if (prior)
                kernels::launch_normal_prior_depth_loss(nc.data_ptr<float>(), dc.data_ptr<float>(), alc.data_ptr<float>(),
                                                        gd.data_ptr<float>(), ga.data_ptr<float>(), out.data_ptr<float>(), part.data_ptr<float>(), w, h, fx, fy, cx, cy, .005);
            else
                kernels::launch_normal_consistency_loss(nc.data_ptr<float>(), dc.data_ptr<float>(), alc.data_ptr<float>(),
                                                        gn.data_ptr<float>(), gd.data_ptr<float>(), ga.data_ptr<float>(), out.data_ptr<float>(), part.data_ptr<float>(), w, h, fx, fy, cx, cy, .005);
            ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
            const double ed = normal_regression_error(gd, d.grad()), ea = normal_regression_error(ga, a.grad());
            EXPECT_NEAR(out.item<float>(), loss.item<float>(), 2e-7);
            EXPECT_LT(ed, 2e-4 * std::max(1e-3, d.grad().abs().max().item<double>()));
            EXPECT_LT(ea, 2e-4 * std::max(1e-3, a.grad().abs().max().item<double>()));
            if (!prior)
                EXPECT_LT(normal_regression_error(gn, n.grad()), 2e-8);
        }
}

// Guards fused normal-channel gradients for means, scales, rotations, and opacity using finite differences.
// Overlapping off-center splats exercise all minimum axes, facing signs, and near-opaque coverage.
TEST(NormalLossRegression, OffCenterOverlappingFullFusedGradients) {
    for (int scenario = 0; scenario < 5; ++scenario) {
        NormalChannelScene scene;
        scene.means_data = {.35f, -.22f, 1.f, -.28f, .3f, 1.3f, .16f, .12f, 1.7f};
        scene.scaling_data = {.3f, -.1f, -1.f, .1f, .4f, -.8f, .2f, 0.f, -.9f};
        std::vector<float> q = {.95f, .15f, -.1f, .05f, .15f, .95f, .05f, -.1f, .9f, -.2f, .15f, .1f};
        std::vector<float> op = {-.3f, .4f, .1f};
        if (scenario == 1)
            op[0] = 8.f;
        if (scenario == 2)
            for (int i = 0; i < 3; ++i) {
                scene.scaling_data[3 * i] = -1.f;
                scene.scaling_data[3 * i + 2] = .3f;
            }
        if (scenario == 3)
            for (int i = 0; i < 3; ++i) {
                scene.scaling_data[3 * i + 1] = -1.f;
                scene.scaling_data[3 * i + 2] = .3f;
            }
        if (scenario == 4) {
            scene.scaling_data[0] = -.799f;
            scene.scaling_data[1] = -.8f;
            scene.scaling_data[2] = .3f;
        }
        auto camera = scene.make_camera();
        auto bg = Tensor::zeros({3}, Device::CUDA);
        auto make = [&](const std::vector<float>& m, const std::vector<float>& s, const std::vector<float>& r, const std::vector<float>& o) {
            auto local = scene;
            local.means_data = m;
            local.scaling_data = s;
            auto splat = local.make_splat(r);
            splat.opacity_raw() = Tensor::from_vector(o, {size_t{3}}, Device::CUDA);
            return splat;
        };
        std::vector<float> upstream = {.7f, -.4f, 1.1f};
        auto loss = [&](const std::vector<float>& m, const std::vector<float>& s, const std::vector<float>& r, const std::vector<float>& o) {
            auto splat = make(m, s, r, o);
            auto f = fast_rasterize_forward(camera, splat, bg, 0, 0, 0, 0, false, Tensor{}, true);
            if (!f)
                throw lfs::Exception(std::move(f.error()));
            auto nc = f->first.normal.cpu();
            double v = 0;
            for (int c = 0; c < 3; ++c)
                v += nc.ptr<float>()[c] * upstream[c];
            f->second.release_forward_context();
            return v;
        };
        auto splat = make(scene.means_data, scene.scaling_data, q, op);
        auto f = fast_rasterize_forward(camera, splat, bg, 0, 0, 0, 0, false, Tensor{}, true);
        ASSERT_TRUE(f.has_value());
        AdamConfig cfg{.lr = .001f, .beta1 = .9, .beta2 = .999, .eps = 1e-15};
        AdamOptimizer opt(splat, cfg);
        opt.allocate_gradients();
        opt.zero_grad(0);
        auto gn = Tensor::from_vector(upstream, {size_t{3}, size_t{1}, size_t{1}}, Device::CUDA);
        fast_rasterize_backward(f->second, Tensor::zeros_like(f->first.image), splat, opt, {}, {}, DensificationType::None, 1, {}, {}, gn);
        int group = 0;
        for (auto type : {ParamType::Means, ParamType::Scaling, ParamType::Rotation, ParamType::Opacity}) {
            auto g = recovered_fused_grad(opt, type).cpu();
            for (size_t c = 0; c < g.numel(); ++c) {
                auto m = scene.means_data, s = scene.scaling_data, r = q, o = op;
                auto* v = group == 0 ? &m : group == 1 ? &s
                                        : group == 2   ? &r
                                                       : &o;
                const float step = scenario == 4 && group == 1 ? 1e-4f : 1e-3f;
                (*v)[c] += step;
                double plus = loss(m, s, r, o);
                (*v)[c] -= 2 * step;
                double minus = loss(m, s, r, o);
                double fd = (plus - minus) / (2 * step), actual = g.ptr<float>()[c];
                EXPECT_NEAR(actual, fd, std::max(3e-4, std::abs(fd) * .015)) << "scenario=" << scenario << " group=" << group << " c=" << c;
            }
            ++group;
        }
    }
}

// Guards exact model-row permutation and bounded joint-moment drift through four 50,000-row Morton reorders.
// Covers means, SH0, scales, rotations, and opacity with distinguishable row payloads.
TEST(NormalLossRegression, Morton50kFourReordersJointAllParameters) {
    constexpr size_t n = 50000;
    auto splat = make_adam_test_splat(n);
    lfs::core::param::OptimizationParameters params;
    params.max_cap = n;
    params.sh_degree = 0;
    MRNF strategy(splat);
    strategy.initialize(params);
    auto& opt = strategy.get_optimizer();
    opt.allocate_gradients(n);
    std::mt19937 rng(2311);
    std::uniform_real_distribution<float> unit(-1, 1);
    std::vector<float> pos(n * 3);
    for (auto& x : pos)
        x = unit(rng);
    splat.means() = Tensor::from_vector(pos, {n, size_t{3}}, Device::CUDA);
    // Unique finite row payloads make a wrong permutation observable even with SH0.
    for (auto* t : {&splat.sh0(), &splat.scaling_raw(), &splat.rotation_raw(), &splat.opacity_raw()}) {
        auto host = t->cpu();
        for (size_t c = 0; c < host.numel(); ++c)
            host.ptr<float>()[c] = unit(rng);
        *t = host.cuda();
    }
    std::vector<ParamType> types = {ParamType::Means, ParamType::Sh0, ParamType::Scaling, ParamType::Rotation, ParamType::Opacity};
    std::vector<Tensor> originals;
    std::vector<std::vector<float>> initial_m, initial_v;
    auto model_tensors = [&]() { return std::vector<Tensor*>{&splat.means(), &splat.sh0(), &splat.scaling_raw(), &splat.rotation_raw(), &splat.opacity_raw()}; };
    for (auto* t : model_tensors())
        originals.push_back(t->cpu());
    auto decode = [&](const AdamParamState& state) {
        auto pc = state.exp_avg.cpu(), bc = state.joint_bounds.cpu();
        size_t attrs = pc.shape()[1] / 4;
        std::vector<float> m(n * attrs), v(n * attrs);
        for (size_t c = 0; c < m.size(); ++c) {
            const float* b = bc.ptr<float>() + 4 * ((c / attrs) / 256);
            joint_adam::Codec16::decode_g1g2(pc.ptr<uint8_t>(), c, b[0], b[1], b[2], b[3], m[c], v[c]);
        }
        return std::make_pair(m, v);
    };
    for (auto type : types) {
        auto* state = opt.get_state_mutable(type);
        ASSERT_NE(state, nullptr);
        ASSERT_TRUE(state->is_joint());
        ASSERT_EQ(state->joint_bits, 16);
        auto pc = state->exp_avg.cpu(), bc = state->joint_bounds.cpu();
        size_t attrs = pc.shape()[1] / 4;
        for (size_t b = 0; b < (n + 255) / 256; ++b) {
            size_t start = b * 256 * attrs, stop = std::min(n, (b + 1) * 256) * attrs;
            std::vector<float> m(stop - start), v(stop - start);
            for (size_t j = 0; j < m.size(); ++j) {
                float scale = j < attrs ? 1e-4f : 1e-10f;
                m[j] = unit(rng) * scale;
                v[j] = scale * scale;
            }
            float* mm = bc.ptr<float>() + b * 4;
            joint_adam::Codec16::reduce_bounds(m.data(), v.data(), m.size(), mm);
            for (size_t j = 0; j < m.size(); ++j)
                joint_adam::Codec16::encode_g1g2(pc.ptr<uint8_t>(), start + j, m[j], v[j], mm[0], mm[1], mm[2], mm[3]);
        }
        state->exp_avg = pc.cuda();
        state->joint_bounds = bc.cuda();
        auto [m, v] = decode(*state);
        initial_m.push_back(std::move(m));
        initial_v.push_back(std::move(v));
    }
    std::vector<int64_t> order(n);
    std::iota(order.begin(), order.end(), 0);
    for (int round = 0; round < 4; ++round) {
        SCOPED_TRACE(::testing::Message() << "round=" << round);
        // Rotate spatial axes between passes to make all four reorderings nontrivial.
        if (round) {
            auto mc = splat.means().cpu();
            float* p = mc.ptr<float>();
            for (size_t i = 0; i < n; ++i)
                std::swap(p[3 * i], p[3 * i + 1]);
            splat.means() = mc.cuda();
            auto* orig = originals[0].ptr<float>();
            for (size_t i = 0; i < n; ++i)
                std::swap(orig[3 * i], orig[3 * i + 1]);
        }
        auto result = morton::apply_morton_reorder(splat, &opt);
        ASSERT_TRUE(result.applied);
        strategy.permute_gaussian_rows(result.permutation);
        auto p = result.permutation.cpu();
        auto previous = order;
        for (size_t i = 0; i < n; ++i)
            order[i] = previous[p.ptr<int64_t>()[i]];
        auto tensors = model_tensors();
        size_t bad = 0;
        for (size_t t = 0; t < tensors.size(); ++t) {
            auto h = tensors[t]->cpu();
            size_t attrs = h.numel() / n;
            for (size_t i = 0; i < n; ++i)
                if (std::memcmp(h.ptr<float>() + i * attrs, originals[t].ptr<float>() + order[i] * attrs, attrs * 4))
                    ++bad;
        }
        EXPECT_EQ(bad, 0); // Host inverse permutation must be bitwise exact for raw model attributes.
        for (size_t t = 0; t < types.size(); ++t) {
            SCOPED_TRACE(::testing::Message() << "parameter group=" << t);
            auto [m, v] = decode(*opt.get_state(types[t]));
            size_t attrs = m.size() / n;
            double em = 0, eu = 0;
            for (size_t i = 0; i < n; ++i)
                for (size_t a = 0; a < attrs; ++a) {
                    size_t c = i * attrs + a, old = order[i] * attrs + a;
                    ASSERT_TRUE(std::isfinite(m[c]));
                    ASSERT_TRUE(std::isfinite(v[c]));
                    ASSERT_GE(v[c], 0);
                    em = std::max(em, std::abs(double(m[c]) - initial_m[t][old]));

                    double u = m[c] / (std::sqrt(v[c]) + 1e-15), u0 = initial_m[t][old] / (std::sqrt(initial_v[t][old]) + 1e-15);
                    eu = std::max(eu, std::abs(u - u0));
                }
            EXPECT_LT(em, 1e-7);
            EXPECT_LT(eu, .001);
        }
    }
}

// Guards the direct normal-prior cutoff below 64 valid pixels and populated prior-depth statistics.
// Prior-depth statistics are checked independently of its activation gate.
TEST(NormalLossRegression, SparsePriorDisablesAndPriorDepthStatsRemainPopulated) {
    constexpr int h = 17, w = 19, p = h * w;
    for (int count : {1, 16, 64, 255})
        for (float z : {.01f, 5.f, 500.f}) {
            SCOPED_TRACE(::testing::Message() << "count=" << count << " depth=" << z);
            auto n = torch::zeros({3, h, w});
            n[2].fill_(-.8f);
            auto t = torch::zeros_like(n);
            int left = count;
            for (int y = 1; y < h - 1 && left; ++y)
                for (int x = 1; x < w - 1 && left; ++x, --left) {
                    t[0][y][x] = .70710678f;
                    t[2][y][x] = -.70710678f;
                }
            auto nc = normal_regression_cuda(n), tc = normal_regression_cuda(t);
            auto a = torch::full({h, w}, .8f, nc.options()), d = torch::full({h, w}, .8f * z, nc.options());
            auto gn = torch::zeros_like(nc), gd = torch::zeros_like(d), ga = torch::zeros_like(a);
            auto out = torch::zeros({1}, nc.options()), priorout = torch::zeros_like(out);
            auto part = torch::zeros({(long)kernels::normal_consistency_partial_count(p)}, nc.options());
            kernels::launch_normal_loss(nc.data_ptr<float>(), a.data_ptr<float>(), tc.data_ptr<float>(), gn.data_ptr<float>(),
                                        priorout.data_ptr<float>(), part.data_ptr<float>(), w, h, .05f);
            part.zero_();
            kernels::launch_normal_prior_depth_loss(tc.data_ptr<float>(), d.data_ptr<float>(), a.data_ptr<float>(),
                                                    gd.data_ptr<float>(), ga.data_ptr<float>(), out.data_ptr<float>(), part.data_ptr<float>(), w, h, 1000, 1000, 9.5, 8.5, .05f);
            ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
            EXPECT_EQ(part[kernels::normal_consistency_slots::kCount].item<float>(), count);
            EXPECT_NEAR(part[kernels::normal_consistency_slots::kSumAlpha].item<float>(), .8f * count, 2e-5);
            EXPECT_NEAR(part[kernels::normal_consistency_slots::kMeanCos].item<float>(), std::sqrt(.5f), 2e-6);
            if (count < 64) {
                EXPECT_EQ(priorout.item<float>(), 0);
                EXPECT_EQ(gn.abs().max().item<float>(), 0);
            } else {
                EXPECT_GT(priorout.item<float>(), 0);
                EXPECT_GT(gn.abs().max().item<float>(), 0);
            }
        }
}

// Guards deterministic minimum-axis tie selection and the camera-facing sign change across grazing angles.
TEST(NormalLossRegression, AxisTieAndGrazingBranchDiscontinuities) {
    NormalChannelScene scene;
    scene.means_data = {0, 0, 1};
    scene.opacity_value = .8;
    auto camera = scene.make_camera();
    auto bg = Tensor::zeros({3}, Device::CUDA);
    auto normal = [&](std::vector<float> scales, std::vector<float> q) {
        scene.scaling_data = scales;
        auto splat = scene.make_splat(q);
        auto f = fast_rasterize_forward(camera, splat, bg, 0, 0, 0, 0, false, Tensor{}, true);
        if (!f)
            throw lfs::Exception(std::move(f.error()));
        auto v = f->first.normal.cpu();
        std::vector<float> result(v.ptr<float>(), v.ptr<float>() + 3);
        f->second.release_forward_context();
        return result;
    };
    auto a = normal({-1, -1, 0}, {1, 0, 0, 0});
    auto b = normal({-.999999f, -1, 0}, {1, 0, 0, 0});
    EXPECT_NEAR(a[0], .8, 1e-5);
    EXPECT_NEAR(b[1], .8, 1e-5);
    auto g1 = normal({0, 0, -1}, {.70710678f + .000001f, .70710678f, 0, 0});
    auto g2 = normal({0, 0, -1}, {.70710678f - .000001f, .70710678f, 0, 0});
    float grazing_jump = 0;
    for (int c = 0; c < 3; ++c)
        grazing_jump += (g1[c] - g2[c]) * (g1[c] - g2[c]);
    EXPECT_GT(std::sqrt(grazing_jump), 1.5);
}

TEST(NormalLossHunt, JointRotationCodec100kSteps) {
    using C = joint_adam::Codec16;
    constexpr int cells = 256 * 4, steps = 100000;
    for (int mode = 0; mode < 3; ++mode) {
        std::vector<float> m(cells, 0), v(cells, 0), mf(cells, 0), vf(cells, 0);
        std::vector<double> param(cells, 0), reference(cells, 0);
        std::vector<std::uint8_t> packed(cells * 4, 0);
        float mm[4] = {0, 0, 0, 0};
        double max_zero_step = 0;
        for (int step = 1; step <= steps; ++step) {
            const float b1 = .9f, b2 = .999f, eps = 1e-15f, lr = .002f;
            const float bc1 = 1 - std::pow(b1, step), bc2 = std::sqrt(1 - std::pow(b2, step));
            for (int c = 0; c < cells; ++c) {
                C::decode_g1g2(packed.data(), c, mm[0], mm[1], mm[2], mm[3], m[c], v[c]);
                // Four large components, four 1e6-times smaller, remaining exact-zero gradients.
                float g = c < 4 ? ((c % 2) ? -7e-5f : 1e-4f) : c < 8 ? ((c % 2) ? -7e-11f : 1e-10f)
                                                                     : 0.f;
                if (mode == 1)
                    g *= std::sin(step * .017f + c * .31f);
                if (mode == 2 && step > 1000)
                    g = c < 4 ? g : 0.f;
                m[c] = b1 * m[c] + (1 - b1) * g;
                v[c] = b2 * v[c] + (1 - b2) * g * g;
                mf[c] = b1 * mf[c] + (1 - b1) * g;
                vf[c] = b2 * vf[c] + (1 - b2) * g * g;
                const double delta = lr / bc1 * m[c] / (std::sqrt(v[c]) / bc2 + eps);
                param[c] -= delta;
                reference[c] -= lr / bc1 * mf[c] / (std::sqrt(vf[c]) / bc2 + eps);
                if (c == 8)
                    max_zero_step = std::max(max_zero_step, std::abs(delta));
            }
            C::reduce_bounds(m.data(), v.data(), cells, mm);
            for (int c = 0; c < cells; ++c)
                C::encode_g1g2(packed.data(), c, m[c], v[c], mm[0], mm[1], mm[2], mm[3]);
        }
        std::cout << "HUNT codec mode=" << mode << " zero_drift=" << param[8] << " max_zero_step=" << max_zero_step
                  << " tiny_drift=" << param[4] << " tiny_fp32=" << reference[4] << " tiny_error=" << param[4] - reference[4]
                  << " tiny_final_m=" << m[4] << " tiny_final_v=" << v[4] << " zero_final_m=" << m[8]
                  << " bounds=" << mm[0] << ',' << mm[1] << ',' << mm[2] << ',' << mm[3] << '\n';
        EXPECT_TRUE(std::isfinite(param[4]));
        EXPECT_LT(std::abs(param[8]), 1e-6);
        EXPECT_EQ(max_zero_step, 0.0);
        EXPECT_EQ(m[8], 0.0f);
        EXPECT_EQ(v[8], 0.0f);
        // Measured pre-fix errors, rounded upward only to the printed precision.
        constexpr double baseline_error[] = {30.135, 0.000488, 0.021965};
        EXPECT_LE(std::abs(param[4] - reference[4]), baseline_error[mode]);
    }
}

TEST(NormalLossHunt, JointRotationCuda100kSteps) {
    constexpr int rows = 256, attrs = 4, cells = rows * attrs;
    auto p = Tensor::zeros({size_t{rows}, size_t{attrs}}, Device::CUDA);
    auto packed = Tensor::zeros({size_t{rows}, size_t{attrs * 4}}, Device::CUDA, DataType::UInt8);
    auto bounds = Tensor::zeros({size_t{1}, size_t{4}}, Device::CUDA);
    std::vector<float> g(cells, 0);
    for (int c = 0; c < 8; ++c)
        g[c] = (c % 2 ? -7.f : 10.f) * (c < 4 ? 1e-5f : 1e-11f);
    auto grad = Tensor::from_vector(g, {size_t{rows}, size_t{attrs}}, Device::CUDA);
    for (int step = 1; step <= 100000; ++step) {
        float bc1 = 1 / (1 - std::pow(.9f, step)), bc2 = 1 / std::sqrt(1 - std::pow(.999f, step));
        fast_lfs::optimizer::adam_step_joint_contiguous_raw(p.ptr<float>(), packed.ptr<uint8_t>(), bounds.ptr<float>(), grad.ptr<float>(),
                                                            nullptr, 0, 1, nullptr, 0, 1, rows, attrs, 16, .002, .9, .999, 1e-15, bc1, bc2);
    }
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    auto pc = p.cpu(), bc = bounds.cpu(), qc = packed.cpu();
    float m, v;
    const float* b = bc.ptr<float>();
    joint_adam::Codec16::decode_g1g2(qc.ptr<uint8_t>(), 4, b[0], b[1], b[2], b[3], m, v);
    std::cout << "HUNT CUDA codec zero_drift=" << pc.ptr<float>()[8] << " tiny_drift=" << pc.ptr<float>()[4]
              << " tiny_final_m=" << m << " tiny_final_v=" << v << " bounds=" << b[0] << ',' << b[1] << ',' << b[2] << ',' << b[3] << '\n';
    EXPECT_TRUE(std::isfinite(pc.ptr<float>()[4]));
    EXPECT_LT(std::abs(pc.ptr<float>()[8]), 1e-6);
    // Host FP32-moment reference is -200.003; pre-fix CUDA travel is -229.930.
    EXPECT_LE(std::abs(pc.ptr<float>()[4] + 200.003f), 29.929f);

    // Starting a real gradient after zero history must not skip its first step.
    g[8] = 1e-10f;
    grad = Tensor::from_vector(g, {size_t{rows}, size_t{attrs}}, Device::CUDA);
    fast_lfs::optimizer::adam_step_joint_contiguous_raw(
        p.ptr<float>(), packed.ptr<uint8_t>(), bounds.ptr<float>(), grad.ptr<float>(),
        nullptr, 0, 1, nullptr, 0, 1, rows, attrs, 16, .002f, .9f, .999f, 1e-15f, 1, 1);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    const auto resumed = p.cpu();
    const float expected_delta = .002f * (1.0f - .9f) * g[8] /
                                 (std::sqrt((1.0f - .999f) * g[8] * g[8]) + 1e-15f);
    EXPECT_NEAR(resumed.ptr<float>()[8] - pc.ptr<float>()[8], -expected_delta, 1e-8f);
}

TEST(JointAdamUpdates, ZeroHistoryStaysFixedInOrdinaryAndFusedAllGroups) {
    if (!torch::cuda::is_available()) {
        GTEST_SKIP() << "CUDA not available";
    }
    constexpr size_t rows = 256;
    constexpr int steps = 16;
    const std::vector<ParamType> types = {ParamType::Means, ParamType::Scaling,
                                          ParamType::Rotation, ParamType::Opacity,
                                          ParamType::Sh0, ParamType::ShN};
    for (const bool fused : {false, true}) {
        SCOPED_TRACE(fused ? "fused backward" : "ordinary optimizer");
        auto splat = make_adam_test_splat(rows, 3);
        // Exercise both visible and invisible zero-gradient rows in fused preprocessing.
        auto means = splat.means().cpu();
        for (size_t row = rows / 2; row < rows; ++row)
            means.ptr<float>()[row * 3] = 10000.0f;
        splat.means() = means.cuda();
        AdamOptimizer optimizer(splat, AdamConfig{});
        optimizer.allocate_gradients();
        const int slots = static_cast<int>(sh_float4_slots_for_rest(splat.max_sh_coeffs_rest()));
        const std::vector<Tensor*> params = {&splat.means(), &splat.scaling_raw(),
                                             &splat.rotation_raw(), &splat.opacity_raw(),
                                             &splat.sh0(), &splat.shN()};
        std::vector<Tensor> before;
        for (size_t group = 0; group < types.size(); ++group) {
            const bool shn = types[group] == ParamType::ShN;
            const int bits = shn ? 8 : 16;
            // Gradients are lazy: allocate_gradients() only creates moment state.
            if (!fused)
                optimizer.get_grad(types[group]).fill_(0.0f);
            auto* state = optimizer.get_state_mutable(types[group]);
            ASSERT_NE(state, nullptr);
            ASSERT_EQ(state->joint_bits, bits);
            ASSERT_EQ(params[group]->dtype(), DataType::Float32);
            before.push_back(params[group]->cpu());
            auto packed = state->exp_avg.cpu();
            auto bounds = state->joint_bounds.cpu();
            const float mm[] = {-1.0f, 1.0f, 0.0f, 25.0f};
            std::memcpy(bounds.ptr<float>(), mm, sizeof(mm));
            const size_t attrs = shn ? static_cast<size_t>(slots * 4) : params[group]->numel() / rows;
            for (size_t row = 0; row < rows; ++row) {
                for (size_t attr = 0; attr < attrs; ++attr) {
                    const size_t cell = shn ? static_cast<size_t>(joint_sh_cell(row, attr / 4, attr % 4, slots))
                                            : row * attrs + attr;
                    // Opposite-sign history in two neighbours, exact-zero history elsewhere.
                    const float u = row == 0 ? -1.0f : row == 1 ? 1.0f
                                                                : 0.0f;
                    const float s = row < 2 ? 25.0f : 0.0f;
                    if (shn)
                        joint_adam::Codec8::encode_us(packed.ptr<uint8_t>(), cell, u, s, mm[0], mm[1], mm[2], mm[3]);
                    else
                        joint_adam::Codec16::encode_us(packed.ptr<uint8_t>(), cell, u, s, mm[0], mm[1], mm[2], mm[3]);
                }
            }
            ASSERT_EQ(cudaMemcpy(state->exp_avg.data_ptr(), packed.data_ptr(), packed.bytes(), cudaMemcpyHostToDevice), cudaSuccess);
            ASSERT_EQ(cudaMemcpy(state->joint_bounds.data_ptr(), bounds.data_ptr(), bounds.bytes(), cudaMemcpyHostToDevice), cudaSuccess);
        }
        Camera camera(Tensor::eye(3, Device::CUDA),
                      Tensor::from_vector(std::vector<float>{0, 0, 4}, {3}, Device::CUDA),
                      8.0f, 8.0f, 3.5f, 3.5f, {}, {}, CameraModelType::PINHOLE,
                      "joint_zero_history", "", {}, 8, 8, 0);
        auto background = Tensor::zeros({3}, Device::CUDA);
        for (int step = 0; step < steps; ++step) {
            const int iteration = 1001 + step;
            optimizer.zero_grad(iteration);
            if (fused) {
                auto forward = fast_rasterize_forward(camera, splat, background, 0, 0, 0, 0, false);
                ASSERT_TRUE(forward.has_value());
                fast_rasterize_backward(forward->second, Tensor::zeros_like(forward->first.image),
                                        splat, optimizer, {}, {}, DensificationType::None, iteration);
            } else {
                optimizer.step(iteration);
            }
        }
        ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
        for (size_t group = 0; group < types.size(); ++group) {
            SCOPED_TRACE(static_cast<int>(types[group]));
            const bool shn = types[group] == ParamType::ShN;
            const auto after = params[group]->cpu();
            const auto* state = optimizer.get_state(types[group]);
            const auto packed = state->exp_avg.cpu();
            const auto bounds = state->joint_bounds.cpu();
            const auto* mm = bounds.ptr<float>();
            const size_t attrs = shn ? static_cast<size_t>(slots * 4) : after.numel() / rows;
            double history_travel = 0.0;
            for (size_t row = 0; row < rows; ++row) {
                for (size_t attr = 0; attr < attrs; ++attr) {
                    const size_t cell = shn ? static_cast<size_t>(joint_sh_cell(row, attr / 4, attr % 4, slots))
                                            : row * attrs + attr;
                    if (row < 2) {
                        history_travel += std::abs(after.ptr<float>()[cell] - before[group].ptr<float>()[cell]);
                        continue;
                    }
                    ASSERT_EQ(after.ptr<float>()[cell], before[group].ptr<float>()[cell]) << "row=" << row << " attr=" << attr;
                    float m, v;
                    if (shn)
                        joint_adam::Codec8::decode_g1g2(packed.ptr<uint8_t>(), cell, mm[0], mm[1], mm[2], mm[3], m, v);
                    else
                        joint_adam::Codec16::decode_g1g2(packed.ptr<uint8_t>(), cell, mm[0], mm[1], mm[2], mm[3], m, v);
                    ASSERT_EQ(m, 0.0f);
                    ASSERT_EQ(v, 0.0f);
                }
            }
            EXPECT_GT(history_travel, 0.0) << "Zero gradient must still decay real momentum";
        }
    }
}
