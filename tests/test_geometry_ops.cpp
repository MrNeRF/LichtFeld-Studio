/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/parameters.hpp"
#include "core/tensor_cuda_interop.hpp"
#include "cuda_backend_test.hpp"
#include "lfs/training/ops/registry.hpp"
#include "training/kernels/depth_loss.hpp"
#include "training/kernels/normal_consistency_loss.hpp"
#include "training/kernels/normal_loss.hpp"

#include <algorithm>
#include <cstring>
#include <gtest/gtest.h>
#include <tuple>

namespace {
    using namespace lfs::core;
    namespace k = lfs::training::kernels;
    namespace geo = lfs::gpu_ops;
    using GeometryOps = lfs::test::CudaBackendTest;
    constexpr size_t H = 40, W = 40, N = H * W;

    struct Inputs {
        Tensor depth, alpha, normal, target, weight;
        Inputs() {
            std::vector<float> d(N), a(N, 0.875f), n(3 * N), t(N), w(N);
            for (size_t y = 0; y < H; ++y) {
                for (size_t x = 0; x < W; ++x) {
                    const size_t i = y * W + x;
                    d[i] = a[i] * (2.f + static_cast<float>(x) / 1024.f);
                    t[i] = 0.2f + static_cast<float>((i * 7919u + 42u) % 2003u) / 4000.f;
                    n[i] = 0.125f;
                    n[N + i] = 0.25f;
                    n[2 * N + i] = -0.875f;
                    // Disjoint stencils make the accumulated gradients byte reproducible.
                    w[i] = x % 3 == 1 && y % 3 == 1 ? 1.f : 0.f;
                }
            }
            depth = Tensor::from_vector(d, {H, W}, Device::GPU);
            alpha = Tensor::from_vector(a, {H, W}, Device::GPU);
            normal = Tensor::from_vector(n, {3, H, W}, Device::GPU);
            target = Tensor::from_vector(t, {H, W}, Device::GPU);
            weight = Tensor::from_vector(w, {H, W}, Device::GPU);
        }
    };

    std::vector<Tensor> evaluate(const Inputs& in, int mode, bool table) {
        auto gd = Tensor::zeros({H, W}, Device::GPU);
        auto ga = Tensor::zeros({H, W}, Device::GPU);
        auto gn = Tensor::zeros({3, H, W}, Device::GPU);
        auto loss = Tensor::zeros({1}, Device::GPU);
        auto partials = Tensor::zeros({mode == 0 ? k::depth_loss_partial_count(N) : k::normal_loss_partial_count(N)}, Device::GPU);
        const auto& g = *lfs::training::training_ops(GpuBackend::CUDA).geometry;
        const geo::Intrinsics intr{40, 40, 20, 20};
        k::DepthAnchor anchor;
        anchor.valid = true;
        anchor.scale = 1.f;
        anchor.shift = 0.1f;
        anchor.floor = 0.01f;
        const auto stream = getCurrentCUDAStream();
        if (mode == 0) {
            if (table)
                g.depth(in.depth, in.alpha, in.target, in.weight, gd, ga, loss, partials, {0.5f, 0.25f, 0.f, &anchor});
            else
                k::launch_depth_loss(in.depth.ptr<float>(), in.alpha.ptr<float>(), in.target.ptr<float>(), gd.ptr<float>(), ga.ptr<float>(), loss.ptr<float>(), partials.ptr<float>(), W, H, 0.5f, 0.25f, 0.f, &anchor, stream, in.weight.ptr<float>());
        } else if (mode == 1) {
            auto target = in.normal + 0.25f;
            if (table)
                g.normal(in.normal, in.alpha, target, in.weight, gn, loss, partials, 0.5f);
            else
                k::launch_normal_loss(in.normal.ptr<float>(), in.alpha.ptr<float>(), target.ptr<float>(), gn.ptr<float>(), loss.ptr<float>(), partials.ptr<float>(), W, H, 0.5f, stream, in.weight.ptr<float>());
        } else if (mode == 2) {
            if (table)
                g.consistency(in.normal, in.depth, in.alpha, in.weight, gn, gd, ga, loss, partials, intr, 0.5f);
            else
                k::launch_normal_consistency_loss(in.normal.ptr<float>(), in.depth.ptr<float>(), in.alpha.ptr<float>(), gn.ptr<float>(), gd.ptr<float>(), ga.ptr<float>(), loss.ptr<float>(), partials.ptr<float>(), W, H, intr.fx, intr.fy, intr.cx, intr.cy, 0.5f, stream, in.weight.ptr<float>());
        } else {
            if (table)
                g.prior_depth(in.normal, in.depth, in.alpha, in.weight, gd, ga, loss, partials, intr, 0.5f);
            else
                k::launch_normal_prior_depth_loss(in.normal.ptr<float>(), in.depth.ptr<float>(), in.alpha.ptr<float>(), gd.ptr<float>(), ga.ptr<float>(), loss.ptr<float>(), partials.ptr<float>(), W, H, intr.fx, intr.fy, intr.cx, intr.cy, 0.5f, stream, in.weight.ptr<float>());
        }
        return {gd, ga, gn, loss, partials};
    }

    void compare(int mode) {
        Inputs in;
        const auto direct = evaluate(in, mode, false);
        const auto dispatched = evaluate(in, mode, true);
        ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
        for (size_t i = 0; i < direct.size(); ++i) {
            const auto a = direct[i].cpu(), b = dispatched[i].cpu();
            ASSERT_EQ(a.bytes(), b.bytes());
            EXPECT_EQ(std::memcmp(a.data_ptr(), b.data_ptr(), a.bytes()), 0) << i;
        }
        EXPECT_EQ(dispatched.back().cpu().ptr<float>()[0], 1.f);
        EXPECT_GT(dispatched[3].cpu().ptr<float>()[0], 0.f);
    }
} // namespace

TEST_F(GeometryOps, DepthMatchesLauncher) { compare(0); }
TEST_F(GeometryOps, NormalMatchesLauncher) { compare(1); }
TEST_F(GeometryOps, ConsistencyMatchesLauncher) { compare(2); }
TEST_F(GeometryOps, PriorDepthMatchesLauncher) { compare(3); }

TEST_F(GeometryOps, AnchorCollectionMatchesLauncher) {
    Inputs in;
    std::vector<float> xyz(1024 * 3);
    for (size_t i = 0; i < 1024; ++i) {
        xyz[3 * i] = (static_cast<float>(i % 32) - 16.f) / 40.f;
        xyz[3 * i + 1] = (static_cast<float>(i / 32) - 16.f) / 40.f;
        xyz[3 * i + 2] = 2.f;
    }
    auto points = Tensor::from_vector(xyz, {1024, 3}, Device::GPU);
    auto view = Tensor::from_vector(std::vector<float>{1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1}, {4, 4}, Device::GPU);
    const geo::AnchorParams params{{40, 40, 20, 20}, 0.01f, {-10, -10, -10}, {10, 10, 10}};
    auto expected = k::collect_depth_anchor_samples(points.ptr<float>(), 1024, view.ptr<float>(), 40, 40, 20, 20, in.target.ptr<float>(), W, H, 0.01f, params.aabb_lo.data(), params.aabb_hi.data(), getCurrentCUDAStream());
    auto actual = lfs::training::training_ops(GpuBackend::CUDA).geometry->collect_anchor_samples(points, view, in.target, params);
    ASSERT_EQ(actual.size(), 1024);
    ASSERT_EQ(actual.size(), expected.size());
    // Collection uses atomic append; sample order is not part of its contract.
    const auto less = [](const auto& a, const auto& b) { return std::tie(a.x, a.y) < std::tie(b.x, b.y); };
    std::sort(actual.begin(), actual.end(), less);
    std::sort(expected.begin(), expected.end(), less);
    EXPECT_EQ(std::memcmp(actual.data(), expected.data(), actual.size() * sizeof(geo::AnchorSample)), 0);
}

TEST_F(GeometryOps, RequiredOnlyForSupervisionAndUnavailableOnOtherBackends) {
    lfs::core::param::TrainingParameters p;
    const auto bit = static_cast<size_t>(lfs::training::Family::Geometry);
    EXPECT_FALSE(lfs::training::required_training_families(p, {}).test(bit));
    for (const bool depth : {false, true}) {
        p.optimization.use_depth_loss = depth;
        p.optimization.use_normal_loss = !depth;
        EXPECT_TRUE(lfs::training::required_training_families(p, {}).test(bit));
        for (const auto backend : {GpuBackend::Vulkan, GpuBackend::Metal}) {
            const auto& table = lfs::training::training_ops(backend);
            EXPECT_EQ(table.geometry, nullptr);
            lfs::training::FamilySet required;
            required.set(bit);
            EXPECT_EQ(lfs::training::missing_training_families(table, required), std::vector<std::string_view>{"Geometry"});
        }
    }
}
