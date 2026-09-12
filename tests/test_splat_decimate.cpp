/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "../src/io/cuda/splat_decimate_internal.hpp"
#include "core/tensor_backend.hpp"
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <io/splat_decimate.hpp>
#include <iostream>
#include <random>

namespace {
    using namespace lfs::core;
    using namespace lfs::io;
    using namespace lfs::io::decimate;

    Data make_decimate_data(size_t n, int degree, uint32_t seed = 42) {
        auto d = allocate(n, (degree + 1) * (degree + 1) - 1, Device::CPU);
        auto v = d.view();
        std::mt19937 rng(seed);
        std::uniform_real_distribution<float> uniform(-1, 1);
        for (size_t i = 0; i < n; ++i) {
            for (int a = 0; a < 3; ++a) {
                v.pos[i * 3 + a] = uniform(rng) * 10;
                v.scale[i * 3 + a] = -3 + uniform(rng);
                v.dc[i * 3 + a] = uniform(rng);
            }
            double norm = 0;
            for (int a = 0; a < 4; ++a) {
                v.rot[i * 4 + a] = uniform(rng);
                norm += double(v.rot[i * 4 + a]) * v.rot[i * 4 + a];
            }
            for (int a = 0; a < 4; ++a)
                v.rot[i * 4 + a] /= std::sqrt(norm);
            v.opacity[i] = uniform(rng) * 2;
            for (int a = 0; a < d.rest * 3; ++a)
                v.sh[i * d.rest * 3 + a] = uniform(rng) * 0.2f;
        }
        return d;
    }
    Data transfer(const Data& d, Device dev) {
        return {d.pos.to(dev), d.rot.to(dev), d.scale.to(dev), d.opacity.to(dev), d.dc.to(dev), d.sh.to(dev), d.n, d.rest};
    }
    SplatData splats(const Data& d, int degree) {
        auto c = transfer(d, Device::CUDA);
        return SplatData(degree, c.pos, c.dc, c.sh, c.scale, c.rot, c.opacity, 2.5f, SplatData::ShNLayout::Canonical);
    }
    void finite(const SplatData& d) {
        for (auto* tensor : {&d.means(), &d.scaling_raw(), &d.rotation_raw(), &d.opacity_raw(), &d.sh0()}) {
            auto host = tensor->cpu();
            for (size_t i = 0; i < host.numel(); ++i)
                ASSERT_TRUE(std::isfinite(host.ptr<float>()[i]));
            EXPECT_EQ(tensor->device(), Device::CUDA);
        }
        auto sh = d.shN_canonical().cpu();
        for (size_t i = 0; i < sh.numel(); ++i)
            ASSERT_TRUE(std::isfinite(sh.ptr<float>()[i]));
        auto q = d.rotation_raw().cpu(), op = d.opacity_raw().cpu();
        for (size_t i = 0; i < d.size(); ++i) {
            double norm = 0;
            for (int a = 0; a < 4; ++a)
                norm += double(q.ptr<float>()[i * 4 + a]) * q.ptr<float>()[i * 4 + a];
            ASSERT_NEAR(norm, 1, 1e-5);
            double alpha = 1 / (1 + std::exp(-double(op.ptr<float>()[i])));
            ASSERT_GT(alpha, 0);
            ASSERT_LT(alpha, 1);
        }
    }
} // namespace

TEST(SplatDecimate, ExactTargetCount) {
    auto input = splats(make_decimate_data(20000, 1), 1);
    DecimateOptions options;
    options.target_count = 7000;
    auto out = decimate_splats(input, options);
    ASSERT_TRUE(out) << out.error().message;
    EXPECT_EQ(out->size(), 7000);
    finite(*out);
    EXPECT_EQ(out->get_scene_scale(), 2.5f);
    EXPECT_EQ(out->get_max_sh_degree(), 1);
    for (size_t target : {20000u, 25000u}) {
        options.target_count = target;
        auto same = decimate_splats(input, options);
        ASSERT_TRUE(same) << same.error().message;
        EXPECT_EQ(same->size(), input.size());
        std::array<const Tensor*, 6> a{&input.means(), &input.rotation_raw(), &input.scaling_raw(), &input.opacity_raw(), &input.sh0(), &input.shN()};
        std::array<const Tensor*, 6> b{&same->means(), &same->rotation_raw(), &same->scaling_raw(), &same->opacity_raw(), &same->sh0(), &same->shN()};
        for (size_t j = 0; j < a.size(); ++j) {
            auto x = a[j]->cpu(), y = b[j]->cpu();
            ASSERT_EQ(x.numel(), y.numel());
            EXPECT_EQ(std::memcmp(x.ptr<float>(), y.ptr<float>(), x.numel() * 4), 0);
        }
        EXPECT_NE(same->means().ptr<float>(), input.means().ptr<float>());
    }
}

TEST(SplatDecimate, VulkanInputMaterializesCudaStorageForGpuKernels) {
    if (!gpu_backend_available(GpuBackend::Vulkan))
        GTEST_SKIP() << "Vulkan backend unavailable";
    const GpuBackendScope vulkan_scope(GpuBackend::Vulkan);
    auto input = splats(make_decimate_data(256, 1), 1);
    ASSERT_EQ(gpu_backend_of(input.means()), GpuBackend::Vulkan);
    for (const bool use_gpu : {false, true}) {
        for (const size_t target : {size_t{128}, size_t{256}}) {
            DecimateOptions options;
            options.target_count = target;
            options.use_gpu = use_gpu;
            auto result = decimate_splats(input, options);
            ASSERT_TRUE(result) << result.error().message;
            EXPECT_EQ(result->size(), target);
            EXPECT_EQ(gpu_backend_of(result->means()), GpuBackend::CUDA);
            finite(*result);
        }
    }
    EXPECT_EQ(gpu_backend_of(input.means()), GpuBackend::Vulkan);
    EXPECT_EQ(input.size(), 256);
}

TEST(SplatDecimate, GpuMatchesCpuReference) {
    for (int degree : {0, 2})
        for (size_t target : {2500u, 700u}) {
            auto host = make_decimate_data(5000, degree);
            auto device = transfer(host, Device::CUDA);
            auto cpu = cpu_candidates(host), gpu = gpu_candidates(device);
            size_t equal = 0;
            double max_error = 0;
            for (size_t i = 0; i < host.n; ++i) {
                bool same = true;
                for (int c = 0; c < 4; ++c) {
                    size_t e = i * 4 + c;
                    same &= cpu.idx[e] == gpu.idx[e];
                    max_error = std::max(max_error, std::abs(double(cpu.cost[e]) - gpu.cost[e]));
                }
                equal += same;
            }
            std::cout << "SH" << degree << " target " << target << " candidate parity " << 100.0 * equal / host.n << "%, max cost error " << max_error << '\n';
            EXPECT_GE(equal, host.n * 99 / 100);
            auto input = splats(host, degree);
            DecimateOptions o;
            o.target_count = target;
            auto a = decimate_splats(input, o);
            o.use_gpu = false;
            auto b = decimate_splats(input, o);
            ASSERT_TRUE(a) << a.error().message;
            ASSERT_TRUE(b) << b.error().message;
            EXPECT_EQ(a->size(), target);
            EXPECT_EQ(b->size(), target);
            auto x = a->means().cpu(), y = b->means().cpu();
            size_t matches = 0;
            for (size_t i = 0; i < target; ++i) {
                bool same = true;
                for (int c = 0; c < 3; ++c)
                    same &= std::abs(x.ptr<float>()[i * 3 + c] - y.ptr<float>()[i * 3 + c]) <= 1e-4;
                matches += same;
            }
            std::cout << "Final ordered mean parity " << 100.0 * matches / target << "%\n";
            EXPECT_GE(matches, target * 99 / 100);
        }
}

TEST(SplatDecimate, BucketSelectionChainCaps) {
    // Stable 1024-bucket reference walk: pair 0/1, cap-3 joins 2,
    // then cap-4 joins 3. Rows 4/5 remain survivors at this target.
    Candidates c{{1, 0, 0, 0, 0, 0}, {0, 0, 1, 2, 3, 4}};
    auto s = select(c, 6, 1, 3);
    EXPECT_EQ(s.removed, 3);
    EXPECT_EQ(s.member_group, (std::vector<int>{0, 0, 0, 0, -1, -1}));
    EXPECT_EQ(s.members, (std::vector<uint32_t>{0, 1, 2, 3}));
    EXPECT_EQ(s.offsets, (std::vector<uint32_t>{0, 4}));
    EXPECT_EQ(s.minimum, (std::vector<uint32_t>{0}));
}

TEST(SplatDecimate, MergeMathMatchesReference) {
    // Reference values: splat-transform 3.4.0 moment-match.ts.
    const float positions[3][6] = {{0, 0, 0, 1, 2, 3}, {-2, .3f, 4, .5f, -1, 2}, {0, 0, 0, 0, 0, 0}};
    const float geo[3][16] = {
        {1, 0, 0, 0, -1, -2, -3, 0, 1, 0, 0, 0, -2, -1, -3, 1},
        {.8f, .2f, -.3f, .4f, -.5f, -1.2f, -2, -.7f, .1f, .7f, .2f, -.1f, -1.5f, -.3f, -.9f, .2f},
        {1, 0, 0, 0, -2, -3, -4, -2, 1, 0, 0, 0, -2, -3, -4, -2}};
    const float colors[3][6] = {{1, 0, 0, 0, 1, 0}, {.3f, -.2f, .8f, -1, .4f, .2f}, {-.1f, .7f, .2f, .9f, .3f, -.2f}};
    // position, quaternion wxyz, log scales, logit opacity, DC
    const double expected[3][14] = {
        {.5938454849513094, 1.1876909699026188, 1.7815364548539283, -.09408512216425112, .7912829578609097, .39752234175413925, .45496731552116215, .6131425608762756, -1.3098395018387212, -1.4834543008304777, -2.2207945100648603, .40615451504869066, .5938454849513094, 0},
        {-.13655732359448242, -.6689901886955273, 2.5092458588755857, .490366602662896, .7861048214497864, -.048822602568460716, -.3730900134882883, .457810740619779, -.6025265854452095, -.9196929419743799, -1.15663017023546, -.6689901886955273, .24722624602128235, .3527737629194144},
        {0, 0, 0, 1, 0, 0, 0, -1.9999997270093244, -2.9999979828601013, -3.9999850954322134, -1.1614457238809361, .399999987334013, .5, 0}};
    const double expected_cost[3] = {4.76947546005249, 3.3579142093658447, 0.9450742602348328};
    Selection s;
    s.member_group = {0, 0};
    s.members = {0, 1};
    s.offsets = {0, 2};
    s.minimum = {0};
    s.removed = 1;
    for (int pair = 0; pair < 3; ++pair) {
        auto d = allocate(2, 0, Device::CPU);
        auto v = d.view();
        std::copy_n(positions[pair], 6, v.pos);
        std::copy_n(colors[pair], 6, v.dc);
        for (int i = 0; i < 2; ++i) {
            std::copy_n(geo[pair] + i * 8, 4, v.rot + i * 4);
            std::copy_n(geo[pair] + i * 8 + 4, 3, v.scale + i * 3);
            v.opacity[i] = geo[pair][i * 8 + 7];
        }
        for (bool gpu : {false, true}) {
            auto candidates = gpu ? gpu_candidates(transfer(d, Device::CUDA)) : cpu_candidates(d);
            EXPECT_EQ(candidates.idx[0], 1);
            EXPECT_EQ(candidates.idx[4], 0);
            EXPECT_NEAR(candidates.cost[0], expected_cost[pair], 1e-5);
            EXPECT_NEAR(candidates.cost[4], expected_cost[pair], 1e-5);
            for (size_t e : {1u, 2u, 3u, 5u, 6u, 7u}) {
                EXPECT_EQ(candidates.idx[e], invalid);
                EXPECT_EQ(candidates.cost[e], INFINITY);
            }
            auto merged = gpu ? gpu_merge(transfer(d, Device::CUDA), s) : cpu_merge(d, s);
            auto h = transfer(merged, Device::CPU);
            auto r = h.view();
            for (int c = 0; c < 3; ++c) {
                EXPECT_NEAR(r.pos[c], expected[pair][c], 1e-5);
                EXPECT_NEAR(r.scale[c], expected[pair][7 + c], 1e-5);
                EXPECT_NEAR(r.dc[c], expected[pair][11 + c], 1e-5);
            }
            double dot = 0;
            for (int c = 0; c < 4; ++c)
                dot += r.rot[c] * expected[pair][3 + c];
            for (int c = 0; c < 4; ++c)
                EXPECT_NEAR(r.rot[c] * (dot < 0 ? -1 : 1), expected[pair][3 + c], 1e-5);
            EXPECT_NEAR(r.opacity[0], expected[pair][10], 1e-5);
        }
    }
}

TEST(SplatDecimate, DeletedCancellationAndSmallInputs) {
    auto input = splats(make_decimate_data(17, 2), 2);
    auto deleted = Tensor::zeros({17}, Device::CPU, DataType::Bool);
    deleted.ptr<bool>()[3] = true;
    deleted.ptr<bool>()[7] = true;
    input.deleted() = deleted.to(Device::CUDA);
    DecimateOptions o;
    o.target_count = 17;
    auto visible = decimate_splats(input, o);
    ASSERT_TRUE(visible) << visible.error().message;
    EXPECT_EQ(visible->size(), 15);
    EXPECT_EQ(input.size(), 17);
    o.target_count = 1;
    auto one = decimate_splats(input, o);
    ASSERT_TRUE(one) << one.error().message;
    EXPECT_EQ(one->size(), 1);
    finite(*one);
    o.progress = [](float, const std::string&) { return false; };
    auto cancelled = decimate_splats(input, o);
    ASSERT_FALSE(cancelled);
    EXPECT_EQ(cancelled.error().code, lfs::io::ErrorCode::CANCELLED);
    o.progress = {};
    input.deleted() = Tensor::ones({17}, Device::CUDA, DataType::Bool);
    auto empty = decimate_splats(input, o);
    ASSERT_TRUE(empty) << empty.error().message;
    EXPECT_EQ(empty->size(), 0);
    o.target_count = 0;
    EXPECT_FALSE(decimate_splats(input, o));
}

TEST(SplatDecimate, DecimatePerfSmoke) {
    if (!std::getenv("LFS_DECIMATE_PERF") || std::string(std::getenv("LFS_DECIMATE_PERF")) != "1")
        GTEST_SKIP() << "Set LFS_DECIMATE_PERF=1 for the nightly 1M SH3 benchmark";
    const size_t count = std::getenv("LFS_DECIMATE_PERF_COUNT") ? std::stoull(std::getenv("LFS_DECIMATE_PERF_COUNT")) : 1000000;
    ASSERT_GE(count, 2);
    ASSERT_LE(count, 10000000);
    auto input = splats(make_decimate_data(count, 3), 3);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    DecimateOptions o;
    o.target_count = count / 2;
    auto start = std::chrono::steady_clock::now();
    o.progress = [&](float progress, const std::string& stage) { std::cout<<stage<<" ("<<progress<<") at "<<std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count()<<" ms\n";return true; };
    auto out = decimate_splats(input, o);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
    ASSERT_TRUE(out) << out.error().message;
    EXPECT_EQ(out->size(), count / 2);
    std::cout << "Decimate " << count << " SH3 -> " << count / 2 << " total: " << ms << " ms\n";
    EXPECT_LT(ms, 3000.0 * count / 1000000);
}

TEST(SplatDecimate, ExactNeighboursAcrossDistributionsAndScales) {
    for (int distribution = 0; distribution < 5; ++distribution) {
        for (const float scale : {1.0f, 1e-23f, 1e20f}) {
            auto host = make_decimate_data(distribution == 4 ? 8193 : 257, 0);
            auto v = host.view();
            for (size_t i = 0; i < host.n; ++i) {
                if (distribution == 0) {
                    v.pos[i * 3] = float(i % 7);
                    v.pos[i * 3 + 1] = v.pos[i * 3 + 2] = 0;
                } else if (distribution == 1) {
                    v.pos[i * 3] *= 100000;
                    v.pos[i * 3 + 1] *= 0.00001f;
                } else if (distribution == 2) {
                    v.pos[i * 3 + 2] = 0;
                } else if (distribution == 3) {
                    for (int a = 0; a < 3; ++a)
                        v.pos[i * 3 + a] += (i % 2 ? 10000 : -10000);
                } else {
                    // Fix bounds and probe both sides of exact grid faces.
                    for (int a = 0; a < 3; ++a) {
                        if (i < 2)
                            v.pos[i * 3 + a] = i == 0 ? -10 : 10;
                        else if (i < 130) {
                            float face = -10 + ((i + a) % 9) * 2.5f;
                            v.pos[i * 3 + a] = std::nextafter(face, i % 2 ? -INFINITY : INFINITY);
                        }
                    }
                }
            }
            for (size_t i = 0; i < host.n * 3; ++i)
                v.pos[i] *= scale;
            auto cpu = cpu_candidates(host);
            auto gpu = gpu_candidates(transfer(host, Device::CUDA));
            EXPECT_EQ(cpu.idx, gpu.idx) << "distribution=" << distribution << " scale=" << scale;
            EXPECT_EQ(cpu.cost, gpu.cost) << "distribution=" << distribution << " scale=" << scale;
        }
    }
}
