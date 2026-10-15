/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/splat_data.hpp"
#include "core/tensor.hpp"
#include "io/exporter.hpp"
#include "io/formats/ply.hpp"
#include "training/rasterization/gsplat/SphericalBeta.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <cuda_runtime.h>
#include <filesystem>
#include <fstream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

namespace {

    namespace fs = std::filesystem;
    using namespace lfs::core;

    constexpr int kChannels = SplatData::SB_PARAMS_PER_LOBE;

    bool cuda_available() {
        int count = 0;
        return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
    }

    double softplus_ref(const double x) {
        constexpr double b = 6.931471805599453;
        const double bx = b * x;
        return bx > 20.0 ? x : std::log1p(std::exp(bx)) / b;
    }

    // Lobe contribution for one view direction, evaluated in double precision.
    void lobe_color_ref(const float* dir, const float* lobes, const int n_lobes, double out[3]) {
        const double n = std::sqrt(double{dir[0]} * dir[0] + double{dir[1]} * dir[1] + double{dir[2]} * dir[2]);
        const double v[3] = {dir[0] / n, dir[1] / n, dir[2] / n};
        out[0] = out[1] = out[2] = 0.0;
        for (int l = 0; l < n_lobes; ++l) {
            const float* q = lobes + l * kChannels;
            const double d[3] = {std::sin(double{q[3]}) * std::cos(double{q[4]}),
                                 std::sin(double{q[3]}) * std::sin(double{q[4]}),
                                 std::cos(double{q[3]})};
            const double dot = v[0] * d[0] + v[1] * d[1] + v[2] * d[2];
            if (dot <= 0.0)
                continue;
            const double term = std::pow(dot, 4.0 * std::exp(double{q[5]}));
            for (int c = 0; c < 3; ++c)
                out[c] += term * softplus_ref(q[c]);
        }
    }

    template <typename T>
    T* to_device(const std::vector<T>& host) {
        T* ptr = nullptr;
        EXPECT_EQ(cudaMalloc(&ptr, host.size() * sizeof(T)), cudaSuccess);
        EXPECT_EQ(cudaMemcpy(ptr, host.data(), host.size() * sizeof(T), cudaMemcpyHostToDevice), cudaSuccess);
        return ptr;
    }

    template <typename T>
    std::vector<T> to_host(const T* device, const size_t count) {
        std::vector<T> host(count);
        EXPECT_EQ(cudaMemcpy(host.data(), device, count * sizeof(T), cudaMemcpyDeviceToHost), cudaSuccess);
        return host;
    }

    struct KernelCase {
        int splats;
        int cameras;
        int lobes;
        int color_stride;
    };

    class SphericalBetaKernelTest : public ::testing::TestWithParam<KernelCase> {};

    // Forward matches a double-precision reference and backward matches central
    // differences of that reference, including multi-camera accumulation and the
    // RGB+depth color strides.
    TEST_P(SphericalBetaKernelTest, ForwardAndGradientMatchReference) {
        if (!cuda_available())
            GTEST_SKIP() << "CUDA device unavailable";
        const auto [N, C, L, stride] = GetParam();
        const size_t total = static_cast<size_t>(C) * N;

        std::mt19937 rng(1234u + static_cast<uint32_t>(N * 31 + C * 7 + L * 13 + stride));
        std::uniform_real_distribution<float> u(-1.0f, 1.0f);
        std::vector<float> dirs(total * 3), params(static_cast<size_t>(N) * L * kChannels);
        std::vector<float> base(total * stride), v_colors(total * stride);
        for (auto& x : dirs)
            x = 3.0f * u(rng);
        for (size_t i = 0; i < static_cast<size_t>(N) * L; ++i) {
            float* q = &params[i * kChannels];
            q[0] = 0.5f * u(rng);
            q[1] = 0.5f * u(rng);
            q[2] = 0.5f * u(rng);
            q[3] = (0.5f * u(rng) + 0.5f) * 3.14159f;
            q[4] = (0.5f * u(rng) + 0.5f) * 6.28318f;
            q[5] = 1.5f * u(rng);
        }
        for (auto& x : base)
            x = u(rng);
        for (auto& x : v_colors)
            x = u(rng);

        float* d_dirs = to_device(dirs);
        float* d_params = to_device(params);
        float* d_colors = to_device(base);
        float* d_v_colors = to_device(v_colors);
        float* d_v_params = to_device(std::vector<float>(params.size(), 0.0f));

        gsplat_lfs::launch_spherical_beta_fwd_kernel(
            L, stride, N, static_cast<int64_t>(total), d_dirs, d_params, nullptr, d_colors);
        gsplat_lfs::launch_spherical_beta_bwd_kernel(
            L, stride, N, static_cast<int64_t>(total), d_dirs, d_params, nullptr, d_v_colors, d_v_params);
        ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
        const auto colors = to_host(d_colors, base.size());
        const auto v_params = to_host(d_v_params, params.size());

        for (size_t e = 0; e < total; ++e) {
            double sb[3];
            lobe_color_ref(&dirs[e * 3], &params[(e % N) * L * kChannels], L, sb);
            for (int c = 0; c < 3; ++c)
                ASSERT_NEAR(colors[e * stride + c], base[e * stride + c] + sb[c], 1e-4) << "element " << e;
            for (int c = 3; c < stride; ++c)
                ASSERT_EQ(colors[e * stride + c], base[e * stride + c]) << "non-RGB channel modified";
        }

        const auto loss = [&](const std::vector<float>& p) {
            double sum = 0.0;
            for (size_t e = 0; e < total; ++e) {
                double sb[3];
                lobe_color_ref(&dirs[e * 3], &p[(e % N) * L * kChannels], L, sb);
                for (int c = 0; c < 3; ++c)
                    sum += v_colors[e * stride + c] * sb[c];
            }
            return sum;
        };
        auto probe = params;
        const size_t step = std::max<size_t>(1, params.size() / 300);
        for (size_t k = 0; k < params.size(); k += step) {
            constexpr float h = 1e-3f;
            const float original = probe[k];
            probe[k] = original + h;
            const double plus = loss(probe);
            probe[k] = original - h;
            const double minus = loss(probe);
            probe[k] = original;
            const double numeric = (plus - minus) / (2.0 * h);
            EXPECT_NEAR(v_params[k], numeric, 2e-2 * std::max(1e-2, std::abs(numeric)))
                << "parameter " << k << " (channel " << k % kChannels << ")";
        }

        cudaFree(d_dirs);
        cudaFree(d_params);
        cudaFree(d_colors);
        cudaFree(d_v_colors);
        cudaFree(d_v_params);
    }

    INSTANTIATE_TEST_SUITE_P(
        Lobes, SphericalBetaKernelTest,
        ::testing::Values(KernelCase{512, 1, 1, 3}, KernelCase{512, 1, 2, 3}, KernelCase{256, 1, 8, 3},
                          KernelCase{128, 2, 2, 4}, KernelCase{128, 3, 3, 8}));

    SplatData make_splat(const size_t n) {
        auto means = Tensor::zeros({n, 3}, Device::CPU);
        auto sh0 = Tensor::zeros({n, 1, 3}, Device::CPU);
        auto scaling = Tensor::full({n, 3}, -3.0f, Device::CPU);
        auto rotation = Tensor::zeros({n, 4}, Device::CPU);
        auto opacity = Tensor::zeros({n, 1}, Device::CPU);
        float* m = means.ptr<float>();
        float* r = rotation.ptr<float>();
        for (size_t i = 0; i < n; ++i) {
            m[i * 3] = static_cast<float>(i);
            r[i * 4] = 1.0f;
        }
        return SplatData(0, std::move(means).cuda(), std::move(sh0).cuda(), Tensor{},
                         std::move(scaling).cuda(), std::move(rotation).cuda(), std::move(opacity).cuda(), 1.0f);
    }

    // Distinct value per (splat, lobe, channel) so any layout mix-up is visible.
    Tensor distinct_lobes(const size_t n, const int lobes) {
        const size_t cols = static_cast<size_t>(lobes) * kChannels;
        auto host = Tensor::empty({n, cols}, Device::CPU);
        float* p = host.ptr<float>();
        for (size_t i = 0; i < n; ++i)
            for (size_t j = 0; j < cols; ++j)
                p[i * cols + j] = static_cast<float>(i) * 100.0f + static_cast<float>(j) * 0.5f;
        return host;
    }

    std::vector<float> host_values(const Tensor& t) {
        auto cpu = t.cpu().contiguous();
        return {cpu.ptr<float>(), cpu.ptr<float>() + cpu.numel()};
    }

    fs::path temp_path(const std::string& name) {
        const auto dir = fs::temp_directory_path() / "lfs_spherical_beta_tests";
        fs::create_directories(dir);
        return dir / name;
    }

    TEST(SphericalBetaSplatData, InitSpreadsLobeDirectionsAndZeroesColorAndSharpness) {
        if (!cuda_available())
            GTEST_SKIP() << "CUDA device unavailable";
        auto splat = make_splat(64);
        EXPECT_FALSE(splat.has_spherical_beta());
        splat.init_spherical_beta(2);
        ASSERT_TRUE(splat.has_spherical_beta());
        EXPECT_EQ(splat.get_sb_lobes(), 2);
        const auto values = host_values(splat.sb_params());
        ASSERT_EQ(values.size(), 64u * 2u * kChannels);
        for (size_t i = 0; i < values.size(); i += kChannels) {
            EXPECT_EQ(values[i + 0], 0.0f);
            EXPECT_EQ(values[i + 1], 0.0f);
            EXPECT_EQ(values[i + 2], 0.0f);
            EXPECT_GE(values[i + 3], 0.0f);
            EXPECT_LE(values[i + 3], 3.1416f);
            EXPECT_GE(values[i + 4], 0.0f);
            EXPECT_LE(values[i + 4], 6.2832f);
            EXPECT_EQ(values[i + 5], 0.0f);
        }
    }

    // PLY stores lobes channel-major (sb_params_{channel * lobes + lobe}), the layout
    // Beta Splatting and SuperSplat read; LichtFeld keeps them lobe-major in memory.
    TEST(SphericalBetaPly, ExportIsChannelMajorAndRoundTrips) {
        if (!cuda_available())
            GTEST_SKIP() << "CUDA device unavailable";
        constexpr size_t n = 5;
        constexpr int lobes = 2;
        auto splat = make_splat(n);
        splat.set_sb_params(distinct_lobes(n, lobes), lobes);
        const auto expected = host_values(splat.sb_params());

        const auto path = temp_path("roundtrip.ply");
        const auto saved = lfs::io::save_ply(splat, {.output_path = path, .binary = true});
        ASSERT_TRUE(saved.has_value()) << saved.error().format();

        std::ifstream file(path, std::ios::binary);
        std::string header;
        for (std::string line; std::getline(file, line) && line.rfind("end_header", 0) != 0;)
            header += line + "\n";
        for (int i = 0; i < lobes * kChannels; ++i)
            EXPECT_NE(header.find("property float sb_params_" + std::to_string(i) + "\n"), std::string::npos)
                << "missing sb_params_" << i;

        auto loaded = lfs::io::load_ply(path);
        ASSERT_TRUE(loaded.has_value());
        ASSERT_TRUE(loaded->value.has_spherical_beta());
        EXPECT_EQ(loaded->value.get_sb_lobes(), lobes);
        EXPECT_EQ(host_values(loaded->value.sb_params()), expected);
    }

    TEST(SphericalBetaPly, IncompleteColumnsAreIgnored) {
        if (!cuda_available())
            GTEST_SKIP() << "CUDA device unavailable";
        constexpr size_t n = 3;
        auto splat = make_splat(n);
        lfs::io::PlyAttributeBlock partial;
        partial.values = Tensor::ones({n, 5}, Device::CPU);
        for (int i = 0; i < 5; ++i)
            partial.names.push_back("sb_params_" + std::to_string(i));
        const auto path = temp_path("partial.ply");
        const auto saved = lfs::io::save_ply(
            splat, {.output_path = path, .binary = true, .extra_attributes = {partial}});
        ASSERT_TRUE(saved.has_value()) << saved.error().format();

        auto loaded = lfs::io::load_ply(path);
        ASSERT_TRUE(loaded.has_value());
        EXPECT_FALSE(loaded->value.has_spherical_beta());
        EXPECT_FALSE(loaded->warnings.empty());
    }

    TEST(SphericalBetaSerialization, LobesRoundTripAndPlainModelsKeepVersion4) {
        if (!cuda_available())
            GTEST_SKIP() << "CUDA device unavailable";
        constexpr size_t n = 4;
        const auto version_of = [](const std::string& bytes) {
            uint32_t version = 0;
            std::memcpy(&version, bytes.data() + sizeof(uint32_t), sizeof(version));
            return version;
        };

        auto plain = make_splat(n);
        std::stringstream plain_stream;
        plain.serialize(plain_stream);
        EXPECT_EQ(version_of(plain_stream.str()), 4u);

        auto splat = make_splat(n);
        splat.set_sb_params(distinct_lobes(n, 3), 3);
        std::stringstream stream;
        splat.serialize(stream);
        EXPECT_EQ(version_of(stream.str()), 5u);

        std::stringstream skip_stream(stream.str());
        SplatData::skip_serialized(skip_stream);
        EXPECT_EQ(static_cast<size_t>(skip_stream.tellg()), stream.str().size());

        SplatData restored;
        restored.deserialize(stream);
        ASSERT_TRUE(restored.has_spherical_beta());
        EXPECT_EQ(restored.get_sb_lobes(), 3);
        EXPECT_EQ(host_values(restored.sb_params()), host_values(splat.sb_params()));
    }

} // namespace
