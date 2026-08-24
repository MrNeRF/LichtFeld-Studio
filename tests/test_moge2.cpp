/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/alloc_counter.hpp"
#include "core/nn.hpp"

#include <array>
#include <cuda_runtime.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <format>
#include <fstream>
#include <gtest/gtest.h>
#include <iostream>
#include <iterator>
#include <string>
#include <vector>

namespace {

    constexpr float kF16Rtol = 2e-2f;
    constexpr float kF16Atol = 2e-2f;
    constexpr float kNormalLinf = 1e-2f;

    std::string project_root() {
        return PROJECT_ROOT_PATH;
    }

    std::vector<float> host_f32(const lfs::core::Tensor& t) {
        return t.to(lfs::core::DataType::Float32).to(lfs::core::Device::CPU).contiguous().to_vector();
    }

    lfs::core::Tensor make_test_image(int height, int width, lfs::core::DataType dtype) {
        std::vector<float> chw(static_cast<std::size_t>(3) * height * width);
        const std::size_t plane = static_cast<std::size_t>(height) * width;
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                const std::size_t i = static_cast<std::size_t>(y) * width + x;
                chw[i] = static_cast<float>(x) / static_cast<float>(std::max(width - 1, 1));
                chw[plane + i] = static_cast<float>(y) / static_cast<float>(std::max(height - 1, 1));
                chw[2 * plane + i] =
                    static_cast<float>(x + y) / static_cast<float>(std::max(width + height - 2, 1));
            }
        }
        auto t = lfs::core::Tensor::from_vector(
            chw,
            lfs::core::TensorShape(std::vector<std::size_t>{
                1, 3, static_cast<std::size_t>(height), static_cast<std::size_t>(width)}),
            lfs::core::Device::CUDA);
        if (dtype != lfs::core::DataType::Float32) {
            return t.to(dtype);
        }
        return t;
    }

    float max_abs_at(const std::vector<float>& got, const nlohmann::json& node) {
        const auto& idx = node.at("indices");
        const auto& vals = node.at("values");
        float worst = 0.0f;
        for (std::size_t i = 0; i < idx.size(); ++i) {
            const auto index = idx[i].get<std::size_t>();
            if (index >= got.size()) {
                return 1e30f;
            }
            worst = std::max(worst, std::abs(got[index] - vals[i].get<float>()));
        }
        return worst;
    }

    float max_rel_at(const std::vector<float>& got, const nlohmann::json& node) {
        const auto& idx = node.at("indices");
        const auto& vals = node.at("values");
        float worst = 0.0f;
        for (std::size_t i = 0; i < idx.size(); ++i) {
            const auto index = idx[i].get<std::size_t>();
            if (index >= got.size()) {
                return 1e30f;
            }
            const float ref = vals[i].get<float>();
            const float diff = std::abs(got[index] - ref);
            const float denom = std::max(std::abs(ref), 1e-6f);
            worst = std::max(worst, diff / denom);
        }
        return worst;
    }

    bool close_sampled(const std::vector<float>& got, const nlohmann::json& node, float rtol,
                       float atol) {
        const auto& idx = node.at("indices");
        const auto& vals = node.at("values");
        int mismatches = 0;
        for (std::size_t i = 0; i < idx.size(); ++i) {
            const auto index = idx[i].get<std::size_t>();
            if (index >= got.size()) {
                ADD_FAILURE() << "index " << index << " out of range " << got.size();
                return false;
            }
            const float ref = vals[i].get<float>();
            const float diff = std::abs(got[index] - ref);
            const float tol = atol + rtol * std::abs(ref);
            if (diff > tol) {
                if (mismatches < 8) {
                    ADD_FAILURE() << "idx " << index << " got " << got[index] << " expected " << ref
                                  << " diff " << diff;
                }
                ++mismatches;
            }
        }
        if (mismatches > 0) {
            ADD_FAILURE() << mismatches << " sampled mismatches";
            return false;
        }
        return true;
    }

} // namespace

TEST(Moge2Test, TokenGridMatchesOnnxFormula) {
    int th = 0;
    int tw = 0;
    lfs::core::nn::models::Moge2::token_grid(70, 70, 400, th, tw);
    EXPECT_EQ(th, 20);
    EXPECT_EQ(tw, 20);
    lfs::core::nn::models::Moge2::token_grid(518, 518, 1800, th, tw);
    EXPECT_EQ(th, 42);
    EXPECT_EQ(tw, 42);
}

TEST(Moge2Test, CommittedFixtureIsSmall) {
    const std::string path = project_root() + "/tests/data/nn/moge2_ref_fixture.json";
    std::ifstream in(path);
    ASSERT_TRUE(static_cast<bool>(in));
    std::string body((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    EXPECT_LT(body.size(), 2u * 1024u * 1024u);
    auto payload = nlohmann::json::parse(body);
    EXPECT_TRUE(payload.contains("nodes"));
    EXPECT_TRUE(payload["full"].contains("normal"));
}

TEST(Moge2Test, FullModelParityIsOptIn) {
    const char* weights = std::getenv("LFS_MOGE2_WEIGHTS");
    if (weights == nullptr || weights[0] == '\0') {
        GTEST_SKIP() << "set LFS_MOGE2_WEIGHTS to run full-model parity";
    }
    int devices = 0;
    ASSERT_EQ(cudaGetDeviceCount(&devices), cudaSuccess);
    ASSERT_GT(devices, 0);

    auto model = lfs::core::nn::models::Moge2::load(weights, lfs::core::Device::CUDA,
                                                    lfs::core::DataType::Float32);
    ASSERT_TRUE(model.has_value()) << std::string(model.error().detail());

    const char* fixture_override = std::getenv("LFS_MOGE2_FIXTURE");
    const std::string path = fixture_override && fixture_override[0]
                                 ? std::string(fixture_override)
                                 : project_root() + "/tests/data/nn/moge2_ref_fixture.json";
    std::ifstream in(path);
    ASSERT_TRUE(static_cast<bool>(in)) << path;
    auto fixture = nlohmann::json::parse(in);
    const auto& ishape = fixture.at("input_shape");
    const int h = ishape[2].get<int>();
    const int w = ishape[3].get<int>();
    const std::int64_t num_tokens = fixture.value("num_tokens", 400);
    auto image = make_test_image(h, w, lfs::core::DataType::Float32);
    auto ran = model->forward_with_taps(image, num_tokens);
    ASSERT_TRUE(ran.has_value()) << std::string(ran.error().detail());
    auto& [out, taps] = *ran;

    const auto report = [&](const char* name, const lfs::core::Tensor& tensor,
                            const nlohmann::json& node) {
        const auto got = host_f32(tensor);
        const float abs_err = max_abs_at(got, node);
        const float rel_err = max_rel_at(got, node);
        std::cout << "tap " << name << " max_abs=" << abs_err << " max_rel=" << rel_err << "\n";
        EXPECT_TRUE(close_sampled(got, node, kF16Rtol, kF16Atol)) << name;
    };

    const auto& nodes = fixture.at("nodes");
    const auto maybe = [&](const char* name, const lfs::core::Tensor& tensor) {
        if (nodes.contains(name) && tensor.is_valid()) {
            report(name, tensor, nodes[name]);
        }
    };
    maybe("patch_embed", taps.patch_embed);
    std::array<std::string, 12> block_names{};
    for (int i = 0; i < 12; ++i) {
        block_names[static_cast<std::size_t>(i)] = std::format("block{}", i);
        maybe(block_names[static_cast<std::size_t>(i)].c_str(),
              taps.blocks[static_cast<std::size_t>(i)]);
    }
    maybe("encoder_feat", taps.encoder_feat);
    maybe("neck0", taps.neck[0]);
    maybe("neck1", taps.neck[1]);
    maybe("neck2", taps.neck[2]);
    maybe("neck3", taps.neck[3]);
    maybe("neck4", taps.neck[4]);
    maybe("points_head", taps.points_head);
    maybe("normal_head", taps.normal_head);
    maybe("mask_head", taps.mask_head);
    maybe("points", out.points);
    maybe("normal", out.normal);
    maybe("mask", out.mask);
    maybe("metric_scale", out.metric_scale);

    ASSERT_TRUE(fixture["full"].contains("normal"));
    const auto& full = fixture["full"]["normal"];
    const auto got_n = host_f32(out.normal);
    const auto& vals = full.at("values");
    ASSERT_EQ(got_n.size(), vals.size());
    float linf = 0.0f;
    for (std::size_t i = 0; i < got_n.size(); ++i) {
        linf = std::max(linf, std::abs(got_n[i] - vals[i].get<float>()));
    }
    std::cout << "normal Linf=" << linf << "\n";
    EXPECT_LE(linf, kNormalLinf);

    const char* home = std::getenv("HOME");
    const std::string weights16 = (home && home[0])
                                      ? std::string(home) + "/.lichtfeld/onnx/moge-2-vitb-normal.lfw"
                                      : "";
    if (!weights16.empty()) {
        std::ifstream probe(weights16, std::ios::binary);
        if (probe.good()) {
            auto model16 = lfs::core::nn::models::Moge2::load(weights16, lfs::core::Device::CUDA,
                                                              lfs::core::DataType::Float16);
            ASSERT_TRUE(model16.has_value()) << std::string(model16.error().detail());
            auto ran16 = model16->forward(image, num_tokens);
            ASSERT_TRUE(ran16.has_value()) << std::string(ran16.error().detail());
            const auto got16 = host_f32(ran16->normal);
            ASSERT_EQ(got16.size(), vals.size());
            float linf16 = 0.0f;
            for (std::size_t i = 0; i < got16.size(); ++i) {
                linf16 = std::max(linf16, std::abs(got16[i] - vals[i].get<float>()));
            }
            std::cout << "fp16 normal Linf=" << linf16 << "\n";
            EXPECT_LE(linf16, 5e-3f);
        }
    }

    const auto alloc0 = lfs::core::alloc_counter::snapshot();
    auto again = model->forward(image, num_tokens);
    ASSERT_TRUE(again.has_value()) << std::string(again.error().detail());
    (void)again->normal.to(lfs::core::Device::CPU);
    const auto driver_allocs = lfs::core::alloc_counter::delta_since(alloc0);
    std::cout << "steady-state driver allocs=" << driver_allocs << "\n";
    EXPECT_EQ(driver_allocs, 0u);
}

TEST(Moge2Test, DeviceFootprintStaysUnderBudget) {
    const char* weights = std::getenv("LFS_MOGE2_WEIGHTS");
    if (weights == nullptr || weights[0] == '\0') {
        GTEST_SKIP() << "set LFS_MOGE2_WEIGHTS to run the VRAM budget check";
    }
    int devices = 0;
    ASSERT_EQ(cudaGetDeviceCount(&devices), cudaSuccess);
    ASSERT_GT(devices, 0);

    const char* home = std::getenv("HOME");
    const std::string weights16 = (home && home[0])
                                      ? std::string(home) + "/.lichtfeld/onnx/moge-2-vitb-normal.lfw"
                                      : "";
    const char* path = weights;
    if (!weights16.empty()) {
        std::ifstream probe(weights16, std::ios::binary);
        if (probe.good()) {
            path = weights16.c_str();
        }
    }

    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    std::size_t free0 = 0;
    std::size_t total = 0;
    ASSERT_EQ(cudaMemGetInfo(&free0, &total), cudaSuccess);

    auto model = lfs::core::nn::models::Moge2::load(path, lfs::core::Device::CUDA,
                                                    lfs::core::DataType::Float16);
    ASSERT_TRUE(model.has_value()) << std::string(model.error().detail());
    auto image = make_test_image(518, 518, lfs::core::DataType::Float32);
    auto ran = model->forward(image, 1800);
    ASSERT_TRUE(ran.has_value()) << std::string(ran.error().detail());
    auto again = model->forward(image, 1800);
    ASSERT_TRUE(again.has_value()) << std::string(again.error().detail());
    (void)again->normal.to(lfs::core::Device::CPU);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    std::size_t free1 = 0;
    ASSERT_EQ(cudaMemGetInfo(&free1, &total), cudaSuccess);
    const std::size_t used = free0 > free1 ? free0 - free1 : 0;
    const std::size_t footprint =
        model->weights_bytes() + model->arena_bytes() + model->workspace_bytes();
    std::cout << "vram delta=" << used << " footprint=" << footprint
              << " weights=" << model->weights_bytes() << " arena=" << model->arena_bytes()
              << " workspace=" << model->workspace_bytes() << "\n";
    constexpr std::size_t kBudget = (12ull * 1024ull * 1024ull * 1024ull) / 10ull;
    EXPECT_LE(footprint, kBudget);
    EXPECT_LE(used, kBudget);
}
