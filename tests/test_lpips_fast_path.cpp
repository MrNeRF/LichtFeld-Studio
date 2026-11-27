/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

// The fused fast path (a backend's LpipsOps dispatch) against the exact path of the generic ops, on whichever GPU
// backend the test binary selects.

#include "core/image_io.hpp"
#include "core/nn.hpp"
#include "core/tensor.hpp"
#include "core/tensor_backend.hpp"
#include "lfs/training/ops/registry.hpp"

#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <filesystem>
#include <gtest/gtest.h>
#include <iostream>
#include <string>
#include <vector>

namespace {

    namespace fs = std::filesystem;
    using lfs::core::DataType;
    using lfs::core::Device;
    using lfs::core::Tensor;
    using lfs::core::nn::models::InputScaling;
    using lfs::core::nn::models::Lpips;

    fs::path weights_path() {
        if (const char* env = std::getenv("LFS_LPIPS_WEIGHTS"); env && env[0])
            return env;
        if (const char* home = std::getenv("HOME"); home && home[0])
            return fs::path(home) / ".lichtfeld/onnx/lpips-vgg16-v0.1.lfw";
        return {};
    }

    Tensor load_rgb(const fs::path& path) {
        auto [data, width, height, channels] = lfs::core::load_image(path);
        if (!data || channels != 3)
            throw std::runtime_error("failed to load RGB image " + path.string());
        std::vector<float> chw(static_cast<std::size_t>(3) * width * height);
        const std::size_t plane = static_cast<std::size_t>(width) * height;
        for (std::size_t pixel = 0; pixel < plane; ++pixel)
            for (std::size_t c = 0; c < 3; ++c)
                chw[c * plane + pixel] = static_cast<float>(data[pixel * 3 + c]) / 255.0f;
        lfs::core::free_image(data);
        return Tensor::from_vector(chw, lfs::core::TensorShape({3, static_cast<std::size_t>(height),
                                                                static_cast<std::size_t>(width)}),
                                   Device::GPU);
    }

} // namespace

// LosslessCropPairMatchesReference lets fp16 differ from fp32 by 5e-4 on CUDA; the fast path must stay within that of
// the exact path, untiled and tiled, with and without a mask.
TEST(LpipsFastPath, MatchesTheExactPath) {
    const auto path = weights_path();
    if (path.empty() || !fs::is_regular_file(path))
        GTEST_SKIP() << "LPIPS weights absent: " << path;
    const auto* dispatch = lfs::training::training_ops(lfs::core::default_gpu_backend()).lpips;
    if (dispatch == nullptr)
        GTEST_SKIP() << "No LPIPS dispatch on this backend";
    const auto root = fs::path(PROJECT_ROOT_PATH);
    const auto crop_a = root / "tests/data/nn/lpips_crop_a.png", crop_b = root / "tests/data/nn/lpips_crop_b.png";
    if (!fs::is_regular_file(crop_a) || !fs::is_regular_file(crop_b))
        GTEST_SKIP() << "LPIPS lossless crop pair is absent";
    const auto a = load_rgb(crop_a), b = load_rgb(crop_b);
    const auto height = a.shape()[1], width = a.shape()[2];
    std::vector<float> mask_values(height * width);
    for (std::size_t i = 0; i < mask_values.size(); ++i)
        mask_values[i] = (i / 53) % 7 == 0 ? 0.0f : 1.0f;
    const auto mask = Tensor::from_vector(mask_values, lfs::core::TensorShape({height, width}), Device::GPU);

    for (const std::size_t budget : {std::size_t{1536} << 20, std::size_t{48} << 20}) {
        auto exact = Lpips::load(path, Device::GPU, DataType::Float16, InputScaling::Identity, budget);
        auto fast = Lpips::load(path, Device::GPU, DataType::Float16, InputScaling::Identity, budget);
        ASSERT_TRUE(exact.has_value()) << exact.error().detail();
        ASSERT_TRUE(fast.has_value()) << fast.error().detail();
        fast->set_dispatch(*dispatch);
        for (const bool masked : {false, true}) {
            const auto expected = masked ? exact->forward(a, b, mask) : exact->forward(a, b);
            const auto actual = masked ? fast->forward(a, b, mask) : fast->forward(a, b);
            ASSERT_TRUE(expected.has_value()) << expected.error().detail();
            ASSERT_TRUE(actual.has_value()) << actual.error().detail();
            std::cout << "LPIPS_FAST_PATH budget=" << (budget >> 20) << "MiB tile="
                      << fast->tile_size_for(static_cast<int>(height), static_cast<int>(width))
                      << (masked ? " masked" : "") << " exact=" << *expected << " fast=" << *actual
                      << " abs_delta=" << std::abs(*actual - *expected) << std::endl;
            EXPECT_NEAR(*actual, *expected, 5e-4f);
        }
    }
}
