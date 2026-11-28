/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

// Lpips::forward_pair against two forward calls, on whichever GPU backend the test binary selects.

#include "core/nn.hpp"
#include "core/tensor.hpp"
#include "core/tensor_backend.hpp"

#include <array>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <gtest/gtest.h>
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

    // Three different smooth-and-sharp patterns in [0, 1].
    Tensor pattern(const int height, const int width, const int seed) {
        const std::size_t pixels = static_cast<std::size_t>(height) * width;
        std::vector<float> values(3 * pixels);
        for (int y = 0; y < height; ++y)
            for (int x = 0; x < width; ++x) {
                const std::size_t i = static_cast<std::size_t>(y) * width + x;
                values[i] = static_cast<float>((x + (2 + seed) * y) % 97) / 96.0f;
                values[pixels + i] = static_cast<float>((x * (3 + seed) + y) % (17 + seed)) / (16.0f + seed);
                values[2 * pixels + i] = static_cast<float>(((5 + seed) * x + 3 * y) % 29) / 28.0f;
            }
        return Tensor::from_vector(values, lfs::core::TensorShape({3, static_cast<std::size_t>(height),
                                                                   static_cast<std::size_t>(width)}),
                                   Device::GPU);
    }

    bool same_bits(const float a, const float b) { return std::memcmp(&a, &b, sizeof(float)) == 0; }

} // namespace

// The scores forward_pair gives must be those of forward(pred_a, target) and forward(pred_b, target) bit for bit:
// tiled, where the two predictions share each tile's target features, and in one tile; with and without a mask; in
// fp32 and fp16.
TEST(LpipsPair, MatchesTwoCallsBitForBit) {
    const auto path = weights_path();
    if (path.empty() || !fs::is_regular_file(path))
        GTEST_SKIP() << "LPIPS weights absent: " << path;
    constexpr int height = 520, width = 640;
    const auto pred_a = pattern(height, width, 0), pred_b = pattern(height, width, 1);
    const auto target = pattern(height, width, 2);
    std::vector<float> mask_values(static_cast<std::size_t>(height) * width);
    for (std::size_t i = 0; i < mask_values.size(); ++i)
        mask_values[i] = (i / 37) % 5 == 0 ? 0.0f : static_cast<float>(i % 11) / 10.0f;
    const auto mask = Tensor::from_vector(mask_values, lfs::core::TensorShape({height, width}), Device::GPU);

    for (const auto dtype : {DataType::Float32, DataType::Float16}) {
        for (const bool tiled : {true, false}) {
            // Tiled: the largest budget, halving from 256 MiB, that still cuts the width into at least two tiles.
            std::size_t budget = tiled ? std::size_t{256} << 20 : std::size_t{1536} << 20;
            auto loaded = Lpips::load(path, Device::GPU, dtype, InputScaling::Identity, budget);
            ASSERT_TRUE(loaded.has_value()) << loaded.error().detail();
            while (tiled && loaded->tile_size_for(height, width) + 224 >= static_cast<std::size_t>(width) &&
                   budget > (std::size_t{16} << 20)) {
                budget /= 2;
                loaded = Lpips::load(path, Device::GPU, dtype, InputScaling::Identity, budget);
                ASSERT_TRUE(loaded.has_value()) << loaded.error().detail();
            }
            auto& model = *loaded;
            if (tiled)
                ASSERT_LT(model.tile_size_for(height, width) + 224, static_cast<std::size_t>(width));
            else
                ASSERT_GE(model.tile_size_for(height, width), static_cast<std::size_t>(width));
            const std::string what = std::string(dtype == DataType::Float32 ? "fp32" : "fp16") +
                                     (tiled ? " tiled" : " one tile");
            for (const bool masked : {false, true}) {
                const auto a = masked ? model.forward(pred_a, target, mask) : model.forward(pred_a, target);
                const auto b = masked ? model.forward(pred_b, target, mask) : model.forward(pred_b, target);
                const auto pair = masked ? model.forward_pair(pred_a, pred_b, target, mask)
                                         : model.forward_pair(pred_a, pred_b, target);
                ASSERT_TRUE(a.has_value()) << what << ": " << a.error().detail();
                ASSERT_TRUE(b.has_value()) << what << ": " << b.error().detail();
                ASSERT_TRUE(pair.has_value()) << what << ": " << pair.error().detail();
                // An fp16 model on CUDA takes the fused fast path even without a dispatch, and that path sums its
                // scores with atomics, so two runs of one call already differ in the last bits: there the pair is held
                // to two calls within 1e-6. Every other case runs the exact path, bit for bit.
                if (dtype == DataType::Float16 && lfs::core::default_gpu_backend() == lfs::core::GpuBackend::CUDA) {
                    EXPECT_NEAR((*pair)[0], *a, 1e-6f) << what << (masked ? " masked" : "");
                    EXPECT_NEAR((*pair)[1], *b, 1e-6f) << what << (masked ? " masked" : "");
                } else {
                    EXPECT_TRUE(same_bits((*pair)[0], *a)) << what << (masked ? " masked" : "") << ": " << (*pair)[0]
                                                           << " against " << *a;
                    EXPECT_TRUE(same_bits((*pair)[1], *b)) << what << (masked ? " masked" : "") << ": " << (*pair)[1]
                                                           << " against " << *b;
                }
                EXPECT_FALSE(same_bits(*a, *b)) << what << ": the two predictions should score differently";
            }
            model.release_activations();
        }
    }
}
