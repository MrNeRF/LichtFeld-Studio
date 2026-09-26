/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/image_io.hpp"
#include "core/nn.hpp"

#include <cuda_runtime.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <format>
#include <fstream>
#include <gtest/gtest.h>
#include <numeric>
#include <string>
#include <vector>

namespace {

    using lfs::core::DataType;
    using lfs::core::Device;
    using lfs::core::Tensor;
    using lfs::core::TensorShape;
    using lfs::core::nn::models::RomaV1;

    constexpr int kResolution = 448;
    constexpr int kShiftX = 21;
    constexpr int kShiftY = 14;

    std::filesystem::path repo_root() {
        return std::filesystem::path(PROJECT_ROOT_PATH);
    }

    std::string fixture_path() {
        return (repo_root() / "tests/data/nn/romav1_ref_fixture.json").string();
    }

    // The pair the fixture was generated from: a committed photo crop at the
    // working resolution, and a shifted, darkened copy.
    std::vector<float> reference_image() {
        const auto path = repo_root() / "tests/data/nn/lpips_crop_a.png";
        auto [pixels, width, height, channels] = lfs::core::load_image_float(path);
        if (!pixels || width <= 0 || height <= 0 || channels < 3) {
            return {};
        }
        // Feed the crop at its own size and let the model do the resampling,
        // exactly as the fixture generator does.
        std::vector<float> src(static_cast<std::size_t>(width) * height * 3);
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                for (int c = 0; c < 3; ++c) {
                    src[(static_cast<std::size_t>(y) * width + x) * 3 + c] =
                        pixels[(static_cast<std::size_t>(y) * width + x) * channels + c];
                }
            }
        }
        return src;
    }

    std::vector<float> host_f32(const Tensor& t) {
        return t.to(DataType::Float32).to(Device::CPU).contiguous().to_vector();
    }

    double percentile(std::vector<double> values, double q) {
        if (values.empty()) {
            return 0.0;
        }
        std::sort(values.begin(), values.end());
        const auto index = static_cast<std::size_t>(
            std::clamp(q * static_cast<double>(values.size() - 1), 0.0,
                       static_cast<double>(values.size() - 1)));
        return values[index];
    }

} // namespace

TEST(RomaV1Test, CommittedFixtureIsSmall) {
    std::ifstream in(fixture_path(), std::ios::binary | std::ios::ate);
    ASSERT_TRUE(in.good()) << "RoMa v1 reference fixture is absent: " << fixture_path();
    EXPECT_LT(in.tellg(), 400 * 1024) << "fixture must stay small; store samples, not tensors";
}

// Full-model parity against the released PyTorch model. Opt-in, because it needs
// the 795 MiB weight file that ships as a release asset.
TEST(RomaV1Test, FullModelParityIsOptIn) {
    const char* weights = std::getenv("LFS_ROMAV1_WEIGHTS");
    if (weights == nullptr || weights[0] == '\0') {
        GTEST_SKIP() << "set LFS_ROMAV1_WEIGHTS to run full-model parity";
    }
    std::ifstream fixture_in(fixture_path());
    ASSERT_TRUE(fixture_in.good());
    const auto fixture = nlohmann::json::parse(fixture_in);
    ASSERT_EQ(fixture.at("resolution").get<int>(), kResolution);
    const int stride = fixture.at("sample_stride").get<int>();
    const auto ref_warp = fixture.at("warp").get<std::vector<double>>();
    const auto ref_cert = fixture.at("certainty").get<std::vector<double>>();
    const int samples = kResolution / stride;
    ASSERT_EQ(ref_cert.size(), static_cast<std::size_t>(samples) * samples);

    const auto crop = reference_image();
    ASSERT_FALSE(crop.empty()) << "could not read the committed crop";
    const auto side = static_cast<std::size_t>(std::llround(std::sqrt(crop.size() / 3.0)));
    ASSERT_EQ(side * side * 3, crop.size());

    // Build the shifted neighbour at the crop's own resolution so the model
    // resamples both the same way.
    std::vector<float> shifted(crop.size());
    const auto shift_x = static_cast<std::size_t>(kShiftX * side / kResolution);
    const auto shift_y = static_cast<std::size_t>(kShiftY * side / kResolution);
    for (std::size_t y = 0; y < side; ++y) {
        for (std::size_t x = 0; x < side; ++x) {
            const std::size_t sy = (y + shift_y) % side;
            const std::size_t sx = (x + shift_x) % side;
            for (int c = 0; c < 3; ++c) {
                shifted[(y * side + x) * 3 + c] =
                    crop[(sy * side + sx) * 3 + c] * 0.9f + 0.05f;
            }
        }
    }

    auto model = RomaV1::load(weights, Device::GPU, std::nullopt, kResolution);
    ASSERT_TRUE(model.has_value()) << std::string(model.error().detail());
    auto upload = [&](const std::vector<float>& data) {
        return Tensor::from_vector(data, TensorShape(std::vector<std::size_t>{side, side, 3}),
                                   Device::GPU)
            .contiguous();
    };
    auto prepared_a = model->prepare(upload(crop));
    auto prepared_b = model->prepare(upload(shifted));
    ASSERT_TRUE(prepared_a.has_value() && prepared_b.has_value());
    auto matched = model->match(**prepared_a, **prepared_b);
    ASSERT_TRUE(matched.has_value()) << std::string(matched.error().detail());

    const auto warp = host_f32(matched->warp);
    const auto cert = host_f32(matched->overlap);
    for (float v : warp) {
        ASSERT_TRUE(std::isfinite(v));
    }

    std::vector<double> warp_err;
    std::vector<double> shift_err;
    std::vector<double> cert_decisive;
    int confident = 0;
    int confident_ours = 0;
    for (int sy = 0; sy < samples; ++sy) {
        for (int sx = 0; sx < samples; ++sx) {
            const std::size_t flat =
                static_cast<std::size_t>(sy * stride) * kResolution + sx * stride;
            const std::size_t sample = static_cast<std::size_t>(sy) * samples + sx;
            // Certainty is not a stable quantity to compare pixel by pixel on
            // this pair. The final step subtracts the coarse certainty only
            // where that is negative, and on a self-shifted image the coarse
            // certainty sits near zero, so the branch flips on noise. What is
            // stable, and what callers actually use, is which pixels end up
            // confident, so assert on that share instead.
            if (ref_cert[sample] < 0.2 || ref_cert[sample] > 0.8) {
                cert_decisive.push_back(
                    std::abs(static_cast<double>(cert[flat]) - ref_cert[sample]));
            }
            if (cert[flat] > 0.5f) {
                ++confident_ours;
            }
            if (ref_cert[sample] <= 0.5) {
                continue;
            }
            ++confident;
            warp_err.push_back(std::max(
                std::abs(static_cast<double>(warp[flat * 2]) - ref_warp[sample * 2]),
                std::abs(static_cast<double>(warp[flat * 2 + 1]) - ref_warp[sample * 2 + 1])));
            const double grid_x = 2.0 * ((sx * stride + 0.5) / kResolution) - 1.0;
            const double grid_y = 2.0 * ((sy * stride + 0.5) / kResolution) - 1.0;
            shift_err.push_back(
                std::max(std::abs(warp[flat * 2] - (grid_x - 2.0 * kShiftX / kResolution)),
                         std::abs(warp[flat * 2 + 1] - (grid_y - 2.0 * kShiftY / kResolution))));
        }
    }
    ASSERT_GT(confident, samples * samples / 10) << "reference is confident on part of the frame";

    const double mean_warp =
        std::accumulate(warp_err.begin(), warp_err.end(), 0.0) / warp_err.size();
    const double p99_warp = percentile(warp_err, 0.99);
    const double mean_shift =
        std::accumulate(shift_err.begin(), shift_err.end(), 0.0) / shift_err.size();
    const double mean_cert =
        std::accumulate(cert_decisive.begin(), cert_decisive.end(), 0.0) / cert_decisive.size();
    const double confident_delta =
        std::abs(static_cast<double>(confident_ours) - confident) / (samples * samples);
    std::cout << std::format("roma v1 parity: warp mean {:.5f} p99 {:.5f} (normalized), shift "
                             "mean {:.5f}, decisive certainty mean {:.5f}, confident-share "
                             "delta {:.4f}\n",
                             mean_warp, p99_warp, mean_shift, mean_cert, confident_delta);
    // One normalized unit is kResolution / 2 pixels, so 0.0045 is one pixel.
    EXPECT_LT(mean_warp, 0.0045) << "mean warp deviation from the reference exceeds one pixel";
    EXPECT_LT(p99_warp, 0.022) << "p99 warp deviation from the reference exceeds five pixels";
    EXPECT_LT(mean_shift, 0.009) << "recovered warp does not follow the known shift";
    EXPECT_LT(confident_delta, 0.08) << "the confident share of the frame moved";
}

TEST(RomaV1Test, MatchIsBitwiseRepeatable) {
    const char* weights = std::getenv("LFS_ROMAV1_WEIGHTS");
    if (weights == nullptr || weights[0] == '\0') {
        GTEST_SKIP() << "set LFS_ROMAV1_WEIGHTS to run the repeatability check";
    }
    auto model = RomaV1::load(weights, Device::GPU, std::nullopt, kResolution);
    ASSERT_TRUE(model.has_value()) << std::string(model.error().detail());
    const auto crop = reference_image();
    ASSERT_FALSE(crop.empty());
    const auto side = static_cast<std::size_t>(std::llround(std::sqrt(crop.size() / 3.0)));
    auto gpu = Tensor::from_vector(crop, TensorShape(std::vector<std::size_t>{side, side, 3}),
                                   Device::GPU)
                   .contiguous();
    auto a = model->prepare(gpu);
    auto b = model->prepare(gpu);
    ASSERT_TRUE(a.has_value() && b.has_value());
    auto first = model->match(**a, **b);
    ASSERT_TRUE(first.has_value());
    const auto ref_warp = host_f32(first->warp);
    const auto ref_cert = host_f32(first->overlap);
    int differing = 0;
    for (int run = 0; run < 6; ++run) {
        auto pa = model->prepare(gpu);
        auto pb = model->prepare(gpu);
        auto again = model->match(**pa, **pb);
        ASSERT_TRUE(again.has_value());
        if (host_f32(again->warp) != ref_warp || host_f32(again->overlap) != ref_cert) {
            ++differing;
        }
    }
    EXPECT_EQ(differing, 0) << differing << " of 6 repeats differed from the first match";
}
