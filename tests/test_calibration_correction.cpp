// SPDX-License-Identifier: GPL-3.0-or-later
#include "training/components/calibration_correction.hpp"
#include "training/losses/photometric_loss.hpp"
#include <array>
#include <cstring>
#include <gtest/gtest.h>

namespace {
    using namespace lfs::core;
    using namespace lfs::training;
    Tensor warp(CalibrationCorrection& fit, int slot, const Tensor& photo, const Tensor& render, float fx, float fy, int x = 0, int y = 0) {
        Tensor mask, structure, edge;
        return fit.warp(slot, fx, fy, x, y, photo, render, mask, structure, edge);
    }
} // namespace
TEST(CalibrationCorrectionTest, ZeroRowPreservesPixels) {
    CalibrationCorrection fit(1);
    std::vector<float> pixels(3 * 31 * 67);
    for (size_t i = 0; i < pixels.size(); ++i)
        pixels[i] = static_cast<float>(i % 511) / 510.f;
    pixels[0] = -0.f;
    auto image = Tensor::from_vector(pixels, {3, 31, 67}, Device::CUDA);
    auto bytes = (image * 255.f).to(DataType::UInt8);
    const auto actual = warp(fit, 0, image, image, 300, 270).to_vector();
    EXPECT_EQ(std::memcmp(actual.data(), pixels.data(), pixels.size() * sizeof(float)), 0);
    auto normalized = bytes.to(DataType::Float32) * (1.f / 255.f);
    EXPECT_EQ(warp(fit, 0, bytes, image, 300, 270).to_vector(), normalized.to_vector());
}

namespace {
    Tensor texture(int w, int h) {
        std::vector<float> data(3 * w * h);
        for (int c = 0; c < 3; ++c)
            for (int y = 0; y < h; ++y)
                for (int x = 0; x < w; ++x) {
                    // Distinct wave directions constrain both components of the displacement.
                    data[(c * h + y) * w + x] = .5f + .28f * std::sin(.59f * x + .17f * y + 2.1f * c) + .19f * std::cos(.21f * x - .61f * y + 1.7f * c);
                }
        return Tensor::from_vector(data, {3, size_t(h), size_t(w)}, Device::CUDA);
    }
    void set_field(CalibrationCorrection& fit, const std::vector<float>& row) {
        fit.field() = Tensor::from_vector(row, fit.field().shape(), Device::CUDA);
    }
} // namespace
TEST(CalibrationCorrectionTest, AngularDisplacementScalesWithResolutionAndUsesFullImageCoordinates) {
    CalibrationCorrection fit(1);
    std::vector<float> field(162);
    for (int i = 0; i < 81; ++i) {
        field[2 * i] = 2.5f;
        field[2 * i + 1] = -1.25f;
    }
    set_field(fit, field);
    for (int scale : {1, 2}) {
        const int w = 80 / scale, h = 64 / scale, x0 = 7, y0 = 6;
        std::vector<float> ramp(3 * w * h);
        for (int c = 0; c < 3; ++c)
            for (int y = 0; y < h; ++y)
                for (int x = 0; x < w; ++x)
                    ramp[(c * h + y) * w + x] = .001f * x + .005f * y + .1f * c;
        auto photo = Tensor::from_vector(ramp, {3, size_t(h), size_t(w)}, Device::CUDA);
        auto tile = Tensor::zeros({3, 9, 13}, Device::CUDA);
        auto structure = photo.slice(0, 0, 1).squeeze(0), edge = structure;
        std::vector<float> bands(w * h);
        for (int i = 0; i < w * h; ++i)
            bands[i] = static_cast<float>((i * 13) % 251);
        auto mask = Tensor::from_vector(bands, {size_t(h), size_t(w)}, Device::CUDA).to(DataType::UInt8);
        const auto out = fit.warp(0, 800.f / scale, 1200.f / scale, x0, y0, photo, tile, mask, structure, edge).to_vector();
        const auto mask_values = mask.to(DataType::Float32).to_vector(), edges = edge.to_vector(), ridges = structure.to_vector();
        for (int y = 0; y < 9; ++y)
            for (int x = 0; x < 13; ++x) {
                const float px = x + x0 + 2.f / scale, py = y + y0 - 1.5f / scale;
                const int i = y * 13 + x;
                for (int c = 0; c < 3; ++c)
                    EXPECT_NEAR(out[c * 9 * 13 + i], .001f * px + .005f * py + .1f * c, 1e-6);
                EXPECT_NEAR(edges[i], .001f * px + .005f * py, 1e-6);
                EXPECT_EQ(ridges[i], edges[i]);
                EXPECT_EQ(mask_values[i], bands[int(py + .5f) * w + int(px + .5f)]);
            }
    }
}
TEST(CalibrationCorrectionTest, MaskedTileEquationsMatchHostAndHaveFixedReductionOrder) {
    constexpr int w = 65, h = 47, tw = 31, th = 27, x0 = 11, y0 = 9;
    auto photo = texture(w, h), render = texture(tw, th);
    auto g = texture(tw, th) * .00002f;
    const float values = 3 * tw * th;
    const auto grad = g.to_vector(), target = render.to_vector(), source = photo.to_vector();
    std::vector<float> mask(tw * th);
    for (int i = 0; i < tw * th; ++i)
        mask[i] = (i % 5) != 0;
    auto weight = Tensor::from_vector(mask, {th, tw}, Device::CUDA);
    double sums[81][5]{};
    for (int y = 1; y < th - 1; ++y)
        for (int x = 1; x < tw - 1; ++x) {
            const int i = y * tw + x, p = (y + y0) * w + x + x0;
            if (!mask[i])
                continue;
            double q[5]{}, ss = 0;
            for (int c = 0; c < 3; ++c) {
                const auto* src = source.data() + c * w * h;
                const double gx = .4 * (src[p + 1] - src[p - 1]), gy = .6 * (src[p + w] - src[p - w]);
                const double residual = src[p] - target[c * tw * th + i];
                const double v = -.03 * values * grad[c * tw * th + i];
                ss += residual * residual;
                q[0] += gx * v;
                q[1] += gy * v;
                q[2] += gx * gx;
                q[3] += gx * gy;
                q[4] += gy * gy;
            }
            const double rho = .03 / std::sqrt(ss / 3 + .03 * .03);
            const double u = (x + x0 + .5) * 8 / w, v = (y + y0 + .5) * 8 / h;
            for (int j = int(v); j <= int(v) + 1; ++j)
                for (int k = int(u); k <= int(u) + 1; ++k) {
                    const double b = (1 - std::abs(u - k)) * (1 - std::abs(v - j));
                    for (int f = 0; f < 5; ++f)
                        sums[j * 9 + k][f] += b * q[f] * (f < 2 ? 1 : rho);
                }
        }
    double mean = 0;
    for (auto& q : sums)
        mean += (q[2] + q[4]) / 162;
    std::vector<float> previous;
    for (int repeat = 0; repeat < 2; ++repeat) {
        CalibrationCorrection fit(1);
        warp(fit, 0, photo, render, 800, 1200, x0, y0);
        fit.accumulate(render, g, weight, values);
        // A second, entirely masked tile must preserve the first tile's equations.
        auto next = texture(17, 9);
        auto excluded = Tensor::zeros({9, 17}, Device::CUDA);
        Tensor mask_map, structure, edge;
        fit.warp(0, 800, 1200, 45, 0, photo, next, mask_map, structure, edge, false);
        auto next_gradient = next.clone();
        fit.accumulate(next, next_gradient, excluded, next.numel());
        fit.update();
        const auto actual = fit.field().to_vector();
        if (repeat)
            EXPECT_EQ(std::memcmp(actual.data(), previous.data(), actual.size() * sizeof(float)), 0);
        previous = actual;
        for (int i = 0; i < 81; ++i) {
            int degree = 4 - (i % 9 == 0 || i % 9 == 8) - (i / 9 == 0 || i / 9 == 8);
            const auto& q = sums[i];
            double d = mean * (.1 * degree + .11), a = q[2] + d, c = q[4] + d, b = q[3];
            const double det = a * c - b * b;
            double dx = (c * q[0] - b * q[1]) / det, dy = (a * q[1] - b * q[0]) / det;
            const double factor = -.5 * std::min(1., 1.5 / std::hypot(.8 * dx, 1.2 * dy));
            EXPECT_NEAR(actual[2 * i], factor * dx, 3e-6);
            EXPECT_NEAR(actual[2 * i + 1], factor * dy, 3e-6);
        }
    }
}
TEST(CalibrationCorrectionTest, PhotometricLossRecoversSmoothDisplacement) {
    constexpr int w = 256, h = 192;
    auto photo = texture(w, h);
    std::vector<float> planted(162);
    for (int y = 0; y < 9; ++y)
        for (int x = 0; x < 9; ++x) {
            planted[2 * (y * 9 + x)] = 2.5f + .45f * std::sin(x * .45f);
            planted[2 * (y * 9 + x) + 1] = .4f * std::cos(y * .4f);
        }
    CalibrationCorrection known(1);
    set_field(known, planted);
    const auto render = warp(known, 0, photo, photo, 1000, 1000);
    for (float mix : {0.f, .2f}) {
        CalibrationCorrection fit(2);
        losses::PhotometricLoss loss;
        for (int iteration = 0; iteration < 30; ++iteration) {
            const auto target = warp(fit, 0, photo, render, 1000, 1000);
            auto result = loss.forward(render, target, {.lambda_dssim = mix});
            ASSERT_TRUE(result) << result.error();
            fit.accumulate(render, result->second.grad_image, {}, mix == 0 ? 3 * w * h : 3 * (w - 10) * (h - 10));
            fit.update();
        }
        auto estimated = fit.field().to_vector();
        double sq = 0;
        for (int i = 0; i < 162; ++i) {
            sq += std::pow(estimated[i] - planted[i], 2);
            EXPECT_EQ(estimated[i + 162], 0.f);
        }
        const double rms = std::sqrt(sq / 81);
        RecordProperty(mix == 0 ? "l1_rms_px" : "mixed_rms_px", std::to_string(rms));
        EXPECT_LT(rms, .1) << "DSSIM weight " << mix;
    }
}
TEST(CalibrationCorrectionTest, IdentityExclusionsAndStepLimit) {
    CalibrationCorrection fit(1);
    auto photo = texture(73, 57);
    losses::PhotometricLoss loss;
    for (float mix : {0.f, .2f}) {
        auto target = warp(fit, 0, photo, photo, 1000, 700);
        auto result = loss.forward(photo, target, {.lambda_dssim = mix});
        ASSERT_TRUE(result);
        fit.accumulate(photo, result->second.grad_image, {}, photo.numel());
        fit.update();
        EXPECT_EQ(fit.field().to_vector(), std::vector<float>(162));
    }
    auto render = photo * .8f;
    auto gradient = photo * 10;
    auto mask = Tensor::zeros({57, 73}, Device::CUDA);
    warp(fit, 0, photo, render, 1000, 700);
    fit.accumulate(render, gradient, mask, photo.numel());
    fit.update();
    EXPECT_EQ(fit.field().to_vector(), std::vector<float>(162));
    warp(fit, 0, photo, render, 1000, 700);
    fit.accumulate(render, gradient, {}, photo.numel());
    fit.update();
    auto moved = fit.field().to_vector();
    float largest = 0;
    for (int i = 0; i < 81; ++i)
        largest = std::max(largest, std::hypot(moved[2 * i], .7f * moved[2 * i + 1]));
    EXPECT_NEAR(largest, .75f, 1e-5);
    std::vector<float> outside(162, 1000.f);
    set_field(fit, outside);
    EXPECT_EQ(warp(fit, 0, photo, render, 1000, 700).to_vector(), render.to_vector());
    fit.accumulate(render, gradient, {}, photo.numel());
    fit.update();
    EXPECT_EQ(fit.field().to_vector(), outside);
}

namespace {
    constexpr int scoring_width = 512, scoring_height = 384;

    // Evaluate the continuous texture separately at the render and photograph coordinates.
    float scene_value(float x, float y, int c) {
        return .5f + .19f * std::sin(.103f * x + .047f * y + 1.7f * c) +
               .17f * std::cos(.035f * x - .139f * y + 2.1f * c) +
               .09f * std::sin(.161f * x + .093f * y + c);
    }

    std::array<float, 2> planted_shift(int x, int y) {
        const float a = (x + .5f) * 6.2831853f / scoring_width;
        const float b = (y + .5f) * 6.2831853f / scoring_height;
        return {4.f + 2.7f * std::sin(a) * std::cos(b), 1.1f * std::cos(a) * std::sin(b)};
    }

    Tensor scoring_image(bool shifted, bool contradict = false) {
        std::vector<float> pixels(3 * scoring_width * scoring_height);
        for (int y = 0; y < scoring_height; ++y)
            for (int x = 0; x < scoring_width; ++x) {
                const auto [dx, dy] = shifted ? planted_shift(x, y) : std::array<float, 2>{};
                for (int c = 0; c < 3; ++c)
                    pixels[(c * scoring_height + y) * scoring_width + x] =
                        scene_value(x + (contradict && x > 256 ? -dx : dx), y + dy, c);
            }
        return Tensor::from_vector(pixels, {3, scoring_height, scoring_width}, Device::CUDA);
    }
} // namespace

TEST(CalibrationScoring, RecoversTranslationAndSmoothWarp) {
    auto image = scoring_image(false), target = scoring_image(true);
    Tensor field;
    auto aligned = CalibrationCorrection::align_for_scoring(image, target, {}, field);
    ASSERT_EQ(field.shape()[1], 16);
    const auto grid = field.to_vector();
    double squared = 0;
    for (int y = 0; y < scoring_height; ++y)
        for (int x = 0; x < scoring_width; ++x) {
            const float u = (x + .5f) * 15 / scoring_width, v = (y + .5f) * 15 / scoring_height;
            const auto shift = planted_shift(x, y);
            for (int c = 0; c < 2; ++c) {
                double d = 0;
                for (int j = int(v); j <= int(v) + 1; ++j)
                    for (int i = int(u); i <= int(u) + 1; ++i)
                        d += (1 - std::abs(u - i)) * (1 - std::abs(v - j)) * grid[2 * (16 * j + i) + c];
                squared += (d - shift[c]) * (d - shift[c]);
            }
        }
    const double rms = std::sqrt(squared / (scoring_width * scoring_height));
    const auto mse = [&](const Tensor& a) {
        const auto residual = a - target;
        return (residual * residual).mean().item<float>();
    };
    const double gain = 10 * std::log10(mse(image) / mse(aligned));
    EXPECT_LT(rms, .15);
    EXPECT_GE(gain, 10.);
    RecordProperty("rms_px", std::to_string(rms));
    RecordProperty("gain_db", std::to_string(gain));
}

TEST(CalibrationScoring, IdenticalImagesAreBitExact) {
    auto image = scoring_image(false);
    Tensor field;
    const auto actual = CalibrationCorrection::align_for_scoring(image, image, {}, field).to_vector();
    const auto expected = image.to_vector();
    EXPECT_EQ(std::memcmp(actual.data(), expected.data(), expected.size() * sizeof(float)), 0);
    EXPECT_EQ(field.abs().max().item<float>(), 0.f);
}

TEST(CalibrationScoring, ContradictionOutsideMaskHasNoInfluenceAtAnyLevel) {
    auto image = scoring_image(false);
    std::vector<float> pixels(scoring_width * scoring_height);
    for (size_t i = 0; i < pixels.size(); ++i)
        pixels[i] = i % scoring_width <= 256;
    auto mask = Tensor::from_vector(pixels, {scoring_height, scoring_width}, Device::CUDA);
    Tensor baseline, contradicted;
    CalibrationCorrection::align_for_scoring(image, scoring_image(true), mask, baseline);
    CalibrationCorrection::align_for_scoring(image, scoring_image(true, true), mask, contradicted);
    EXPECT_GT(baseline.abs().max().item<float>(), 4.f);
    EXPECT_EQ(baseline.to_vector(), contradicted.to_vector());
}

// A brightness difference over a weak gradient is not misalignment, and no fit may run away.
TEST(CalibrationScoring, BrightnessOffsetIsNotAShiftAndDisplacementIsBounded) {
    std::vector<float> sky(3 * scoring_width * scoring_height), shifted(sky.size());
    for (int c = 0; c < 3; ++c)
        for (int y = 0; y < scoring_height; ++y)
            for (int x = 0; x < scoring_width; ++x) {
                const size_t i = (static_cast<size_t>(c) * scoring_height + y) * scoring_width + x;
                sky[i] = .45f + .1f * x / scoring_width + .05f * y / scoring_height;
                shifted[i] = scene_value(x + 60.f, y, c);
            }
    const auto ramp = Tensor::from_vector(sky, {3, scoring_height, scoring_width}, Device::CUDA);
    Tensor field;
    CalibrationCorrection::align_for_scoring(ramp, ramp + .05f, {}, field);
    EXPECT_LT(field.abs().max().item<float>(), .5f);

    CalibrationCorrection::align_for_scoring(
        scoring_image(false), Tensor::from_vector(shifted, {3, scoring_height, scoring_width}, Device::CUDA), {}, field);
    const auto moved = field.to_vector();
    float reach = 0;
    for (size_t i = 0; i < moved.size(); i += 2)
        reach = std::max(reach, std::hypot(moved[i], moved[i + 1]));
    EXPECT_LE(reach, .05f * scoring_height * 1.0001f);
}
