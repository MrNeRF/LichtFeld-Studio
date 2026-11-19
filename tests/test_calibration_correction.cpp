// SPDX-License-Identifier: GPL-3.0-or-later
#include "training/components/calibration_correction.hpp"
#include "training/losses/photometric_loss.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <gtest/gtest.h>
#include <vector>

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
// The curvature carries each pixel's weight, like the loss gradient, and leaves out the loss's valid padding.
TEST(CalibrationCorrectionTest, MaskedTileEquationsMatchHostAndHaveFixedReductionOrder) {
    for (const int margin : {0, 5}) {
        SCOPED_TRACE(margin);
        constexpr int w = 65, h = 47, tw = 31, th = 27, x0 = 11, y0 = 9;
        auto photo = texture(w, h), render = texture(tw, th);
        auto g = texture(tw, th) * .00002f;
        const float values = 3 * tw * th;
        const auto grad = g.to_vector(), target = render.to_vector(), source = photo.to_vector();
        // Masked, soft, thin-structure and crop-box weights.
        constexpr float weights[] = {0.f, .35f, 1.f, 2.5f, .8f};
        std::vector<float> mask(tw * th);
        for (int i = 0; i < tw * th; ++i)
            mask[i] = weights[i % 5];
        auto weight = Tensor::from_vector(mask, {th, tw}, Device::CUDA);
        double sums[81][5]{};
        for (int y = 1; y < th - 1; ++y)
            for (int x = 1; x < tw - 1; ++x) {
                const int i = y * tw + x, p = (y + y0) * w + x + x0;
                if (!mask[i])
                    continue;
                const bool counted = x >= margin && y >= margin && x < tw - margin && y < th - margin;
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
                            sums[j * 9 + k][f] += b * q[f] * (f < 2 ? 1 : counted ? rho * mask[i]
                                                                                  : 0);
                    }
            }
        double mean = 0;
        for (auto& q : sums)
            mean += (q[2] + q[4]) / 162;
        std::vector<float> previous;
        for (int repeat = 0; repeat < 2; ++repeat) {
            CalibrationCorrection fit(1);
            warp(fit, 0, photo, render, 800, 1200, x0, y0);
            fit.accumulate(render, g, weight, values, margin);
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
    constexpr int side = CalibrationCorrection::grid;
    ASSERT_EQ(field.numel(), size_t(2 * side * side));
    const auto grid = field.to_vector();
    double squared = 0;
    for (int y = 0; y < scoring_height; ++y)
        for (int x = 0; x < scoring_width; ++x) {
            const float u = (x + .5f) * (side - 1) / scoring_width, v = (y + .5f) * (side - 1) / scoring_height;
            const auto shift = planted_shift(x, y);
            for (int c = 0; c < 2; ++c) {
                double d = 0;
                for (int j = int(v); j <= int(v) + 1; ++j)
                    for (int i = int(u); i <= int(u) + 1; ++i)
                        d += (1 - std::abs(u - i)) * (1 - std::abs(v - j)) * grid[2 * (side * j + i) + c];
                squared += (d - shift[c]) * (d - shift[c]);
            }
        }
    const double rms = std::sqrt(squared / (scoring_width * scoring_height));
    const auto mse = [&](const Tensor& a) {
        const auto residual = a - target;
        return (residual * residual).mean().item<float>();
    };
    const double gain = 10 * std::log10(mse(image) / mse(aligned));
    // The training grid holds the planted field to .05 px rms at best; the pull to zero, strongest at the border where
    // fewer pixels constrain the control points, costs the rest.
    EXPECT_LT(rms, .2);
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

// The alignment may correct what training can correct, not more: it fits the training grid.
TEST(CalibrationScoring, DisplacementHasTheTrainingGrid) {
    Tensor field;
    CalibrationCorrection::align_for_scoring(scoring_image(false), scoring_image(true), {}, field);
    EXPECT_EQ(field.shape().str(), TensorShape({1, CalibrationCorrection::grid, CalibrationCorrection::grid, 2}).str());
}

namespace {
    void set_uniform(CalibrationCorrection& fit, float dx, float dy) {
        std::vector<float> row(fit.field().numel());
        for (size_t i = 0; i < row.size(); i += 2) {
            row[i] = dx;
            row[i + 1] = dy;
        }
        set_field(fit, row);
    }
    Tensor checkerboard(int w, int h) {
        std::vector<float> pixels(3 * w * h);
        for (int c = 0; c < 3; ++c)
            for (int y = 0; y < h; ++y)
                for (int x = 0; x < w; ++x)
                    pixels[(c * h + y) * w + x] = (x >= w / 2) == (y >= h / 2) ? 1.f : 0.f;
        return Tensor::from_vector(pixels, {3, size_t(h), size_t(w)}, Device::CUDA);
    }
} // namespace

// Catmull-Rom alone overshoots next to a hard edge, to values no render can reach.
TEST(CalibrationCorrectionTest, CubicResamplingStaysWithinTheNearestTaps) {
    constexpr int w = 40, h = 24;
    auto photo = checkerboard(w, h);
    CalibrationCorrection fit(1);
    set_uniform(fit, .37f, .29f);
    const auto in_range = [](const Tensor& image, const char* what) {
        EXPECT_GE(image.min().item<float>(), 0.f) << what;
        EXPECT_LE(image.max().item<float>(), 1.f) << what;
    };
    in_range(warp(fit, 0, photo, photo * .5f, 1000, 1000), "photograph");

    auto alpha = Tensor::ones({size_t(h), size_t(w)}, Device::CUDA);
    auto color = Tensor::from_vector({.1f, .2f, .3f}, {3}, Device::CUDA);
    auto stable = Tensor::zeros({3, size_t(h), size_t(w)}, Device::CUDA);
    const CalibrationCorrection::Composite composite{&alpha, &color, &color, &stable};
    Tensor mask, structure, edge;
    in_range(fit.warp(0, 1000, 1000, 0, 0, photo, photo * .5f, mask, structure, edge, true, &composite), "composited");
    in_range(stable, "stable target");

    // A band of hard edges that both images share, inside a textured image with a planted shift.
    auto render = scoring_image(false), target = scoring_image(true);
    for (auto* image : {&render, &target}) {
        auto pixels = image->to_vector();
        for (int c = 0; c < 3; ++c)
            for (int y = 0; y < scoring_height; ++y)
                for (int x = 380; x < 440; ++x)
                    pixels[(c * scoring_height + y) * scoring_width + x] = x >= 410 ? .9f : .1f;
        *image = Tensor::from_vector(pixels, image->shape(), Device::CUDA);
    }
    Tensor field;
    const auto aligned = CalibrationCorrection::align_for_scoring(render, target, {}, field).to_vector();
    EXPECT_GT(field.abs().max().item<float>(), 1.f);
    for (int c = 0; c < 3; ++c)
        for (int y = 8; y < scoring_height - 8; ++y)
            for (int x = 395; x <= 425; ++x) {
                const float value = aligned[(c * scoring_height + y) * scoring_width + x];
                ASSERT_GE(value, .1f) << x << ", " << y;
                ASSERT_LE(value, .9f) << x << ", " << y;
            }
}

TEST(CalibrationCorrectionTest, TrainingFieldStaysWithinFivePercentOfTheShortSide) {
    CalibrationCorrection fit(1);
    auto photo = texture(73, 57);
    set_uniform(fit, 5.f, 0.f);
    auto render = photo * .8f;
    auto gradient = photo * 10;
    warp(fit, 0, photo, render, 1000, 1000);
    fit.accumulate(render, gradient, {}, photo.numel());
    fit.update();
    const auto moved = fit.field().to_vector();
    float reach = 0;
    for (size_t i = 0; i < moved.size(); i += 2)
        reach = std::max(reach, std::hypot(moved[i], moved[i + 1]));
    EXPECT_LE(reach, .05f * 57 * 1.0001f);
}

TEST(CalibrationCorrectionTest, ReportStartsWithTheField) {
    EXPECT_FALSE(CalibrationCorrection::report_due(500, 7500, 1000));
    EXPECT_FALSE(CalibrationCorrection::report_due(1001, 7500, 1000));
    EXPECT_TRUE(CalibrationCorrection::report_due(1000, 7500, 1000));
    EXPECT_TRUE(CalibrationCorrection::report_due(1500, 7500, 1000));
    EXPECT_TRUE(CalibrationCorrection::report_due(7500, 7500, 1000));
}

// A per-pixel background (random or image) stays where the render has it; only foreground and alpha move.
TEST(CalibrationCorrectionTest, AlphaTargetKeepsThePerPixelBackgroundInPlace) {
    constexpr int w = 24, h = 16;
    auto photo = texture(w, h);
    std::vector<float> alpha_pixels(w * h), background_pixels(3 * w * h);
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x) {
            alpha_pixels[y * w + x] = float((x / 3 + y / 2) % 3) / 2.f;
            for (int c = 0; c < 3; ++c) {
                const float v = .37f * x + .61f * y + .2f * c;
                background_pixels[(c * h + y) * w + x] = v - std::floor(v);
            }
        }
    auto alpha = Tensor::from_vector(alpha_pixels, {h, w}, Device::CUDA);
    auto background = Tensor::from_vector(background_pixels, {3, h, w}, Device::CUDA);
    auto color = Tensor::from_vector({.1f, .2f, .3f}, {3}, Device::CUDA);
    auto stable = Tensor::zeros({3, h, w}, Device::CUDA);
    auto render = Tensor::zeros({3, h, w}, Device::CUDA);
    CalibrationCorrection fit(1);
    set_uniform(fit, 2.f, 1.f);
    const CalibrationCorrection::Composite composite{&alpha, &background, &color, &stable};
    Tensor mask, structure, edge;
    const auto out = fit.warp(0, 1000, 1000, 0, 0, photo, render, mask, structure, edge, true, &composite).to_vector();
    const auto fg = photo.to_vector(), still = stable.to_vector();
    const float colors[] = {.1f, .2f, .3f};
    int checked = 0;
    for (int y = 0; y + 1 < h; ++y)
        for (int x = 0; x + 2 < w; ++x) {
            const int s = (y + 1) * w + x + 2;
            const float a = alpha_pixels[s];
            for (int c = 0; c < 3; ++c) {
                const int i = (c * h + y) * w + x;
                EXPECT_NEAR(out[i], fg[c * w * h + s] * a + background_pixels[i] * (1 - a), 1e-5) << x << ", " << y;
                EXPECT_NEAR(still[i], fg[c * w * h + s] * a + colors[c] * (1 - a), 1e-5) << x << ", " << y;
            }
            checked += a < 1;
        }
    EXPECT_GT(checked, 100);
}

TEST(CalibrationCorrectionTest, GradientsAreClearedWhereTheSampleLeavesThePhoto) {
    constexpr int w = 40, h = 30;
    auto photo = texture(w, h);
    CalibrationCorrection fit(1);
    set_uniform(fit, 6.5f, 0.f);
    warp(fit, 0, photo, photo * .5f, 1000, 1000);
    auto gradient = Tensor::ones({3, h, w}, Device::CUDA);
    fit.clear_outside(gradient);
    const auto cleared = gradient.to_vector();
    for (int c = 0; c < 3; ++c)
        for (int y = 0; y < h; ++y)
            for (int x = 0; x < w; ++x)
                EXPECT_EQ(cleared[(c * h + y) * w + x], x >= 33 ? 0.f : 1.f) << x << ", " << y;
}

// With the raw render equal to the corrected one, the decoupled loss is the plain loss: the fit must see the same
// gradient, which needs the contrast and structure part as well.
TEST(CalibrationCorrectionTest, DecoupledLossGivesTheFitThePlainLossGradient) {
    constexpr int w = 256, h = 192;
    constexpr float mix = .2f;
    auto photo = texture(w, h);
    std::vector<float> planted(162);
    for (int y = 0; y < 9; ++y)
        for (int x = 0; x < 9; ++x) {
            planted[2 * (y * 9 + x)] = .5f + .15f * std::sin(x * .45f);
            planted[2 * (y * 9 + x) + 1] = .2f * std::cos(y * .4f);
        }
    CalibrationCorrection known(1);
    set_field(known, planted);
    const auto render = warp(known, 0, photo, photo, 1000, 1000).clone();
    const float values = 3 * (w - 10) * (h - 10);
    losses::PhotometricLoss loss;

    CalibrationCorrection plain(1);
    auto target = warp(plain, 0, photo, render, 1000, 1000).clone();
    auto result = loss.forward(render, target, {.lambda_dssim = mix});
    ASSERT_TRUE(result) << result.error();
    plain.accumulate(render, result->second.grad_image, {}, values, 5);
    plain.update();
    const auto expected = plain.field().to_vector();

    CalibrationCorrection decoupled(1);
    warp(decoupled, 0, photo, render, 1000, 1000);
    auto& workspace = loss.arena().decoupled();
    auto [value, context] = kernels::decoupled_fused_l1_ssim_forward(render, render, target, mix, workspace, true);
    auto gradients = kernels::decoupled_fused_l1_ssim_backward(context, workspace);
    const auto raw = kernels::decoupled_raw_gradient(context);
    decoupled.accumulate(render, gradients.grad_corrected.squeeze(0), {}, values, 5, &raw);
    decoupled.update();
    const auto actual = decoupled.field().to_vector();

    float largest = 0, difference = 0;
    for (size_t i = 0; i < expected.size(); ++i) {
        largest = std::max(largest, std::abs(expected[i]));
        difference = std::max(difference, std::abs(actual[i] - expected[i]));
    }
    EXPECT_GT(largest, .1f);
    EXPECT_LT(difference, .01f * largest);
}

// An equirectangular camera carries a placeholder focal length; its field is in milliradians of longitude and
// latitude, and the panorama continues across the seam.
TEST(CalibrationCorrectionTest, PanoramaShiftAcrossTheSeamIsRecoveredInMilliradians) {
    constexpr int w = 256, h = 128;
    constexpr float two_pi = 6.2831853f;
    const auto lens = CalibrationCorrection::lens(CameraModelType::EQUIRECTANGULAR, 20.f, 20.f, w, h);
    EXPECT_NEAR(lens.fx, w / two_pi, 1e-3);
    EXPECT_NEAR(lens.fy, h / (two_pi / 2), 1e-3);
    EXPECT_TRUE(lens.wrap);
    const auto pano = [&](float x, float y, int c) {
        // Whole periods around the panorama, so the texture continues across the seam.
        return .5f + .28f * std::sin(two_pi * 24 * x / w + .17f * y + 2.1f * c) +
               .19f * std::cos(two_pi * 9 * x / w - .61f * y + 1.7f * c);
    };
    constexpr float longitude = 60.f, latitude = 25.f; // milliradians
    std::vector<float> photo_pixels(3 * w * h), render_pixels(3 * w * h);
    for (int c = 0; c < 3; ++c)
        for (int y = 0; y < h; ++y)
            for (int x = 0; x < w; ++x) {
                const int i = (c * h + y) * w + x;
                photo_pixels[i] = pano(float(x), float(y), c);
                render_pixels[i] = pano(x + longitude * .001f * w / two_pi, y + latitude * .001f * h / (two_pi / 2), c);
            }
    auto photo = Tensor::from_vector(photo_pixels, {3, h, w}, Device::CUDA);
    auto render = Tensor::from_vector(render_pixels, {3, h, w}, Device::CUDA);
    CalibrationCorrection fit(1);
    losses::PhotometricLoss loss;
    for (int iteration = 0; iteration < 40; ++iteration) {
        Tensor mask, structure, edge;
        const auto target = fit.warp(0, lens.fx, lens.fy, 0, 0, photo, render, mask, structure, edge, true, nullptr, lens.wrap);
        auto result = loss.forward(render, target, {.lambda_dssim = 0.f});
        ASSERT_TRUE(result) << result.error();
        fit.accumulate(render, result->second.grad_image, {}, 3 * w * h);
        fit.update();
    }
    const auto field = fit.field().to_vector();
    double squared = 0, seam = 0, lon = 0, lat = 0;
    for (int y = 0; y < 9; ++y)
        for (int x = 0; x < 9; ++x) {
            const int i = 2 * (y * 9 + x);
            squared += std::pow(field[i] - longitude, 2) + std::pow(field[i + 1] - latitude, 2);
            seam += x == 0 ? std::pow(field[i] - longitude, 2) + std::pow(field[i + 1] - latitude, 2) : 0;
            lon += (field[i] - longitude) / 81;
            lat += (field[i + 1] - latitude) / 81;
            EXPECT_NEAR(field[i], longitude, 6.f) << x << ", " << y;
            EXPECT_NEAR(field[i + 1], latitude, 6.f) << x << ", " << y;
            if (x == 0) {
                EXPECT_EQ(field[i], field[i + 16]) << "seam column, row " << y;
                EXPECT_EQ(field[i + 1], field[i + 17]) << "seam column, row " << y;
            }
        }
    const double rms = std::sqrt(squared / 81);
    RecordProperty("rms_mrad", std::to_string(rms));
    RecordProperty("seam_rms_mrad", std::to_string(std::sqrt(seam / 9)));
    RecordProperty("mean_error_mrad", std::to_string(lon) + " " + std::to_string(lat));
    EXPECT_LT(rms, 2.5);
}

// Depth and normal priors are read through the field at their own resolution; a zero field leaves them untouched.
TEST(CalibrationCorrectionTest, PriorsFollowThePhotoThroughTheField) {
    constexpr int w = 40, h = 30, pw = 20, ph = 15;
    auto photo = texture(w, h);
    std::vector<float> depth(pw * ph), normal(3 * pw * ph);
    for (int y = 0; y < ph; ++y)
        for (int x = 0; x < pw; ++x) {
            const int i = y * pw + x;
            depth[i] = (x * 7 + y * 3) % 5 == 0 ? 0.f : 1 + .1f * x + .05f * y;
            const float n[3] = {.3f + .05f * x, -.2f + .03f * y, 1.f - .01f * x};
            for (int c = 0; c < 3; ++c)
                normal[c * pw * ph + i] = (x + y) % 6 == 0 ? 0.f : n[c] * 1.1f;
        }
    auto depth_map = Tensor::from_vector(depth, {ph, pw}, Device::CUDA);
    auto normal_map = Tensor::from_vector(normal, {3, ph, pw}, Device::CUDA);

    CalibrationCorrection fit(1);
    warp(fit, 0, photo, photo, 1000, 1000);
    const auto same_depth = fit.follow(depth_map, false).to_vector(), same_normal = fit.follow(normal_map, true).to_vector();
    EXPECT_EQ(std::memcmp(same_depth.data(), depth.data(), depth.size() * sizeof(float)), 0);
    EXPECT_EQ(std::memcmp(same_normal.data(), normal.data(), normal.size() * sizeof(float)), 0);

    // 2.5 photo pixels are 1.25 prior pixels: taps x + 1 and x + 2 weigh .75 and .25.
    set_uniform(fit, 2.5f, 0.f);
    warp(fit, 0, photo, photo, 1000, 1000);
    const auto moved_depth = fit.follow(depth_map, false).to_vector(), moved_normal = fit.follow(normal_map, true).to_vector();
    for (int y = 0; y < ph; ++y)
        for (int x = 0; x < pw; ++x) {
            const int i = y * pw + x;
            if (x + 1.25f > pw - 1) {
                EXPECT_EQ(moved_depth[i], 0.f) << "outside " << x;
                for (int c = 0; c < 3; ++c)
                    EXPECT_EQ(moved_normal[c * pw * ph + i], 0.f) << "outside " << x;
                continue;
            }
            const int a = i + 1, b = i + 2;
            const float da = depth[a], db = depth[b];
            const float total = (da > 0 ? .75f : 0) + (db > 0 ? .25f : 0);
            const float expected = total > 0 ? ((da > 0 ? .75f * da : 0) + (db > 0 ? .25f * db : 0)) / total : 0;
            EXPECT_NEAR(moved_depth[i], expected, 1e-5) << x << ", " << y;

            float blend[3]{}, length = 0;
            for (int c = 0; c < 3; ++c) {
                const float na = normal[c * pw * ph + a], nb = normal[c * pw * ph + b];
                blend[c] = (normal[a] != 0 || normal[pw * ph + a] != 0 ? .75f * na : 0) +
                           (normal[b] != 0 || normal[pw * ph + b] != 0 ? .25f * nb : 0);
                length += blend[c] * blend[c];
            }
            for (int c = 0; c < 3; ++c)
                EXPECT_NEAR(moved_normal[c * pw * ph + i], length > 0 ? blend[c] / std::sqrt(length) : 0.f, 1e-5)
                    << x << ", " << y;
        }
}
