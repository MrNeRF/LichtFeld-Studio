/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/camera.hpp"
#include "core/tensor.hpp"
#include "kernels/blob_seeding.hpp"
#include "strategies/blob_seeding.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <memory>
#include <random>
#include <stop_token>
#include <vector>

namespace {
    using lfs::core::Camera;
    using lfs::core::DataType;
    using lfs::core::Device;
    using lfs::core::Tensor;
    using Vec3 = std::array<float, 3>;

    constexpr int WIDTH = 160;
    constexpr int HEIGHT = 120;
    constexpr float FOCAL = 140.0f;
    constexpr float BACKGROUND = 0.5f;

    Tensor grey_image() {
        return Tensor::full({3, HEIGHT, WIDTH}, BACKGROUND, Device::CPU);
    }

    void draw_square(Tensor& image, const int x, const int y, const float value) {
        float* data = image.ptr<float>();
        for (int c = 0; c < 3; ++c)
            for (int dy = -1; dy <= 1; ++dy)
                for (int dx = -1; dx <= 1; ++dx)
                    if (x + dx >= 0 && x + dx < WIDTH && y + dy >= 0 && y + dy < HEIGHT)
                        data[(c * HEIGHT + y + dy) * WIDTH + x + dx] = value;
    }

    Vec3 normalized(const Vec3& v) {
        const float n = std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
        return {v[0] / n, v[1] / n, v[2] / n};
    }

    Vec3 cross(const Vec3& a, const Vec3& b) {
        return {a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]};
    }

    struct RingView {
        std::shared_ptr<Camera> camera;
        std::array<float, 9> R;
        Vec3 t;
    };

    // Looks at the origin from a ring of radius 6; rows of R are the camera axes in world space.
    RingView ring_view(const int index, const int count, Tensor radial = Tensor()) {
        const float angle = 2.0f * static_cast<float>(M_PI) * static_cast<float>(index) / static_cast<float>(count);
        const Vec3 center{6.0f * std::cos(angle), 0.8f, 6.0f * std::sin(angle)};
        const Vec3 z = normalized({-center[0], -center[1], -center[2]});
        const Vec3 x = normalized(cross({0.0f, -1.0f, 0.0f}, z));
        const Vec3 y = cross(z, x);
        RingView view;
        view.R = {x[0], x[1], x[2], y[0], y[1], y[2], z[0], z[1], z[2]};
        for (int r = 0; r < 3; ++r)
            view.t[r] = -(view.R[r * 3] * center[0] + view.R[r * 3 + 1] * center[1] + view.R[r * 3 + 2] * center[2]);
        auto R = Tensor::from_vector(std::vector<float>(view.R.begin(), view.R.end()), {3, 3}, Device::CPU);
        auto T = Tensor::from_vector(std::vector<float>(view.t.begin(), view.t.end()), {3}, Device::CPU);
        view.camera = std::make_shared<Camera>(R, T, FOCAL, FOCAL, WIDTH / 2.0f, HEIGHT / 2.0f, radial, Tensor(), lfs::core::CameraModelType::PINHOLE,
                                               "view.png", std::filesystem::path{}, std::filesystem::path{},
                                               WIDTH, HEIGHT, index);
        view.camera->set_image_dimensions(WIDTH, HEIGHT);
        return view;
    }

    bool project(const RingView& view, const Vec3& p, int& x, int& y) {
        float c[3];
        for (int r = 0; r < 3; ++r)
            c[r] = view.R[r * 3] * p[0] + view.R[r * 3 + 1] * p[1] + view.R[r * 3 + 2] * p[2] + view.t[r];
        if (c[2] <= 0.1f)
            return false;
        x = static_cast<int>(std::floor(FOCAL * c[0] / c[2] + WIDTH / 2.0f));
        y = static_cast<int>(std::floor(FOCAL * c[1] / c[2] + HEIGHT / 2.0f));
        return x >= 0 && x < WIDTH && y >= 0 && y < HEIGHT;
    }

    Tensor shell_points(const std::vector<Vec3>& extra) {
        std::mt19937 rng(7);
        std::normal_distribution<float> gauss(0.0f, 1.0f);
        std::uniform_real_distribution<float> radius(2.0f, 2.5f);
        std::vector<float> points;
        for (int i = 0; i < 600; ++i) {
            const Vec3 d = normalized({gauss(rng), gauss(rng), gauss(rng)});
            const float r = radius(rng);
            points.insert(points.end(), {d[0] * r, d[1] * r, d[2] * r});
        }
        for (const auto& p : extra)
            points.insert(points.end(), p.begin(), p.end());
        return Tensor::from_vector(points, {points.size() / 3, 3}, Device::CPU);
    }

    float distance(const float* a, const Vec3& b) {
        return std::sqrt((a[0] - b[0]) * (a[0] - b[0]) + (a[1] - b[1]) * (a[1] - b[1]) + (a[2] - b[2]) * (a[2] - b[2]));
    }
} // namespace

TEST(BlobSeeding, DetectsBrightAndDarkBlobsWithDilatedBitmap) {
    namespace bs = lfs::training::kernels::blob_seeding;
    auto image = grey_image();
    draw_square(image, 40, 30, 1.0f);
    draw_square(image, 100, 70, 0.0f);
    bs::DetectionWorkspace workspace;
    const auto peaks = bs::detect_peaks(image.cuda(), 3, workspace);

    const auto rows = peaks.peaks.cpu();
    ASSERT_EQ(rows.ndim(), 2u);
    ASSERT_EQ(rows.shape()[0], 2u);
    std::array<bool, 2> found{};
    for (size_t i = 0; i < 2; ++i) {
        const float* row = rows.ptr<float>() + i * bs::PeakFieldCount;
        EXPECT_EQ(row[bs::PeakView], 3.0f);
        const int polarity = static_cast<int>(row[bs::PeakPolarity]);
        found[polarity] = true;
        EXPECT_EQ(row[bs::PeakX], polarity == 0 ? 40.0f : 100.0f);
        EXPECT_EQ(row[bs::PeakY], polarity == 0 ? 30.0f : 70.0f);
        EXPECT_EQ(row[bs::PeakR], polarity == 0 ? 1.0f : 0.0f);
    }
    EXPECT_TRUE(found[0] && found[1]);

    const auto bitmap = peaks.bitmap.cpu();
    ASSERT_EQ(bitmap.numel(), static_cast<size_t>(WIDTH * HEIGHT + 15) / 16);
    const auto bit = [&](const int x, const int y, const int polarity) {
        const size_t i = static_cast<size_t>(y) * WIDTH + x;
        return (bitmap.ptr<uint32_t>()[i / 16] >> (2 * (i % 16) + polarity)) & 1u;
    };
    EXPECT_EQ(bit(41, 31, 0), 1u);
    EXPECT_EQ(bit(42, 30, 0), 0u);
    EXPECT_EQ(bit(99, 69, 1), 1u);
    EXPECT_EQ(bit(99, 69, 0), 0u);
    EXPECT_FLOAT_EQ(peaks.density[0], 9.0f / (WIDTH * HEIGHT));
    EXPECT_FLOAT_EQ(peaks.density[1], 9.0f / (WIDTH * HEIGHT));
}

// Fails if the sweep or the back-projection mixes up world and camera frames (seeds miss the blobs),
// if the reference filter is skipped (the explained blob gets a seed) or if detections are not merged
// across views (one seed per view instead of one per blob). A cancelled triangulation must return no seeds.
TEST(BlobSeeding, SeedsTriangulatedBlobsThatReferencePointsMiss) {
    const std::vector<Vec3> bright{{0.3f, 0.2f, -0.4f}, {-0.5f, -0.1f, 0.6f}, {0.7f, 0.4f, 0.2f}};
    const Vec3 dark{-0.2f, -0.3f, -0.7f};
    const Vec3 explained{0.1f, -0.5f, 0.1f};
    lfs::training::BlobSeeder seeder(shell_points({explained}));

    constexpr int VIEWS = 24;
    for (int i = 0; i < VIEWS; ++i) {
        const auto view = ring_view(i, VIEWS);
        auto image = grey_image();
        int x = 0, y = 0;
        for (const auto& p : bright)
            if (project(view, p, x, y))
                draw_square(image, x, y, 1.0f);
        if (project(view, explained, x, y))
            draw_square(image, x, y, 1.0f);
        if (project(view, dark, x, y))
            draw_square(image, x, y, 0.0f);
        seeder.capture(*view.camera, image.cuda());
        seeder.capture(*view.camera, image.cuda());
    }
    ASSERT_EQ(seeder.captured_views(), static_cast<size_t>(VIEWS));

    const auto seeds = seeder.triangulate();
    std::vector<Vec3> expected = bright;
    expected.push_back(dark);
    ASSERT_EQ(seeds.size(), expected.size());
    for (size_t i = 0; i < seeds.size(); ++i) {
        const float* m = seeds.means.data() + i * 3;
        EXPECT_GT(distance(m, explained), 0.3f);
        size_t nearest = 0;
        for (size_t e = 1; e < expected.size(); ++e)
            if (distance(m, expected[e]) < distance(m, expected[nearest]))
                nearest = e;
        EXPECT_LT(distance(m, expected[nearest]), 0.1f) << "seed " << i;
        const float colour = nearest < bright.size() ? 1.0f : 0.0f;
        EXPECT_FLOAT_EQ(seeds.colors[i * 3], colour);
        EXPECT_NEAR(std::exp(seeds.log_scales[i]), 1.5f * 6.0f / FOCAL, 0.03f);
        expected[nearest] = {1e3f, 1e3f, 1e3f};
    }

    std::stop_source cancelled;
    cancelled.request_stop();
    EXPECT_EQ(seeder.triangulate(cancelled.get_token()).size(), 0u);
}

// Fails if detection on block-averaged images keeps the full-resolution intrinsics (seeds miss the blobs and get
// the full-resolution footprint) or if uint8 targets are not normalised before detection.
TEST(BlobSeeding, CappedDetectionResolutionKeepsSeedsOnTheBlobs) {
    const std::vector<Vec3> bright{{0.3f, 0.2f, -0.4f}, {-0.5f, -0.1f, 0.6f}, {0.7f, 0.4f, 0.2f}};
    lfs::training::BlobSeeder seeder(shell_points({}), WIDTH * HEIGHT / 4);

    constexpr int VIEWS = 24;
    for (int i = 0; i < VIEWS; ++i) {
        const auto view = ring_view(i, VIEWS);
        auto image = grey_image();
        int x = 0, y = 0;
        for (const auto& p : bright)
            if (project(view, p, x, y))
                draw_square(image, x, y, 1.0f);
        seeder.capture(*view.camera, (image * 255.0f).to(DataType::UInt8).cuda());
    }
    ASSERT_EQ(seeder.captured_views(), static_cast<size_t>(VIEWS));

    // One detection pixel spans more depth here than the merge radius, so a blob may get several seeds.
    const auto seeds = seeder.triangulate();
    ASSERT_GE(seeds.size(), bright.size());
    std::vector<bool> covered(bright.size(), false);
    for (size_t i = 0; i < seeds.size(); ++i) {
        const float* m = seeds.means.data() + i * 3;
        size_t nearest = 0;
        for (size_t b = 1; b < bright.size(); ++b)
            if (distance(m, bright[b]) < distance(m, bright[nearest]))
                nearest = b;
        EXPECT_LT(distance(m, bright[nearest]), 0.1f) << "seed " << i;
        covered[nearest] = true;
        EXPECT_NEAR(std::exp(seeds.log_scales[i]), 2.0f * 1.5f * 6.0f / FOCAL, 0.03f);
    }
    EXPECT_TRUE(std::ranges::all_of(covered, [](const bool c) { return c; }));
}

TEST(BlobSeeding, IgnoresViewsWithoutPinholeImages) {
    lfs::training::BlobSeeder seeder(shell_points({}));
    const auto distorted = ring_view(0, 8, Tensor::from_vector(std::vector<float>{0.1f}, {1}, Device::CPU));
    auto image = grey_image();
    draw_square(image, 40, 30, 1.0f);
    seeder.capture(*distorted.camera, image.cuda());
    EXPECT_EQ(seeder.captured_views(), 0u);
}

// Fails if a view without any contrast peak throws instead of contributing an empty peak list.
TEST(BlobSeeding, ViewWithoutPeaksYieldsEmptyPeakList) {
    namespace bs = lfs::training::kernels::blob_seeding;
    bs::DetectionWorkspace workspace;
    const auto peaks = bs::detect_peaks(grey_image().cuda(), 0, workspace);
    EXPECT_EQ(peaks.peaks.numel(), 0u);
    EXPECT_FLOAT_EQ(peaks.density[0], 0.0f);
    EXPECT_FLOAT_EQ(peaks.density[1], 0.0f);
}
