/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

// Rectangle, brush, polygon, and hover pick are tensor programs over projected
// screen positions; they run on every GPU backend and follow the CPU rules:
// invalid positions never select, the pick wants a finite position inside the
// radius, and equal distances pick the largest index. Brush inclusion is the
// CUDA disk test dx^2+dy^2 <= r^2. Polygon inclusion is even-odd.

#include "core/tensor.hpp"
#include "core/tensor_backend.hpp"
#include "rendering/selection_ops.hpp"
#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <limits>
#include <string>
#include <vector>

namespace {
    using namespace lfs::core;

    constexpr float kInvalid = -1.0e8f;
    constexpr float kInvalidThreshold = -1000.0f;

    std::vector<GpuBackend> backends_under_test() {
        std::vector<GpuBackend> backends{GpuBackend::CUDA};
        if (gpu_backend_available(GpuBackend::Vulkan)) {
            backends.push_back(GpuBackend::Vulkan);
        }
        return backends;
    }

    std::string label(const GpuBackend backend) {
        return backend == GpuBackend::CUDA ? "cuda" : "vulkan";
    }

    Tensor upload(const std::vector<float>& values, const TensorShape& shape, const GpuBackend backend) {
        GpuBackendScope scope(backend);
        return Tensor::from_vector(values, shape, Device::CPU).to(Device::GPU);
    }

    Tensor uploadBool(const std::vector<bool>& values, const GpuBackend backend) {
        GpuBackendScope scope(backend);
        return Tensor::from_vector(values, TensorShape{values.size()}, Device::CPU).to(Device::GPU);
    }

    std::vector<float> positions(const size_t count) {
        std::vector<float> values(count * 2);
        for (size_t i = 0; i < count; ++i) {
            values[i * 2] = static_cast<float>((i * 37) % 640);
            values[i * 2 + 1] = static_cast<float>((i * 53) % 480);
        }
        return values;
    }

    bool cpuDisk(const float x, const float y, const float mx, const float my, const float radius) {
        if (!(x >= kInvalidThreshold) || !(y >= kInvalidThreshold)) {
            return false;
        }
        const float dx = x - mx;
        const float dy = y - my;
        return dx * dx + dy * dy <= radius * radius;
    }

    bool cpuPointInPolygon(const float px, const float py, const std::vector<float>& poly) {
        const int n = static_cast<int>(poly.size() / 2);
        bool inside = false;
        for (int i = 0, j = n - 1; i < n; j = i++) {
            const float yi = poly[static_cast<size_t>(i) * 2 + 1];
            const float yj = poly[static_cast<size_t>(j) * 2 + 1];
            if ((yi > py) != (yj > py)) {
                const float xi = poly[static_cast<size_t>(i) * 2];
                const float xj = poly[static_cast<size_t>(j) * 2];
                if (px < (xj - xi) * (py - yi) / (yj - yi) + xi) {
                    inside = !inside;
                }
            }
        }
        return inside;
    }
} // namespace

TEST(SelectionScreenOps, RectangleSelectionAccumulatesAndSkipsInvalidPositions) {
    constexpr size_t n = 2003;
    auto values = positions(n);
    values[10 * 2] = kInvalid;
    values[11 * 2 + 1] = kInvalid;
    const float x0 = 100.0f, y0 = 50.0f, x1 = 300.0f, y1 = 200.0f;
    for (const GpuBackend backend : backends_under_test()) {
        SCOPED_TRACE(label(backend));
        const Tensor screen = upload(values, TensorShape{n, 2}, backend);
        std::vector<bool> seed(n, false);
        seed[5] = true;
        Tensor selection = [&] {
            GpuBackendScope scope(backend);
            return Tensor::from_vector(seed, TensorShape{n}, Device::CPU).to(Device::GPU);
        }();
        lfs::rendering::rect_select_tensor(screen, x0, y0, x1, y1, selection);
        const auto got = selection.to_vector_bool();
        for (size_t i = 0; i < n; ++i) {
            const float x = values[i * 2];
            const float y = values[i * 2 + 1];
            const bool valid = x >= -1000.0f && y >= -1000.0f;
            const bool expected = seed[i] || (valid && x >= x0 && x <= x1 && y >= y0 && y <= y1);
            EXPECT_EQ(got[i], expected) << "index=" << i;
        }
    }
}

TEST(SelectionScreenOps, PickFindsTheNearestValidPositionAndBreaksTiesHigh) {
    constexpr size_t n = 1501;
    auto values = positions(n);
    values[700 * 2] = 320.5f;
    values[700 * 2 + 1] = 240.5f;
    values[900 * 2] = 320.5f;
    values[900 * 2 + 1] = 240.5f;
    values[1200 * 2] = 320.6f;
    values[1200 * 2 + 1] = 240.5f;
    values[42 * 2] = std::numeric_limits<float>::quiet_NaN();
    values[43 * 2] = kInvalid;
    for (const GpuBackend backend : backends_under_test()) {
        SCOPED_TRACE(label(backend));
        const Tensor screen = upload(values, TensorShape{n, 2}, backend);
        EXPECT_EQ(lfs::rendering::pick_projected_gaussian_tensor(screen, 320.5f, 240.5f, 4.0f), 900);
        EXPECT_EQ(lfs::rendering::pick_projected_gaussian_tensor(screen, 320.58f, 240.5f, 4.0f), 1200);
        EXPECT_EQ(lfs::rendering::pick_projected_gaussian_tensor(screen, -5000.0f, -5000.0f, 4.0f), -1);
        const Tensor none = upload(std::vector<float>(8, kInvalid), TensorShape{4, 2}, backend);
        EXPECT_EQ(lfs::rendering::pick_projected_gaussian_tensor(none, 0.0f, 0.0f, 1.0e9f), -1);
    }
}

TEST(SelectionScreenOps, BrushMatchesCpuDiskAndSkipsInvalid) {
    constexpr size_t n = 2003;
    auto values = positions(n);
    values[10 * 2] = kInvalid;
    values[11 * 2 + 1] = kInvalid;
    values[12 * 2] = std::numeric_limits<float>::quiet_NaN();
    values[13 * 2] = 200.0f;
    values[13 * 2 + 1] = 150.0f;
    values[14 * 2] = std::numeric_limits<float>::infinity();
    values[14 * 2 + 1] = 150.0f;
    values[15 * 2] = 200.0f;
    values[15 * 2 + 1] = -std::numeric_limits<float>::infinity();
    constexpr float mx = 200.0f;
    constexpr float my = 150.0f;
    constexpr float radius = 12.5f;
    for (const GpuBackend backend : backends_under_test()) {
        SCOPED_TRACE(label(backend));
        const Tensor screen = upload(values, TensorShape{n, 2}, backend);
        std::vector<bool> seed(n, false);
        seed[5] = true;
        Tensor selection = uploadBool(seed, backend);
        lfs::rendering::brush_select_tensor(screen, mx, my, radius, selection);
        const auto got = selection.to_vector_bool();
        size_t selected = 0;
        for (size_t i = 0; i < n; ++i) {
            const bool expected = seed[i] || cpuDisk(values[i * 2], values[i * 2 + 1], mx, my, radius);
            EXPECT_EQ(got[i], expected) << "index=" << i;
            selected += got[i] ? 1 : 0;
        }
        EXPECT_GT(selected, 1u);
        EXPECT_TRUE(got[13]);
        EXPECT_FALSE(got[10]);
        EXPECT_FALSE(got[12]);
        EXPECT_FALSE(got[14]);
        EXPECT_FALSE(got[15]);
    }
}

TEST(SelectionScreenOps, BrushProgramMatchesCudaKernel) {
    if (!gpu_backend_available(GpuBackend::CUDA)) {
        GTEST_SKIP() << "CUDA backend required for kernel vs program parity";
    }
    constexpr size_t n = 1024;
    auto values = positions(n);
    values[3 * 2] = kInvalid;
    values[4 * 2] = 40.0f;
    values[4 * 2 + 1] = 40.0f;
    constexpr float mx = 40.0f;
    constexpr float my = 40.0f;
    constexpr float radius = 8.0f;
    const Tensor screen = upload(values, TensorShape{n, 2}, GpuBackend::CUDA);
    Tensor kernel_sel = uploadBool(std::vector<bool>(n, false), GpuBackend::CUDA);
    Tensor program_sel = uploadBool(std::vector<bool>(n, false), GpuBackend::CUDA);
    lfs::rendering::brush_select_tensor(screen, mx, my, radius, kernel_sel);
    lfs::rendering::brush_select_tensor_program(screen, mx, my, radius, program_sel);
    EXPECT_EQ(kernel_sel.to_vector_bool(), program_sel.to_vector_bool());
}

TEST(SelectionScreenOps, BrushStrokeUnionIsBoundedOrNotMatrix) {
    constexpr size_t n = 64;
    std::vector<float> values(n * 2, 0.0f);
    values[0] = 10.0f;
    values[1] = 10.0f;
    values[2] = 18.0f;
    values[3] = 10.0f;
    values[4] = 50.0f;
    values[5] = 50.0f;
    values[6] = kInvalid;
    values[7] = 10.0f;
    const std::vector<float> disks{10.0f, 10.0f, 18.0f, 10.0f};
    constexpr float radius = 3.0f;
    for (const GpuBackend backend : backends_under_test()) {
        SCOPED_TRACE(label(backend));
        const Tensor screen = upload(values, TensorShape{n, 2}, backend);
        Tensor selection = uploadBool(std::vector<bool>(n, false), backend);
        lfs::rendering::brush_select_disks_tensor(screen, disks, radius, selection);
        const auto got = selection.to_vector_bool();
        EXPECT_TRUE(got[0]);
        EXPECT_TRUE(got[1]);
        EXPECT_FALSE(got[2]);
        EXPECT_FALSE(got[3]);
    }
}

TEST(SelectionScreenOps, BrushEmptyAndZeroRadius) {
    for (const GpuBackend backend : backends_under_test()) {
        SCOPED_TRACE(label(backend));
        Tensor empty_sel;
        Tensor empty_screen;
        lfs::rendering::brush_select_tensor(empty_screen, 0.0f, 0.0f, 4.0f, empty_sel);

        std::vector<float> values{5.0f, 5.0f, 6.0f, 5.0f};
        const Tensor screen = upload(values, TensorShape{2, 2}, backend);
        Tensor selection = uploadBool(std::vector<bool>{false, false}, backend);
        lfs::rendering::brush_select_tensor(screen, 5.0f, 5.0f, 0.0f, selection);
        const auto got = selection.to_vector_bool();
        EXPECT_TRUE(got[0]);
        EXPECT_FALSE(got[1]);
    }
}

TEST(SelectionScreenOps, PolygonEvenOddMatchesCpuAndSkipsInvalid) {
    constexpr size_t n = 32;
    std::vector<float> values(n * 2, 0.0f);
    values[0] = 5.0f;
    values[1] = 5.0f;
    values[2] = 50.0f;
    values[3] = 50.0f;
    values[4] = kInvalid;
    values[5] = 5.0f;
    values[6] = 1.0f;
    values[7] = 1.0f;
    values[8] = 9.0f;
    values[9] = 9.0f;
    const std::vector<float> square{0.0f, 0.0f, 10.0f, 0.0f, 10.0f, 10.0f, 0.0f, 10.0f};
    for (const GpuBackend backend : backends_under_test()) {
        SCOPED_TRACE(label(backend));
        const Tensor screen = upload(values, TensorShape{n, 2}, backend);
        const Tensor polygon = upload(square, TensorShape{4, 2}, backend);
        std::vector<bool> seed(n, false);
        seed[2] = true;
        Tensor selection = uploadBool(seed, backend);
        lfs::rendering::polygon_select_tensor(screen, polygon, selection);
        const auto got = selection.to_vector_bool();
        for (size_t i = 0; i < n; ++i) {
            const float x = values[i * 2];
            const float y = values[i * 2 + 1];
            const bool valid = x >= kInvalidThreshold && y >= kInvalidThreshold;
            const bool expected = seed[i] || (valid && cpuPointInPolygon(x, y, square));
            EXPECT_EQ(got[i], expected) << "index=" << i;
        }
        EXPECT_TRUE(got[0]);
        EXPECT_FALSE(got[1]);
        EXPECT_TRUE(got[2]);
        Tensor short_poly_sel = uploadBool(std::vector<bool>(n, false), backend);
        const Tensor line = upload({0.0f, 0.0f, 10.0f, 0.0f}, TensorShape{2, 2}, backend);
        lfs::rendering::polygon_select_tensor(screen, line, short_poly_sel);
        EXPECT_EQ(short_poly_sel.to_vector_bool(), std::vector<bool>(n, false));
    }
}

TEST(SelectionScreenOps, PolygonProgramMatchesCudaKernel) {
    if (!gpu_backend_available(GpuBackend::CUDA)) {
        GTEST_SKIP() << "CUDA backend required for kernel vs program parity";
    }
    constexpr size_t n = 64;
    auto values = positions(n);
    const std::vector<float> bowtie{
        0.0f,
        0.0f,
        40.0f,
        40.0f,
        40.0f,
        0.0f,
        0.0f,
        40.0f,
    };
    const Tensor screen = upload(values, TensorShape{n, 2}, GpuBackend::CUDA);
    const Tensor polygon = upload(bowtie, TensorShape{4, 2}, GpuBackend::CUDA);
    Tensor kernel_sel = uploadBool(std::vector<bool>(n, false), GpuBackend::CUDA);
    Tensor program_sel = uploadBool(std::vector<bool>(n, false), GpuBackend::CUDA);
    lfs::rendering::polygon_select_tensor(screen, polygon, kernel_sel);
    lfs::rendering::polygon_select_tensor_program(screen, polygon, program_sel);
    EXPECT_EQ(kernel_sel.to_vector_bool(), program_sel.to_vector_bool());
}

TEST(SelectionScreenOps, ProjectPinholeMarksBehindCameraInvalid) {
    constexpr std::array<float, 9> rotation{1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f};
    constexpr std::array<float, 3> translation{0.0f, 0.0f, 0.0f};
    const std::vector<float> means{
        0.0f,
        0.0f,
        -2.0f,
        0.0f,
        0.0f,
        2.0f,
        0.0f,
        0.0f,
        0.0f,
    };
    for (const GpuBackend backend : backends_under_test()) {
        SCOPED_TRACE(label(backend));
        const Tensor gpu_means = upload(means, TensorShape{3, 3}, backend);
        const auto projected = lfs::rendering::project_screen_positions_tensor(
            gpu_means, 640, 480, rotation, translation, 500.0f, 500.0f,
            320.0f, 240.0f, lfs::rendering::ScreenWindowCameraModel::Pinhole, 1.0f,
            nullptr, nullptr, {});
        ASSERT_TRUE(projected.is_valid());
        const auto xy = projected.cpu().to_vector();
        ASSERT_EQ(xy.size(), 6u);
        EXPECT_GE(xy[0], kInvalidThreshold);
        EXPECT_GE(xy[1], kInvalidThreshold);
        EXPECT_LT(xy[2], kInvalidThreshold);
        EXPECT_LT(xy[3], kInvalidThreshold);
        EXPECT_LT(xy[4], kInvalidThreshold);
        EXPECT_LT(xy[5], kInvalidThreshold);
    }
}
