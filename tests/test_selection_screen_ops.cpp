/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

// Rectangle selection and the hover pick are tensor programs over projected
// screen positions; both run on every GPU backend and follow the CPU rules:
// invalid positions never select, the pick wants a finite position inside the
// radius, and equal distances pick the largest index.

#include "core/tensor.hpp"
#include "core/tensor_backend.hpp"
#include "rendering/selection_ops.hpp"
#include <gtest/gtest.h>

#include <cmath>
#include <limits>
#include <string>
#include <vector>

namespace {
    using namespace lfs::core;

    constexpr float kInvalid = -1.0e8f;

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

    std::vector<float> positions(const size_t count) {
        std::vector<float> values(count * 2);
        for (size_t i = 0; i < count; ++i) {
            values[i * 2] = static_cast<float>((i * 37) % 640);
            values[i * 2 + 1] = static_cast<float>((i * 53) % 480);
        }
        return values;
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
