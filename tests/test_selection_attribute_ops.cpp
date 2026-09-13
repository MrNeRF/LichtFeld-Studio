/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

// The attribute selections (opacity, scale, color) are tensor programs, so
// they run on every GPU backend; each is checked against a CPU reference away
// from the threshold boundary.

#include "core/cuda/selection_ops.hpp"
#include "core/tensor.hpp"
#include "core/tensor_backend.hpp"
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

namespace {
    using namespace lfs::core;

    constexpr uint8_t kGroup = 7;
    constexpr float kBoundaryBand = 1.0e-4f;

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

    float pattern(const size_t i, const float low, const float high) {
        const float t = static_cast<float>((i * 7919) % 10007) / 10007.0f;
        return low + t * (high - low);
    }
} // namespace

TEST(SelectionAttributeOps, OpacityThresholdMatchesTheSigmoidReference) {
    constexpr size_t n = 5003;
    constexpr float lo = 0.2f;
    constexpr float hi = 0.7f;
    std::vector<float> raw(n);
    for (size_t i = 0; i < n; ++i)
        raw[i] = pattern(i, -8.0f, 8.0f);
    for (const GpuBackend backend : backends_under_test()) {
        SCOPED_TRACE(label(backend));
        for (const TensorShape& shape : {TensorShape{n}, TensorShape{n, 1}}) {
            const auto mask = cuda::select_by_opacity(upload(raw, shape, backend), lo, hi, kGroup).to_vector_uint8();
            ASSERT_EQ(mask.size(), n);
            size_t checked = 0;
            for (size_t i = 0; i < n; ++i) {
                const float activated = 1.0f / (1.0f + std::exp(-raw[i]));
                if (std::abs(activated - lo) < kBoundaryBand || std::abs(activated - hi) < kBoundaryBand)
                    continue;
                const uint8_t expected = activated >= lo && activated <= hi ? kGroup : 0;
                EXPECT_EQ(mask[i], expected) << "index=" << i;
                ++checked;
            }
            EXPECT_GT(checked, n * 9 / 10);
        }
    }
}

TEST(SelectionAttributeOps, ScaleThresholdUsesTheLargestAxis) {
    constexpr size_t n = 4001;
    constexpr float max_scale = 1.5f;
    std::vector<float> raw(n * 3);
    for (size_t i = 0; i < raw.size(); ++i)
        raw[i] = pattern(i, -3.0f, 1.5f);
    for (const GpuBackend backend : backends_under_test()) {
        SCOPED_TRACE(label(backend));
        const auto mask = cuda::select_by_scale(upload(raw, TensorShape{n, 3}, backend), max_scale, kGroup).to_vector_uint8();
        ASSERT_EQ(mask.size(), n);
        size_t checked = 0;
        for (size_t i = 0; i < n; ++i) {
            const float largest = std::max({std::exp(raw[i * 3]), std::exp(raw[i * 3 + 1]), std::exp(raw[i * 3 + 2])});
            if (std::abs(largest - max_scale) < kBoundaryBand)
                continue;
            EXPECT_EQ(mask[i], largest <= max_scale ? kGroup : 0) << "index=" << i;
            ++checked;
        }
        EXPECT_GT(checked, n * 9 / 10);
    }
}

TEST(SelectionAttributeOps, ColorThresholdMatchesTheDecodedReferenceOnBothLayouts) {
    constexpr size_t n = 3001;
    constexpr float sh_c0 = 0.28209479177387814f;
    constexpr float threshold = 0.15f;
    const float ref[3] = {0.55f, 0.35f, 0.6f};
    std::vector<float> sh0(n * 3);
    for (size_t i = 0; i < sh0.size(); ++i)
        sh0[i] = pattern(i, -2.5f, 2.5f);
    for (const GpuBackend backend : backends_under_test()) {
        SCOPED_TRACE(label(backend));
        for (const TensorShape& shape : {TensorShape{n, 3}, TensorShape{n, 1, 3}}) {
            const auto mask = cuda::select_by_color(upload(sh0, shape, backend), ref[0], ref[1], ref[2], threshold, kGroup).to_vector_uint8();
            ASSERT_EQ(mask.size(), n);
            size_t checked = 0;
            for (size_t i = 0; i < n; ++i) {
                bool near_boundary = false;
                bool match = true;
                for (size_t c = 0; c < 3; ++c) {
                    const float decoded = std::clamp(0.5f + sh0[i * 3 + c] * sh_c0, 0.0f, 1.0f);
                    const float distance = std::abs(decoded - ref[c]);
                    near_boundary = near_boundary || std::abs(distance - threshold) < kBoundaryBand;
                    match = match && distance <= threshold;
                }
                if (near_boundary)
                    continue;
                EXPECT_EQ(mask[i], match ? kGroup : 0) << "index=" << i;
                ++checked;
            }
            EXPECT_GT(checked, n * 9 / 10);
        }
    }
}

TEST(SelectionAttributeOps, EmptyInputsProduceEmptyMasks) {
    const Tensor none = Tensor::empty({0}, Device::GPU, DataType::Float32);
    EXPECT_EQ(cuda::select_by_opacity(none, 0.0f, 1.0f, kGroup).numel(), 0u);
    EXPECT_EQ(cuda::select_by_scale(Tensor::empty({0, 3}, Device::GPU, DataType::Float32), 1.0f, kGroup).numel(), 0u);
    EXPECT_EQ(cuda::select_by_color(Tensor::empty({0, 3}, Device::GPU, DataType::Float32), 0.0f, 0.0f, 0.0f, 0.1f, kGroup).numel(), 0u);
}
