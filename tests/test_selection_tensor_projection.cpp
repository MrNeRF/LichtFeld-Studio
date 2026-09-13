/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

// Screen-position projection as a tensor program. CPU reference copies
// projectScreenPositionsKernel: pinhole/ortho reject view_z >= -1e-6, equirect
// does not, hidden nodes and non-finite views write the invalid marker.

#include "core/tensor.hpp"
#include "core/tensor_backend.hpp"
#include "rendering/selection_ops.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <gtest/gtest.h>
#include <limits>
#include <string>
#include <vector>

namespace {
    using namespace lfs::core;
    using lfs::rendering::ScreenWindowCameraModel;

    constexpr float kInvalidFill = -1.0e8f;
    constexpr float kInvalidThreshold = -1000.0f;
    constexpr float kPi = 3.14159265358979323846f;

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

    Tensor uploadI32(const std::vector<int>& values, const GpuBackend backend) {
        GpuBackendScope scope(backend);
        return Tensor::from_vector(values, TensorShape{values.size()}, Device::CPU).to(Device::GPU);
    }

    bool isInvalid(const float x, const float y) {
        return x < kInvalidThreshold || y < kInvalidThreshold;
    }

    void cpuProject(const std::vector<float>& means,
                    const int width,
                    const int height,
                    const std::array<float, 9>& rotation,
                    const std::array<float, 3>& translation,
                    const float fx,
                    const float fy,
                    const float cx,
                    const float cy,
                    const ScreenWindowCameraModel model,
                    const float ortho_scale,
                    const std::vector<float>* const transforms,
                    const std::vector<int>* const indices,
                    const std::vector<bool>& visibility,
                    std::vector<float>& out_xy) {
        const size_t n = means.size() / 3;
        out_xy.assign(n * 2, kInvalidFill);
        const int transform_count = transforms ? static_cast<int>(transforms->size() / 16) : 0;
        for (size_t i = 0; i < n; ++i) {
            int transform_idx = indices ? (*indices)[i] : 0;
            if (!visibility.empty() && indices) {
                if (transform_idx < 0 ||
                    transform_idx >= static_cast<int>(visibility.size()) ||
                    !visibility[static_cast<size_t>(transform_idx)]) {
                    continue;
                }
            }
            float px = means[i * 3];
            float py = means[i * 3 + 1];
            float pz = means[i * 3 + 2];
            if (transforms && transform_count > 0) {
                transform_idx = std::clamp(transform_idx, 0, transform_count - 1);
                const float* const m = transforms->data() + static_cast<size_t>(transform_idx) * 16;
                const float ox = m[0] * px + m[1] * py + m[2] * pz + m[3];
                const float oy = m[4] * px + m[5] * py + m[6] * pz + m[7];
                const float oz = m[8] * px + m[9] * py + m[10] * pz + m[11];
                px = ox;
                py = oy;
                pz = oz;
            }
            const float dx = px - translation[0];
            const float dy = py - translation[1];
            const float dz = pz - translation[2];
            const float view_x = rotation[0] * dx + rotation[1] * dy + rotation[2] * dz;
            const float view_y = rotation[3] * dx + rotation[4] * dy + rotation[5] * dz;
            const float view_z = rotation[6] * dx + rotation[7] * dy + rotation[8] * dz;
            if (!std::isfinite(view_x) || !std::isfinite(view_y) || !std::isfinite(view_z)) {
                continue;
            }
            const bool equirect = model == ScreenWindowCameraModel::Equirectangular;
            if (!equirect && view_z >= -1.0e-6f) {
                continue;
            }
            if (equirect) {
                const float eq_x = view_x;
                const float eq_y = -view_y;
                const float eq_z = -view_z;
                const float len = std::sqrt(eq_x * eq_x + eq_y * eq_y + eq_z * eq_z);
                if (len <= 1.0e-6f || !std::isfinite(len)) {
                    continue;
                }
                const float dir_x = eq_x / len;
                const float dir_y = eq_y / len;
                const float dir_z = eq_z / len;
                out_xy[i * 2] = (std::atan2(dir_x, dir_z) / (2.0f * kPi) + 0.5f) * static_cast<float>(width);
                out_xy[i * 2 + 1] =
                    (std::asin(std::clamp(dir_y, -1.0f, 1.0f)) / kPi + 0.5f) * static_cast<float>(height);
                continue;
            }
            if (model == ScreenWindowCameraModel::Orthographic) {
                if (!std::isfinite(ortho_scale) || ortho_scale <= 0.0f) {
                    continue;
                }
                out_xy[i * 2] = cx + view_x * ortho_scale;
                out_xy[i * 2 + 1] = cy - view_y * ortho_scale;
                continue;
            }
            const float depth = -view_z;
            out_xy[i * 2] = cx + view_x * fx / depth;
            out_xy[i * 2 + 1] = cy - view_y * fy / depth;
        }
    }

    void expectProjectedNear(const std::vector<float>& got,
                             const std::vector<float>& expected,
                             const float abs_tol,
                             const char* tag) {
        ASSERT_EQ(got.size(), expected.size()) << tag;
        for (size_t i = 0; i < got.size(); i += 2) {
            const bool got_invalid = isInvalid(got[i], got[i + 1]);
            const bool exp_invalid = isInvalid(expected[i], expected[i + 1]);
            EXPECT_EQ(got_invalid, exp_invalid) << tag << " index=" << (i / 2);
            if (got_invalid || exp_invalid) {
                continue;
            }
            EXPECT_NEAR(got[i], expected[i], abs_tol) << tag << " x index=" << (i / 2);
            EXPECT_NEAR(got[i + 1], expected[i + 1], abs_tol) << tag << " y index=" << (i / 2);
        }
    }

    constexpr std::array<float, 9> kRotation{
        0.9393727f,
        0.0f,
        -0.3428978f,
        0.0f,
        1.0f,
        0.0f,
        0.3428978f,
        0.0f,
        0.9393727f,
    };
    constexpr std::array<float, 3> kTranslation{1.0f, -2.0f, 0.5f};
} // namespace

TEST(SelectionTensorProjection, EmptyMeans) {
    for (const GpuBackend backend : backends_under_test()) {
        SCOPED_TRACE(label(backend));
        GpuBackendScope scope(backend);
        const Tensor means = Tensor::empty({0, 3}, Device::GPU, DataType::Float32);
        const auto projected = lfs::rendering::project_screen_positions_tensor(
            means, 64, 64, kRotation, kTranslation, 50.0f, 50.0f, 32.0f, 32.0f,
            ScreenWindowCameraModel::Pinhole, 1.0f, nullptr, nullptr, {});
        EXPECT_FALSE(projected.is_valid() && projected.numel() > 0);
    }
}

TEST(SelectionTensorProjection, PinholeOrthoEquirectMatchCpu) {
    const std::vector<float> means{
        0.4f,
        -0.2f,
        -3.0f,
        2.0f,
        1.5f,
        4.0f,
        0.0f,
        0.0f,
        0.0f,
        std::numeric_limits<float>::quiet_NaN(),
        0.0f,
        -2.0f,
        -1.0f,
        0.5f,
        -8.0f,
        0.1f,
        -0.4f,
        0.5f,
    };
    constexpr int width = 640;
    constexpr int height = 480;
    constexpr float fx = 500.0f;
    constexpr float fy = 480.0f;
    constexpr float cx = 320.0f;
    constexpr float cy = 240.0f;
    const ScreenWindowCameraModel models[] = {
        ScreenWindowCameraModel::Pinhole,
        ScreenWindowCameraModel::Orthographic,
        ScreenWindowCameraModel::Equirectangular,
    };
    for (const auto model : models) {
        std::vector<float> expected;
        cpuProject(means, width, height, kRotation, kTranslation, fx, fy, cx, cy, model, 42.0f,
                   nullptr, nullptr, {}, expected);
        const float tol = model == ScreenWindowCameraModel::Equirectangular ? 2.0e-3f * width : 1.0e-3f;
        for (const GpuBackend backend : backends_under_test()) {
            SCOPED_TRACE(label(backend) + std::string(" model=") + std::to_string(static_cast<int>(model)));
            const Tensor gpu_means = upload(means, TensorShape{6, 3}, backend);
            const auto projected = lfs::rendering::project_screen_positions_tensor(
                gpu_means, width, height, kRotation, kTranslation, fx, fy, cx, cy, model, 42.0f,
                nullptr, nullptr, {});
            ASSERT_TRUE(projected.is_valid());
            expectProjectedNear(projected.cpu().to_vector(), expected, tol, "dispatch");
            const auto program = lfs::rendering::project_screen_positions_tensor_program(
                gpu_means, width, height, kRotation, kTranslation, fx, fy, cx, cy, model, 42.0f,
                nullptr, nullptr, {});
            expectProjectedNear(program.cpu().to_vector(), expected, tol, "program");
        }
    }
}

TEST(SelectionTensorProjection, HiddenNodesAndClampedTransforms) {
    const std::vector<float> means{
        0.0f,
        0.0f,
        -2.0f,
        0.2f,
        0.0f,
        -2.0f,
        -0.2f,
        0.1f,
        -2.0f,
        0.0f,
        0.2f,
        -2.0f,
    };
    const std::vector<float> transforms{
        1.0f,
        0.0f,
        0.0f,
        0.0f,
        0.0f,
        1.0f,
        0.0f,
        0.0f,
        0.0f,
        0.0f,
        1.0f,
        0.0f,
        0.0f,
        0.0f,
        0.0f,
        1.0f,
        1.0f,
        0.0f,
        0.0f,
        1.5f,
        0.0f,
        1.0f,
        0.0f,
        0.0f,
        0.0f,
        0.0f,
        1.0f,
        0.0f,
        0.0f,
        0.0f,
        0.0f,
        1.0f,
    };
    const std::vector<int> indices{0, 1, -4, 99};
    const std::vector<bool> visibility{true, false};
    std::vector<float> expected;
    cpuProject(means, 320, 240, kRotation, kTranslation, 200.0f, 200.0f, 160.0f, 120.0f,
               ScreenWindowCameraModel::Pinhole, 1.0f, &transforms, &indices, visibility, expected);
    for (const GpuBackend backend : backends_under_test()) {
        SCOPED_TRACE(label(backend));
        const Tensor gpu_means = upload(means, TensorShape{4, 3}, backend);
        const Tensor gpu_transforms = upload(transforms, TensorShape{2, 4, 4}, backend);
        const Tensor gpu_indices = uploadI32(indices, backend);
        const auto projected = lfs::rendering::project_screen_positions_tensor(
            gpu_means, 320, 240, kRotation, kTranslation, 200.0f, 200.0f, 160.0f, 120.0f,
            ScreenWindowCameraModel::Pinhole, 1.0f, &gpu_transforms, &gpu_indices, visibility);
        expectProjectedNear(projected.cpu().to_vector(), expected, 1.0e-3f, "hidden");
        EXPECT_TRUE(isInvalid(expected[2], expected[3]));
        EXPECT_TRUE(isInvalid(expected[4], expected[5]));
        EXPECT_TRUE(isInvalid(expected[6], expected[7]));
    }
}

TEST(SelectionTensorProjection, NegativeOrthoScaleIsInvalid) {
    const std::vector<float> means{0.0f, 0.0f, -2.0f};
    const float bad_scales[] = {-5.0f, 0.0f, std::numeric_limits<float>::infinity()};
    for (const float scale : bad_scales) {
        for (const GpuBackend backend : backends_under_test()) {
            SCOPED_TRACE(label(backend) + std::string(" scale=") + std::to_string(scale));
            const Tensor gpu_means = upload(means, TensorShape{1, 3}, backend);
            const auto projected = lfs::rendering::project_screen_positions_tensor(
                gpu_means, 64, 64, kRotation, kTranslation, 10.0f, 10.0f, 32.0f, 32.0f,
                ScreenWindowCameraModel::Orthographic, scale, nullptr, nullptr, {});
            const auto xy = projected.cpu().to_vector();
            ASSERT_EQ(xy.size(), 2u);
            EXPECT_TRUE(isInvalid(xy[0], xy[1]));
        }
    }
}

TEST(SelectionTensorProjection, EquirectProjectsBehindCamera) {
    const std::vector<float> means{0.0f, 0.0f, 2.0f};
    constexpr std::array<float, 9> identity{1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f};
    constexpr std::array<float, 3> origin{0.0f, 0.0f, 0.0f};
    std::vector<float> expected;
    cpuProject(means, 256, 128, identity, origin, 1.0f, 1.0f, 128.0f, 64.0f,
               ScreenWindowCameraModel::Equirectangular, 1.0f, nullptr, nullptr, {}, expected);
    ASSERT_FALSE(isInvalid(expected[0], expected[1]));
    for (const GpuBackend backend : backends_under_test()) {
        SCOPED_TRACE(label(backend));
        const Tensor gpu_means = upload(means, TensorShape{1, 3}, backend);
        const auto projected = lfs::rendering::project_screen_positions_tensor(
            gpu_means, 256, 128, identity, origin, 1.0f, 1.0f, 128.0f, 64.0f,
            ScreenWindowCameraModel::Equirectangular, 1.0f, nullptr, nullptr, {});
        expectProjectedNear(projected.cpu().to_vector(), expected, 1.0f, "equirect-behind");
    }
}

TEST(SelectionTensorProjection, CudaKernelMatchesProgramOnPinhole) {
    if (!gpu_backend_available(GpuBackend::CUDA)) {
        GTEST_SKIP() << "CUDA backend required for kernel vs program parity";
    }
    const std::vector<float> means{
        0.3f,
        -0.1f,
        -1.5f,
        -0.4f,
        0.2f,
        -4.0f,
        1.0f,
        0.0f,
        3.0f,
    };
    const Tensor gpu_means = upload(means, TensorShape{3, 3}, GpuBackend::CUDA);
    const auto kernel = lfs::rendering::project_screen_positions_tensor(
        gpu_means, 800, 600, kRotation, kTranslation, 600.0f, 600.0f, 400.0f, 300.0f,
        ScreenWindowCameraModel::Pinhole, 1.0f, nullptr, nullptr, {});
    const auto program = lfs::rendering::project_screen_positions_tensor_program(
        gpu_means, 800, 600, kRotation, kTranslation, 600.0f, 600.0f, 400.0f, 300.0f,
        ScreenWindowCameraModel::Pinhole, 1.0f, nullptr, nullptr, {});
    expectProjectedNear(kernel.cpu().to_vector(), program.cpu().to_vector(), 1.0e-4f, "cuda-vs-program");
}

TEST(SelectionTensorProjection, SingleModelTransformBroadcastsWithoutGather) {
    const std::vector<float> means{0.0f, 0.0f, -2.0f, 0.4f, -0.2f, -3.0f};
    const std::vector<float> transforms{
        1.0f,
        0.0f,
        0.0f,
        1.5f,
        0.0f,
        1.0f,
        0.0f,
        0.0f,
        0.0f,
        0.0f,
        1.0f,
        0.0f,
        0.0f,
        0.0f,
        0.0f,
        1.0f,
    };
    const std::vector<int> indices{0, 0};
    std::vector<float> expected;
    cpuProject(means, 320, 240, kRotation, kTranslation, 200.0f, 200.0f, 160.0f, 120.0f,
               ScreenWindowCameraModel::Pinhole, 1.0f, &transforms, &indices, {}, expected);
    for (const GpuBackend backend : backends_under_test()) {
        SCOPED_TRACE(label(backend));
        const Tensor gpu_means = upload(means, TensorShape{2, 3}, backend);
        const Tensor gpu_transforms = upload(transforms, TensorShape{1, 4, 4}, backend);
        const Tensor gpu_indices = uploadI32(indices, backend);
        const auto projected = lfs::rendering::project_screen_positions_tensor(
            gpu_means, 320, 240, kRotation, kTranslation, 200.0f, 200.0f, 160.0f, 120.0f,
            ScreenWindowCameraModel::Pinhole, 1.0f, &gpu_transforms, &gpu_indices, {});
        expectProjectedNear(projected.cpu().to_vector(), expected, 1.0e-3f, "broadcast");
        const auto program = lfs::rendering::project_screen_positions_tensor_program(
            gpu_means, 320, 240, kRotation, kTranslation, 200.0f, 200.0f, 160.0f, 120.0f,
            ScreenWindowCameraModel::Pinhole, 1.0f, &gpu_transforms, &gpu_indices, {});
        expectProjectedNear(program.cpu().to_vector(), expected, 1.0e-3f, "broadcast-program");
    }
}

TEST(SelectionTensorProjection, VisibleModelTransformsApplyPerIndex) {
    const std::vector<float> means{
        0.0f,
        0.0f,
        -2.0f,
        0.0f,
        0.0f,
        -2.0f,
        0.4f,
        0.1f,
        -3.0f,
    };
    const std::vector<float> transforms{
        1.0f,
        0.0f,
        0.0f,
        0.0f,
        0.0f,
        1.0f,
        0.0f,
        0.0f,
        0.0f,
        0.0f,
        1.0f,
        0.0f,
        0.0f,
        0.0f,
        0.0f,
        1.0f,
        1.0f,
        0.0f,
        0.0f,
        1.5f,
        0.0f,
        1.0f,
        0.0f,
        0.0f,
        0.0f,
        0.0f,
        1.0f,
        0.0f,
        0.0f,
        0.0f,
        0.0f,
        1.0f,
    };
    const std::vector<int> indices{0, 1, 1};
    const std::vector<bool> visibility{true, true};
    std::vector<float> expected;
    cpuProject(means, 320, 240, kRotation, kTranslation, 200.0f, 200.0f, 160.0f, 120.0f,
               ScreenWindowCameraModel::Pinhole, 1.0f, &transforms, &indices, visibility, expected);
    ASSERT_FALSE(isInvalid(expected[0], expected[1]));
    ASSERT_FALSE(isInvalid(expected[2], expected[3]));
    EXPECT_GT(std::abs(expected[2] - expected[0]), 1.0f);
    for (const GpuBackend backend : backends_under_test()) {
        SCOPED_TRACE(label(backend));
        const Tensor gpu_means = upload(means, TensorShape{3, 3}, backend);
        const Tensor gpu_transforms = upload(transforms, TensorShape{2, 4, 4}, backend);
        const Tensor gpu_indices = uploadI32(indices, backend);
        const auto projected = lfs::rendering::project_screen_positions_tensor(
            gpu_means, 320, 240, kRotation, kTranslation, 200.0f, 200.0f, 160.0f, 120.0f,
            ScreenWindowCameraModel::Pinhole, 1.0f, &gpu_transforms, &gpu_indices, visibility);
        expectProjectedNear(projected.cpu().to_vector(), expected, 1.0e-3f, "per-index");
        const auto program = lfs::rendering::project_screen_positions_tensor_program(
            gpu_means, 320, 240, kRotation, kTranslation, 200.0f, 200.0f, 160.0f, 120.0f,
            ScreenWindowCameraModel::Pinhole, 1.0f, &gpu_transforms, &gpu_indices, visibility);
        expectProjectedNear(program.cpu().to_vector(), expected, 1.0e-3f, "per-index-program");
    }
}

TEST(SelectionTensorProjection, NonFiniteMeansAreInvalid) {
    const std::vector<float> means{
        std::numeric_limits<float>::infinity(),
        0.0f,
        -2.0f,
        -std::numeric_limits<float>::infinity(),
        0.0f,
        -2.0f,
        0.0f,
        std::numeric_limits<float>::quiet_NaN(),
        -2.0f,
        0.0f,
        0.0f,
        std::numeric_limits<float>::infinity(),
    };
    std::vector<float> expected;
    cpuProject(means, 64, 64, kRotation, kTranslation, 10.0f, 10.0f, 32.0f, 32.0f,
               ScreenWindowCameraModel::Pinhole, 1.0f, nullptr, nullptr, {}, expected);
    for (size_t i = 0; i < expected.size(); i += 2) {
        EXPECT_TRUE(isInvalid(expected[i], expected[i + 1])) << "cpu index=" << (i / 2);
    }
    for (const GpuBackend backend : backends_under_test()) {
        SCOPED_TRACE(label(backend));
        const Tensor gpu_means = upload(means, TensorShape{4, 3}, backend);
        const auto projected = lfs::rendering::project_screen_positions_tensor(
            gpu_means, 64, 64, kRotation, kTranslation, 10.0f, 10.0f, 32.0f, 32.0f,
            ScreenWindowCameraModel::Pinhole, 1.0f, nullptr, nullptr, {});
        expectProjectedNear(projected.cpu().to_vector(), expected, 1.0e-3f, "nonfinite");
    }
}

TEST(SelectionTensorProjection, EquirectViewZZeroMatchesAtan2Seam) {
    const std::vector<float> means{1.0f, 0.0f, 0.0f};
    constexpr std::array<float, 9> identity{1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f};
    constexpr std::array<float, 3> origin{0.0f, 0.0f, 0.0f};
    constexpr int width = 256;
    std::vector<float> expected;
    cpuProject(means, width, 128, identity, origin, 1.0f, 1.0f, 128.0f, 64.0f,
               ScreenWindowCameraModel::Equirectangular, 1.0f, nullptr, nullptr, {}, expected);
    ASSERT_FALSE(isInvalid(expected[0], expected[1]));
    // A direction along +X maps to three quarters of the panorama width.
    EXPECT_NEAR(expected[0], static_cast<float>(width) * 0.75f, 1.0e-3f);
    for (const GpuBackend backend : backends_under_test()) {
        SCOPED_TRACE(label(backend));
        const Tensor gpu_means = upload(means, TensorShape{1, 3}, backend);
        const auto projected = lfs::rendering::project_screen_positions_tensor(
            gpu_means, width, 128, identity, origin, 1.0f, 1.0f, 128.0f, 64.0f,
            ScreenWindowCameraModel::Equirectangular, 1.0f, nullptr, nullptr, {});
        expectProjectedNear(projected.cpu().to_vector(), expected, 1.0f, "equirect-seam");
        const auto program = lfs::rendering::project_screen_positions_tensor_program(
            gpu_means, width, 128, identity, origin, 1.0f, 1.0f, 128.0f, 64.0f,
            ScreenWindowCameraModel::Equirectangular, 1.0f, nullptr, nullptr, {});
        expectProjectedNear(program.cpu().to_vector(), expected, 1.0f, "equirect-seam-program");
    }
}

TEST(SelectionTensorProjection, MoreThanOneMillionExceedsOldCudaGridCap) {
    // Old projectScreenPositionsKernel launch was
    // min(ceil(n / 256), 4096) → 1,048,576 threads. n=1,100,003 leaves
    // [1,048,576, n) unwritten under that cap; Tensor::empty tail would not
    // match the CPU / program / Vulkan values.
    constexpr int n = 1'100'003;
    constexpr int kOldCudaThreadCap = 4096 * 256;
    static_assert(n > kOldCudaThreadCap);
    constexpr int width = 640;
    constexpr int height = 480;
    constexpr float fx = 500.0f;
    constexpr float fy = 500.0f;
    constexpr float cx = 320.0f;
    constexpr float cy = 240.0f;
    constexpr std::array<float, 9> identity{1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f};
    constexpr std::array<float, 3> origin{0.0f, 0.0f, 0.0f};

    std::vector<float> means(static_cast<size_t>(n) * 3);
    for (int i = 0; i < n; ++i) {
        means[static_cast<size_t>(i) * 3] = 0.25f + static_cast<float>(i) * 1.0e-6f;
        means[static_cast<size_t>(i) * 3 + 1] = -0.1f;
        means[static_cast<size_t>(i) * 3 + 2] = -2.0f;
    }
    std::vector<float> expected;
    cpuProject(means, width, height, identity, origin, fx, fy, cx, cy,
               ScreenWindowCameraModel::Pinhole, 1.0f, nullptr, nullptr, {}, expected);

    const int probes[] = {0, kOldCudaThreadCap - 1, kOldCudaThreadCap, kOldCudaThreadCap + 1, n - 1};
    auto tailMismatches = [&](const std::vector<float>& got, const char* tag) {
        EXPECT_EQ(got.size(), expected.size()) << tag;
        if (got.size() != expected.size()) {
            return;
        }
        for (const int i : probes) {
            const size_t x = static_cast<size_t>(i) * 2;
            EXPECT_NEAR(got[x], expected[x], 1.0e-3f) << tag << " x index=" << i;
            EXPECT_NEAR(got[x + 1], expected[x + 1], 1.0e-3f) << tag << " y index=" << i;
        }
        size_t mismatches = 0;
        size_t first = static_cast<size_t>(n);
        for (int i = kOldCudaThreadCap; i < n; ++i) {
            const size_t x = static_cast<size_t>(i) * 2;
            if (std::abs(got[x] - expected[x]) > 1.0e-3f ||
                std::abs(got[x + 1] - expected[x + 1]) > 1.0e-3f) {
                if (first == static_cast<size_t>(n)) {
                    first = static_cast<size_t>(i);
                }
                ++mismatches;
            }
        }
        EXPECT_EQ(mismatches, 0u) << tag << " first tail mismatch index=" << first;
    };

    std::vector<float> cuda_tail;
    for (const GpuBackend backend : backends_under_test()) {
        SCOPED_TRACE(label(backend));
        const Tensor gpu_means = upload(means, TensorShape{static_cast<size_t>(n), 3}, backend);
        const auto projected = lfs::rendering::project_screen_positions_tensor(
            gpu_means, width, height, identity, origin, fx, fy, cx, cy,
            ScreenWindowCameraModel::Pinhole, 1.0f, nullptr, nullptr, {});
        ASSERT_TRUE(projected.is_valid());
        const auto got = projected.cpu().to_vector();
        tailMismatches(got, "dispatch");
        const auto program = lfs::rendering::project_screen_positions_tensor_program(
            gpu_means, width, height, identity, origin, fx, fy, cx, cy,
            ScreenWindowCameraModel::Pinhole, 1.0f, nullptr, nullptr, {});
        tailMismatches(program.cpu().to_vector(), "program");
        if (backend == GpuBackend::CUDA) {
            cuda_tail = got;
        } else if (!cuda_tail.empty()) {
            tailMismatches(got, "vulkan-vs-cpu");
            ASSERT_EQ(got.size(), cuda_tail.size());
            size_t mismatches = 0;
            size_t first = static_cast<size_t>(n);
            for (int i = kOldCudaThreadCap; i < n; ++i) {
                const size_t x = static_cast<size_t>(i) * 2;
                if (std::abs(got[x] - cuda_tail[x]) > 1.0e-3f ||
                    std::abs(got[x + 1] - cuda_tail[x + 1]) > 1.0e-3f) {
                    if (first == static_cast<size_t>(n)) {
                        first = static_cast<size_t>(i);
                    }
                    ++mismatches;
                }
            }
            EXPECT_EQ(mismatches, 0u) << "vulkan vs cuda first tail mismatch index=" << first;
        }
    }
}
