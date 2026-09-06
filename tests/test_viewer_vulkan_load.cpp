/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/sh_value_quant.hpp"
#include "core/splat_data.hpp"
#include "core/tensor.hpp"
#include "core/tensor_backend.hpp"
#include "io/formats/ply.hpp"
#include "io/loader.hpp"

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <filesystem>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

namespace {

    using namespace lfs::core;

    [[nodiscard]] std::filesystem::path splat_ply_path() {
        return std::filesystem::path(PROJECT_ROOT_PATH) / "output" / "splat_1581.ply";
    }

    void skip_if_missing_ply() {
        if (!std::filesystem::exists(splat_ply_path())) {
            GTEST_SKIP() << "Missing test asset: " << splat_ply_path();
        }
    }

    void require_cuda() {
        int device_count = 0;
        if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
            GTEST_SKIP() << "CUDA device unavailable";
        }
    }

    lfs::io::SplatTensorAllocator plain_splat_allocator() {
        return [](TensorShape shape,
                  const size_t capacity,
                  const DataType dtype,
                  const std::string_view name) {
            (void)capacity;
            Tensor tensor = Tensor::empty(std::move(shape), Device::CUDA, dtype);
            tensor.set_name(std::string{name});
            return tensor;
        };
    }

    lfs::io::LoadOptions viewer_load_options() {
        lfs::io::LoadOptions options;
        options.splat_tensor_allocator = plain_splat_allocator();
        options.shN_q16 = sh_value_quant::enabled();
        return options;
    }

    SplatData load_splat(const lfs::io::LoadOptions& options) {
        auto loader = lfs::io::Loader::create();
        auto loaded = loader->load(splat_ply_path(), options);
        EXPECT_TRUE(loaded.has_value()) << (loaded ? "" : loaded.error().format());
        if (!loaded.has_value()) {
            return {};
        }
        auto* splat = std::get_if<std::shared_ptr<SplatData>>(&loaded->data);
        EXPECT_TRUE(splat != nullptr && *splat);
        if (splat == nullptr || !*splat) {
            return {};
        }
        return std::move(**splat);
    }

    void expect_backend(const Tensor& tensor, const GpuBackend backend, const char* const name) {
        ASSERT_TRUE(tensor.is_valid()) << name;
        EXPECT_EQ(gpu_backend_of(tensor), backend) << name;
    }

    void expect_means_match_cpu_ply(const SplatData& model) {
        auto cpu_pc = lfs::io::load_ply_point_cloud(splat_ply_path(), {});
        ASSERT_TRUE(cpu_pc.has_value()) << cpu_pc.error();
        ASSERT_TRUE(cpu_pc->means.is_valid());
        const Tensor gpu_means = model.means_raw().cpu().contiguous();
        const Tensor cpu_means = cpu_pc->means.contiguous();
        ASSERT_EQ(gpu_means.dtype(), DataType::Float32);
        ASSERT_EQ(cpu_means.dtype(), DataType::Float32);
        ASSERT_GE(gpu_means.size(0), 3u);
        ASSERT_GE(cpu_means.size(0), 3u);
        const size_t rows = 3;
        for (size_t i = 0; i < rows * 3; ++i) {
            EXPECT_NEAR(gpu_means.ptr<float>()[i], cpu_means.ptr<float>()[i], 1e-6f) << "element " << i;
        }
    }

} // namespace

TEST(ViewerVulkanLoad, PlyLoadsOnVulkanBackendWithoutQ16) {
    require_cuda();
    skip_if_missing_ply();
    if (!gpu_backend_available(GpuBackend::Vulkan)) {
        GTEST_SKIP() << "Vulkan backend unavailable";
    }

    GpuBackendScope scope(GpuBackend::Vulkan);
    SplatData model = load_splat(viewer_load_options());
    ASSERT_TRUE(model.means_raw().is_valid());
    EXPECT_EQ(model.size(), 86976);
    EXPECT_FALSE(model.shN_value_quantized());
    expect_backend(model.means_raw(), GpuBackend::Vulkan, "means");
    expect_backend(model.sh0_raw(), GpuBackend::Vulkan, "sh0");
    expect_backend(model.scaling_raw(), GpuBackend::Vulkan, "scaling");
    expect_backend(model.rotation_raw(), GpuBackend::Vulkan, "rotation");
    expect_backend(model.opacity_raw(), GpuBackend::Vulkan, "opacity");
    expect_backend(model.shN_raw(), GpuBackend::Vulkan, "shN");
    expect_means_match_cpu_ply(model);
}

TEST(ViewerVulkanLoad, PlyCudaDefaultKeepsQ16) {
    require_cuda();
    skip_if_missing_ply();
    ASSERT_TRUE(sh_value_quant::enabled());

    GpuBackendScope scope(GpuBackend::CUDA);
    SplatData model = load_splat(viewer_load_options());
    ASSERT_TRUE(model.means_raw().is_valid());
    EXPECT_EQ(model.size(), 86976);
    EXPECT_TRUE(model.shN_value_quantized());
    expect_backend(model.means_raw(), GpuBackend::CUDA, "means");
    expect_backend(model.sh0_raw(), GpuBackend::CUDA, "sh0");
    expect_backend(model.scaling_raw(), GpuBackend::CUDA, "scaling");
    expect_backend(model.rotation_raw(), GpuBackend::CUDA, "rotation");
    expect_backend(model.opacity_raw(), GpuBackend::CUDA, "opacity");
    expect_backend(model.shN_raw(), GpuBackend::CUDA, "shN");
    expect_means_match_cpu_ply(model);
}
