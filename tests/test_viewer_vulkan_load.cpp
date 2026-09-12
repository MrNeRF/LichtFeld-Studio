/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/sh_value_quant.hpp"
#include "core/splat_data.hpp"
#include "core/tensor.hpp"
#include "core/tensor_backend.hpp"
#include "io/formats/ply.hpp"
#include "io/loader.hpp"
#include "io/splat_chapter.hpp"

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <future>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

namespace {

    using namespace lfs::core;

    constexpr size_t kTestSplats = 257;

    [[nodiscard]] std::filesystem::path splat_ply_path() {
        struct FixtureFile {
            std::filesystem::path path = std::filesystem::temp_directory_path() /
                                         ("lfs-viewer-backends-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) + ".ply");
            FixtureFile() {
                std::ofstream out(path, std::ios::binary);
                out << "ply\nformat binary_little_endian 1.0\nelement vertex " << kTestSplats << '\n';
                for (const char* name : {"x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2"})
                    out << "property float " << name << '\n';
                for (int i = 0; i < 45; ++i)
                    out << "property float f_rest_" << i << '\n';
                for (const char* name : {"opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"})
                    out << "property float " << name << '\n';
                out << "end_header\n";
                auto write_float = [&](float value) {
                    const auto bits = std::bit_cast<uint32_t>(value);
                    for (int byte = 0; byte < 4; ++byte)
                        out.put(static_cast<char>((bits >> (byte * 8)) & 0xff));
                };
                for (size_t row = 0; row < kTestSplats; ++row) {
                    for (float value : {row * 0.01f, 0.0f, 1.0f, 0.1f, 0.2f, 0.3f})
                        write_float(value);
                    for (int coefficient = 0; coefficient < 45; ++coefficient)
                        write_float(std::sin(float(row + coefficient) * 0.13f));
                    for (float value : {1.0f, -3.0f, -3.0f, -3.0f, 1.0f, 0.0f, 0.0f, 0.0f})
                        write_float(value);
                }
                out.close();
                if (!out)
                    throw std::runtime_error("Failed to write viewer PLY fixture");
            }
            ~FixtureFile() {
                std::error_code error;
                std::filesystem::remove(path, error);
            }
        };
        static const FixtureFile file;
        return file.path;
    }

    class ViewerVulkanLoad : public ::testing::Test {
        void SetUp() override {
            int device_count = 0;
            if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0)
                GTEST_SKIP() << "CUDA device unavailable";
        }
    };

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

TEST_F(ViewerVulkanLoad, PlyLoadsOnVulkanBackendWithQ16) {
    if (!gpu_backend_available(GpuBackend::Vulkan)) {
        GTEST_SKIP() << "Vulkan backend unavailable";
    }
    ASSERT_TRUE(sh_value_quant::enabled());

    SplatData cuda_model;
    {
        GpuBackendScope scope(GpuBackend::CUDA);
        cuda_model = load_splat(viewer_load_options());
    }
    ASSERT_TRUE(cuda_model.means_raw().is_valid());
    ASSERT_TRUE(cuda_model.shN_value_quantized());

    GpuBackendScope scope(GpuBackend::Vulkan);
    SplatData model = load_splat(viewer_load_options());
    ASSERT_TRUE(model.means_raw().is_valid());
    EXPECT_EQ(model.size(), kTestSplats);
    EXPECT_TRUE(model.shN_value_quantized());
    expect_backend(model.means_raw(), GpuBackend::Vulkan, "means");
    expect_backend(model.sh0_raw(), GpuBackend::Vulkan, "sh0");
    expect_backend(model.scaling_raw(), GpuBackend::Vulkan, "scaling");
    expect_backend(model.rotation_raw(), GpuBackend::Vulkan, "rotation");
    expect_backend(model.opacity_raw(), GpuBackend::Vulkan, "opacity");
    expect_backend(model.shN_raw(), GpuBackend::Vulkan, "shN");
    expect_backend(model.shN_value_bounds(), GpuBackend::Vulkan, "shN_value_bounds");
    expect_means_match_cpu_ply(model);

    const Tensor vk_codes = model.shN_raw().contiguous().cpu();
    const Tensor cu_codes = cuda_model.shN_raw().contiguous().cpu();
    ASSERT_EQ(vk_codes.bytes(), cu_codes.bytes());
    const auto* vk_u16 = static_cast<const std::uint16_t*>(vk_codes.data_ptr());
    const auto* cu_u16 = static_cast<const std::uint16_t*>(cu_codes.data_ptr());
    const size_t words = vk_codes.bytes() / sizeof(std::uint16_t);
    size_t code_mismatches = 0;
    size_t first_mismatch = words;
    const size_t cells = sh_value_quant::n_value_cells_per_prim(model.max_sh_coeffs_rest());
    for (size_t i = 0; i < words; ++i) {
        // CUDA leaves inactive lanes of the final SH block unspecified.
        const size_t primitive = (i / (cells * kShReorderSize)) * kShReorderSize + i % kShReorderSize;
        if (primitive >= model.size())
            continue;
        if (vk_u16[i] != cu_u16[i]) {
            if (code_mismatches < 8) {
                EXPECT_EQ(vk_u16[i], cu_u16[i]) << "q16 word " << i;
            }
            if (first_mismatch == words) {
                first_mismatch = i;
            }
            ++code_mismatches;
        }
    }
    EXPECT_EQ(code_mismatches, 0u) << "first mismatch at word " << first_mismatch;
    const Tensor vk_bounds = model.shN_value_bounds().contiguous().cpu();
    const Tensor cu_bounds = cuda_model.shN_value_bounds().contiguous().cpu();
    ASSERT_EQ(vk_bounds.numel(), cu_bounds.numel());
    ASSERT_EQ(vk_bounds.bytes(), cu_bounds.bytes());
    EXPECT_EQ(std::memcmp(vk_bounds.data_ptr(), cu_bounds.data_ptr(), vk_bounds.bytes()), 0);

    SplatData model2 = load_splat(viewer_load_options());
    ASSERT_TRUE(model2.means_raw().is_valid());
    EXPECT_FLOAT_EQ(model.means_raw().mean_scalar(), model2.means_raw().mean_scalar());

    using lfs::io::project::SplatChapterPayload;
    using lfs::io::project::SplatSourceKind;
    auto expected = SplatChapterPayload::capture(cuda_model, SplatSourceKind::ImportedPly, false);
    auto captured = SplatChapterPayload::capture(model, SplatSourceKind::ImportedPly, false);
    ASSERT_TRUE(expected.has_value());
    ASSERT_TRUE(captured.has_value());
    ASSERT_EQ(expected->bytes().size(), captured->bytes().size());
    EXPECT_TRUE(std::equal(expected->bytes().begin(), expected->bytes().end(), captured->bytes().begin()));

    struct HeadroomGuard {
        HeadroomGuard() { lfs::io::project::detail::set_splat_capture_no_headroom_for_testing(false); }
        ~HeadroomGuard() { lfs::io::project::detail::set_splat_capture_no_headroom_for_testing(std::nullopt); }
    } headroom;
    auto pending = SplatChapterPayload::start_async_capture(model, SplatSourceKind::ImportedPly, false);
    ASSERT_TRUE(pending.has_value());
    ASSERT_TRUE(*pending);
    model.means().fill_(42.0f);
    auto worker = std::async(std::launch::async, [snapshot = std::move(*pending)]() {
        return snapshot->complete();
    });
    auto asynchronous = worker.get();
    ASSERT_TRUE(asynchronous.has_value());
    ASSERT_EQ(captured->bytes().size(), asynchronous->bytes().size());
    EXPECT_TRUE(std::equal(captured->bytes().begin(), captured->bytes().end(), asynchronous->bytes().begin()));
    auto hydrated = asynchronous->hydrate();
    ASSERT_TRUE(hydrated.has_value());
    expect_means_match_cpu_ply(**hydrated);
}

TEST_F(ViewerVulkanLoad, PlyCudaDefaultKeepsQ16) {
    ASSERT_TRUE(sh_value_quant::enabled());

    GpuBackendScope scope(GpuBackend::CUDA);
    SplatData model = load_splat(viewer_load_options());
    ASSERT_TRUE(model.means_raw().is_valid());
    EXPECT_EQ(model.size(), kTestSplats);
    EXPECT_TRUE(model.shN_value_quantized());
    expect_backend(model.means_raw(), GpuBackend::CUDA, "means");
    expect_backend(model.sh0_raw(), GpuBackend::CUDA, "sh0");
    expect_backend(model.scaling_raw(), GpuBackend::CUDA, "scaling");
    expect_backend(model.rotation_raw(), GpuBackend::CUDA, "rotation");
    expect_backend(model.opacity_raw(), GpuBackend::CUDA, "opacity");
    expect_backend(model.shN_raw(), GpuBackend::CUDA, "shN");
    expect_means_match_cpu_ply(model);
}
