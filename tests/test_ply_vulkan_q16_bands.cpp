/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/cuda/sh_layout.cuh"
#include "core/error.hpp"
#include "core/sh_value_quant.hpp"
#include "core/splat_data.hpp"
#include "core/tensor.hpp"
#include "core/tensor_backend.hpp"
#include "io/error.hpp"
#include "io/formats/ply.hpp"
#include "io/loader.hpp"

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <atomic>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

using namespace lfs::core;

namespace {

    constexpr size_t kPartialFinalPrims = 513; // 256 + 256 + 1
    constexpr std::uint32_t kShRest = 15;

    [[nodiscard]] bool has_cuda_device() {
        int device_count = 0;
        return cudaGetDeviceCount(&device_count) == cudaSuccess && device_count != 0;
    }

    class ScopedPlyQ16BandPrims {
    public:
        explicit ScopedPlyQ16BandPrims(const std::size_t band_prims) {
            lfs::io::set_ply_q16_band_prims_for_tests(band_prims);
        }

        ~ScopedPlyQ16BandPrims() {
            lfs::io::set_ply_q16_band_prims_for_tests(0);
        }

        ScopedPlyQ16BandPrims(const ScopedPlyQ16BandPrims&) = delete;
        ScopedPlyQ16BandPrims& operator=(const ScopedPlyQ16BandPrims&) = delete;
    };

    struct AllocCall {
        std::string name;
        DataType dtype;
        size_t capacity = 0;
        size_t elements = 0;
    };

    lfs::io::SplatTensorAllocator recording_allocator(std::vector<AllocCall>& calls) {
        return [&calls](TensorShape shape,
                        const size_t capacity,
                        const DataType dtype,
                        const std::string_view name) {
            calls.push_back(AllocCall{
                .name = std::string{name},
                .dtype = dtype,
                .capacity = capacity,
                .elements = shape.elements(),
            });
            Tensor tensor = Tensor::empty(std::move(shape), Device::GPU, dtype);
            tensor.set_name(std::string{name});
            return tensor;
        };
    }

    [[nodiscard]] bool called_for_float_shN(const std::vector<AllocCall>& calls) {
        return std::any_of(calls.begin(), calls.end(), [](const AllocCall& call) {
            return call.name == "SplatData.shN" && call.dtype == DataType::Float32;
        });
    }

    void expect_q16_allocator_contract(const std::vector<AllocCall>& calls,
                                       const size_t n_prims) {
        EXPECT_FALSE(called_for_float_shN(calls));

        bool saw_codes = false;
        bool saw_bounds = false;
        const size_t expected_cells =
            sh_value_quant::sh_value_u16_count(n_prims, kShRest);
        const size_t expected_bounds =
            sh_value_quant::n_bounds_for_prims(n_prims) * 2;
        for (const AllocCall& call : calls) {
            if (call.name == "SplatData.shN") {
                saw_codes = true;
                EXPECT_EQ(call.dtype, DataType::Float16);
                EXPECT_EQ(call.elements, expected_cells);
                EXPECT_EQ(call.capacity, expected_cells);
            }
            if (call.name == "SplatData.shN_value_bounds") {
                saw_bounds = true;
                EXPECT_EQ(call.dtype, DataType::Float32);
                EXPECT_EQ(call.elements, expected_bounds);
                EXPECT_EQ(call.capacity, expected_bounds);
            }
        }
        EXPECT_TRUE(saw_codes);
        EXPECT_TRUE(saw_bounds);
    }

    void write_float(std::ostream& out, const float value) {
        const auto bits = std::bit_cast<std::uint32_t>(value);
        for (int byte = 0; byte < 4; ++byte) {
            out.put(static_cast<char>((bits >> (byte * 8)) & 0xff));
        }
    }

    struct PlyFixture {
        std::filesystem::path path;
        size_t file_rows = 0;
        size_t expected_valid = 0;

        explicit PlyFixture(const size_t n,
                            const bool nan_rest_tail = false,
                            const bool inf_opacity_first = false) {
            file_rows = n + (nan_rest_tail ? 1 : 0);
            expected_valid = n;
            static std::atomic<std::uint64_t> fixture_id{0};
            path = std::filesystem::temp_directory_path() /
                   ("lfs-ply-vulkan-q16-" +
                    std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) +
                    "-" + std::to_string(fixture_id.fetch_add(1)) + ".ply");
            std::ofstream out(path, std::ios::binary);
            out << "ply\nformat binary_little_endian 1.0\nelement vertex " << file_rows
                << '\n';
            for (const char* name : {"x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2"}) {
                out << "property float " << name << '\n';
            }
            for (int i = 0; i < 45; ++i) {
                out << "property float f_rest_" << i << '\n';
            }
            for (const char* name : {"opacity", "scale_0", "scale_1", "scale_2", "rot_0",
                                     "rot_1", "rot_2", "rot_3"}) {
                out << "property float " << name << '\n';
            }
            out << "end_header\n";

            auto write_row = [&](const size_t row, const bool nan_rest, const bool inf_opacity) {
                write_float(out, static_cast<float>(row) * 0.01f);
                write_float(out, 0.0f);
                write_float(out, 1.0f);
                write_float(out, 0.1f);
                write_float(out, 0.2f);
                write_float(out, 0.3f);
                for (int coefficient = 0; coefficient < 45; ++coefficient) {
                    const float value = nan_rest && coefficient == 0
                                            ? std::numeric_limits<float>::quiet_NaN()
                                            : std::sin(static_cast<float>(row + coefficient) * 0.13f);
                    write_float(out, value);
                }
                write_float(out, inf_opacity ? std::numeric_limits<float>::infinity() : 1.0f);
                write_float(out, -3.0f);
                write_float(out, -3.0f);
                write_float(out, -3.0f);
                write_float(out, 1.0f);
                write_float(out, 0.0f);
                write_float(out, 0.0f);
                write_float(out, 0.0f);
            };

            for (size_t row = 0; row < n; ++row) {
                write_row(row, false, inf_opacity_first && row == 0);
            }
            if (nan_rest_tail) {
                write_row(n, true, false);
            }
            out.close();
            if (!out) {
                throw std::runtime_error("Failed to write Vulkan q16 PLY fixture");
            }
        }

        ~PlyFixture() {
            std::error_code error;
            std::filesystem::remove(path, error);
        }

        PlyFixture(const PlyFixture&) = delete;
        PlyFixture& operator=(const PlyFixture&) = delete;
    };

    SplatData load_ply_q16(const std::filesystem::path& path,
                           std::vector<AllocCall>& calls,
                           std::vector<lfs::io::Diagnostic>* warnings = nullptr) {
        lfs::io::LoadOptions options;
        options.splat_tensor_allocator = recording_allocator(calls);
        options.shN_q16 = true;
        auto loaded = lfs::io::load_ply(path, options);
        EXPECT_TRUE(loaded.has_value()) << lfs::format_for_developer(loaded.error());
        if (!loaded.has_value()) {
            return {};
        }
        if (warnings != nullptr) {
            *warnings = std::move(loaded->warnings);
        }
        return std::move(loaded->value);
    }

    void expect_q16_payload_dtypes(const SplatData& model) {
        ASSERT_TRUE(model.shN_value_quantized());
        const Tensor& codes = model.shN_raw();
        const Tensor& bounds = model.shN_value_bounds();
        EXPECT_EQ(codes.dtype(), DataType::Float16);
        EXPECT_EQ(bounds.dtype(), DataType::Float32);
        EXPECT_EQ(codes.bytes(), codes.numel() * dtype_size(DataType::Float16));
        EXPECT_EQ(bounds.bytes(), bounds.numel() * dtype_size(DataType::Float32));
    }

    void expect_q16_byte_identical(const SplatData& expected, const SplatData& actual) {
        ASSERT_TRUE(expected.shN_value_quantized());
        ASSERT_TRUE(actual.shN_value_quantized());
        ASSERT_EQ(expected.size(), actual.size());
        ASSERT_EQ(expected.get_max_sh_degree(), actual.get_max_sh_degree());
        ASSERT_EQ(expected.get_active_sh_degree(), actual.get_active_sh_degree());
        expect_q16_payload_dtypes(expected);
        expect_q16_payload_dtypes(actual);

        const Tensor codes_a = expected.shN_raw().cpu();
        const Tensor codes_b = actual.shN_raw().cpu();
        ASSERT_EQ(codes_a.dtype(), DataType::Float16);
        ASSERT_EQ(codes_b.dtype(), DataType::Float16);
        ASSERT_EQ(codes_a.numel(), codes_b.numel());
        ASSERT_EQ(codes_a.bytes(), codes_b.bytes());
        EXPECT_EQ(std::memcmp(codes_a.data_ptr(), codes_b.data_ptr(), codes_a.bytes()), 0);

        const Tensor bounds_a = expected.shN_value_bounds().cpu();
        const Tensor bounds_b = actual.shN_value_bounds().cpu();
        ASSERT_EQ(bounds_a.dtype(), DataType::Float32);
        ASSERT_EQ(bounds_b.dtype(), DataType::Float32);
        ASSERT_EQ(bounds_a.numel(), bounds_b.numel());
        ASSERT_EQ(bounds_a.bytes(), bounds_b.bytes());
        EXPECT_EQ(std::memcmp(bounds_a.data_ptr(), bounds_b.data_ptr(), bounds_a.bytes()), 0);
    }

    void expect_q16_active_codes_and_bounds(const SplatData& cuda_model,
                                            const SplatData& vulkan_model) {
        ASSERT_TRUE(cuda_model.shN_value_quantized());
        ASSERT_TRUE(vulkan_model.shN_value_quantized());
        ASSERT_EQ(cuda_model.size(), vulkan_model.size());
        expect_q16_payload_dtypes(cuda_model);
        expect_q16_payload_dtypes(vulkan_model);

        const Tensor cu_codes = cuda_model.shN_raw().contiguous().cpu();
        const Tensor vk_codes = vulkan_model.shN_raw().contiguous().cpu();
        ASSERT_EQ(cu_codes.bytes(), vk_codes.bytes());
        const auto* cu_u16 = static_cast<const std::uint16_t*>(cu_codes.data_ptr());
        const auto* vk_u16 = static_cast<const std::uint16_t*>(vk_codes.data_ptr());
        const size_t words = cu_codes.bytes() / sizeof(std::uint16_t);
        const size_t cells = sh_value_quant::n_value_cells_per_prim(kShRest);
        size_t mismatches = 0;
        size_t first = words;
        for (size_t i = 0; i < words; ++i) {
            const size_t primitive =
                (i / (cells * kShReorderSize)) * kShReorderSize + i % kShReorderSize;
            if (primitive >= vulkan_model.size()) {
                continue;
            }
            if (cu_u16[i] != vk_u16[i]) {
                if (mismatches < 8) {
                    EXPECT_EQ(vk_u16[i], cu_u16[i]) << "q16 word " << i;
                }
                if (first == words) {
                    first = i;
                }
                ++mismatches;
            }
        }
        EXPECT_EQ(mismatches, 0u) << "first mismatch at word " << first;

        const Tensor cu_bounds = cuda_model.shN_value_bounds().contiguous().cpu();
        const Tensor vk_bounds = vulkan_model.shN_value_bounds().contiguous().cpu();
        ASSERT_EQ(cu_bounds.dtype(), DataType::Float32);
        ASSERT_EQ(vk_bounds.dtype(), DataType::Float32);
        ASSERT_EQ(cu_bounds.bytes(), vk_bounds.bytes());
        EXPECT_EQ(std::memcmp(cu_bounds.data_ptr(), vk_bounds.data_ptr(), cu_bounds.bytes()), 0);
    }

    void expect_decoded_canonical_equal(const SplatData& expected, const SplatData& actual) {
        const Tensor a = expected.shN_canonical_cpu().contiguous();
        const Tensor b = actual.shN_canonical_cpu().contiguous();
        ASSERT_EQ(a.dtype(), DataType::Float32);
        ASSERT_EQ(b.dtype(), DataType::Float32);
        ASSERT_EQ(a.numel(), b.numel());
        ASSERT_EQ(a.bytes(), b.bytes());
        EXPECT_EQ(std::memcmp(a.data_ptr(), b.data_ptr(), a.bytes()), 0);
    }

} // namespace

TEST(PlyVulkanQ16Bands, SkipsFloatShNAndBandedMatchesUnbanded) {
    if (!has_cuda_device()) {
        GTEST_SKIP() << "CUDA device unavailable";
    }
    if (!gpu_backend_available(GpuBackend::Vulkan)) {
        GTEST_SKIP() << "Vulkan backend unavailable";
    }

    const PlyFixture ply(kPartialFinalPrims);
    GpuBackendScope scope(GpuBackend::Vulkan);

    std::vector<AllocCall> unbanded_calls;
    SplatData unbanded = load_ply_q16(ply.path, unbanded_calls);
    ASSERT_TRUE(unbanded.shN_value_quantized());
    EXPECT_EQ(unbanded.size(), kPartialFinalPrims);
    expect_q16_allocator_contract(unbanded_calls, kPartialFinalPrims);
    expect_q16_payload_dtypes(unbanded);

    for (const std::size_t band : {std::size_t{256}, std::size_t{512}}) {
        SCOPED_TRACE("band=" + std::to_string(band));
        ScopedPlyQ16BandPrims scoped_band(band);
        std::vector<AllocCall> banded_calls;
        SplatData banded = load_ply_q16(ply.path, banded_calls);
        ASSERT_TRUE(banded.shN_value_quantized());
        EXPECT_EQ(banded.size(), kPartialFinalPrims);
        expect_q16_allocator_contract(banded_calls, kPartialFinalPrims);
        expect_q16_byte_identical(unbanded, banded);
        expect_decoded_canonical_equal(unbanded, banded);
    }
}

TEST(PlyVulkanQ16Bands, CudaVulkanDecodedParityAtBandBoundaries) {
    if (!has_cuda_device()) {
        GTEST_SKIP() << "CUDA device unavailable";
    }
    if (!gpu_backend_available(GpuBackend::Vulkan)) {
        GTEST_SKIP() << "Vulkan backend unavailable";
    }

    const PlyFixture ply(kPartialFinalPrims);
    ScopedPlyQ16BandPrims scoped_band(256);

    SplatData cuda_model;
    std::vector<AllocCall> cuda_calls;
    {
        GpuBackendScope scope(GpuBackend::CUDA);
        cuda_model = load_ply_q16(ply.path, cuda_calls);
    }
    ASSERT_TRUE(cuda_model.shN_value_quantized());
    EXPECT_EQ(cuda_model.size(), kPartialFinalPrims);
    expect_q16_allocator_contract(cuda_calls, kPartialFinalPrims);

    SplatData vulkan_model;
    std::vector<AllocCall> vulkan_calls;
    {
        GpuBackendScope scope(GpuBackend::Vulkan);
        vulkan_model = load_ply_q16(ply.path, vulkan_calls);
    }
    ASSERT_TRUE(vulkan_model.shN_value_quantized());
    EXPECT_EQ(vulkan_model.size(), kPartialFinalPrims);
    expect_q16_allocator_contract(vulkan_calls, kPartialFinalPrims);
    EXPECT_EQ(gpu_backend_of(vulkan_model.shN_raw()), GpuBackend::Vulkan);
    EXPECT_EQ(gpu_backend_of(vulkan_model.shN_value_bounds()), GpuBackend::Vulkan);

    expect_q16_active_codes_and_bounds(cuda_model, vulkan_model);
    expect_decoded_canonical_equal(cuda_model, vulkan_model);
}

TEST(PlyVulkanQ16Bands, NanInfCleanupPreservesQ16Parity) {
    if (!has_cuda_device()) {
        GTEST_SKIP() << "CUDA device unavailable";
    }
    if (!gpu_backend_available(GpuBackend::Vulkan)) {
        GTEST_SKIP() << "Vulkan backend unavailable";
    }

    const PlyFixture ply(kPartialFinalPrims, true, true);
    ASSERT_EQ(ply.file_rows, kPartialFinalPrims + 1);

    auto expect_cleanup_warnings = [](const std::vector<lfs::io::Diagnostic>& warnings) {
        bool discarded = false;
        bool repaired = false;
        for (const auto& warning : warnings) {
            if (warning.message.find("invalid splat") != std::string::npos) {
                discarded = true;
            }
            if (warning.message.find("infinite opacity") != std::string::npos) {
                repaired = true;
            }
        }
        EXPECT_TRUE(discarded);
        EXPECT_TRUE(repaired);
    };

    std::vector<lfs::io::Diagnostic> cuda_warnings;
    SplatData cuda_model;
    {
        GpuBackendScope scope(GpuBackend::CUDA);
        std::vector<AllocCall> calls;
        cuda_model = load_ply_q16(ply.path, calls, &cuda_warnings);
        expect_q16_allocator_contract(calls, kPartialFinalPrims);
    }
    ASSERT_EQ(cuda_model.size(), kPartialFinalPrims);
    expect_cleanup_warnings(cuda_warnings);
    EXPECT_FLOAT_EQ(cuda_model.opacity_raw().cpu().ptr<float>()[0], 20.0f);

    std::vector<lfs::io::Diagnostic> vulkan_unbanded_warnings;
    SplatData vulkan_unbanded;
    {
        GpuBackendScope scope(GpuBackend::Vulkan);
        std::vector<AllocCall> calls;
        vulkan_unbanded = load_ply_q16(ply.path, calls, &vulkan_unbanded_warnings);
        expect_q16_allocator_contract(calls, kPartialFinalPrims);
    }
    ASSERT_EQ(vulkan_unbanded.size(), kPartialFinalPrims);
    expect_cleanup_warnings(vulkan_unbanded_warnings);
    EXPECT_FLOAT_EQ(vulkan_unbanded.opacity_raw().cpu().ptr<float>()[0], 20.0f);

    std::vector<lfs::io::Diagnostic> vulkan_banded_warnings;
    SplatData vulkan_banded;
    {
        GpuBackendScope scope(GpuBackend::Vulkan);
        ScopedPlyQ16BandPrims scoped_band(256);
        std::vector<AllocCall> calls;
        vulkan_banded = load_ply_q16(ply.path, calls, &vulkan_banded_warnings);
        expect_q16_allocator_contract(calls, kPartialFinalPrims);
    }
    ASSERT_EQ(vulkan_banded.size(), kPartialFinalPrims);
    expect_cleanup_warnings(vulkan_banded_warnings);

    expect_q16_byte_identical(vulkan_unbanded, vulkan_banded);
    expect_q16_active_codes_and_bounds(cuda_model, vulkan_unbanded);
    expect_decoded_canonical_equal(cuda_model, vulkan_unbanded);
    expect_decoded_canonical_equal(vulkan_unbanded, vulkan_banded);
}
