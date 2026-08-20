/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

// SPZ format tests. Upstream-equivalent fixtures (v3/v4 SH0-3, coordinate
// declarations, forged unknown version) are generated once per suite via the
// vendored nianticlabs/spz writer rather than checked in, to keep binaries
// out of git.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <istream>
#include <memory>
#include <string>
#include <system_error>
#include <vector>

#include <gtest/gtest.h>

#include "coordinate-system-adobe.h"
#include "core/splat_data.hpp"
#include "io/exporter.hpp"
#include "io/formats/spz.hpp"
#include "io/loader.hpp"
#include "load-spz.h"

namespace fs = std::filesystem;
using namespace lfs::core;
using namespace lfs::io;

static float fixture_rand(uint32_t& state) {
    state = state * 1664525u + 1013904223u;
    return (state >> 8) * (1.0f / 16777216.0f); // [0,1)
}

static spz::GaussianCloud make_fixture_cloud(int sh_degree, int num_points, uint32_t seed) {
    spz::GaussianCloud c;
    c.numPoints = num_points;
    c.shDegree = sh_degree;
    uint32_t s = seed;
    const int sh_per_point = (sh_degree == 0) ? 0 : (sh_degree == 1) ? 9
                                                : (sh_degree == 2)   ? 24
                                                                     : 45;
    for (int i = 0; i < num_points; i++) {
        for (int k = 0; k < 3; k++)
            c.positions.push_back(fixture_rand(s) * 10.0f - 5.0f);
        for (int k = 0; k < 3; k++)
            c.scales.push_back(fixture_rand(s) * -6.0f);
        float q[4];
        float norm = 0;
        for (int k = 0; k < 4; k++) {
            q[k] = fixture_rand(s) * 2.0f - 1.0f;
            norm += q[k] * q[k];
        }
        norm = std::sqrt(norm);
        for (int k = 0; k < 4; k++)
            c.rotations.push_back(q[k] / norm);
        c.alphas.push_back(fixture_rand(s) * 8.0f - 4.0f);
        for (int k = 0; k < 3; k++)
            c.colors.push_back(fixture_rand(s) * 2.0f - 1.0f);
        for (int k = 0; k < sh_per_point; k++)
            c.sh.push_back(fixture_rand(s) * 0.5f - 0.25f);
    }
    return c;
}

class SpzFormatTest : public ::testing::Test {
protected:
    static constexpr float EPSILON = 1e-4f;
    static constexpr float SPZ_TOLERANCE = 0.15f; // SPZ uses lossy 8-bit quantization

    const fs::path test_ply = fs::path(PROJECT_ROOT_PATH) / "windmill.ply";
    const fs::path temp_dir = fs::temp_directory_path() / "lfs_spz_test";

    static fs::path fixture_dir() {
        return fs::temp_directory_path() / "lfs_spz_test" / "fixtures";
    }

    static bool write_spz_file(
        const fs::path& path,
        const spz::GaussianCloud& cloud,
        uint32_t version) {
        spz::PackOptions options;
        options.from = spz::CoordinateSystem::RDF;
        options.version = version;
        return spz::saveSpz(cloud, options, path.string());
    }

    static void attach_coordinate_extension(
        spz::GaussianCloud& cloud,
        spz::CoordinateSystem system) {
        auto ext = std::make_shared<spz::SpzExtensionCoordinateSystemAdobe>();
        ext->coordinateSystem = system;
        cloud.extensions.push_back(std::move(ext));
    }

    static void write_upstream_fixtures(const fs::path& dir) {
        for (int degree = 0; degree <= 3; ++degree) {
            auto cloud = make_fixture_cloud(degree, 100, 42u + static_cast<uint32_t>(degree));
            const auto v3_name = std::string("upstream_v3_sh") + std::to_string(degree) + ".spz";
            const auto v4_name = std::string("upstream_v4_sh") + std::to_string(degree) + ".spz";
            ASSERT_TRUE(write_spz_file(dir / v3_name, cloud, 3)) << v3_name;
            attach_coordinate_extension(cloud, spz::CoordinateSystem::RDF);
            ASSERT_TRUE(write_spz_file(dir / v4_name, cloud, 4)) << v4_name;
        }

        {
            const auto cloud = make_fixture_cloud(3, 100, 7u);
            ASSERT_TRUE(write_spz_file(dir / "upstream_v4_noext.spz", cloud, 4));
        }

        {
            const auto cloud = make_fixture_cloud(3, 100, 777u);
            auto rdf = cloud;
            attach_coordinate_extension(rdf, spz::CoordinateSystem::RDF);
            ASSERT_TRUE(write_spz_file(dir / "upstream_v4_declRDF.spz", rdf, 4));
            auto rub = cloud;
            attach_coordinate_extension(rub, spz::CoordinateSystem::RUB);
            ASSERT_TRUE(write_spz_file(dir / "upstream_v4_declRUB.spz", rub, 4));
        }

        {
            std::ifstream in(dir / "upstream_v4_noext.spz", std::ios::binary | std::ios::ate);
            ASSERT_TRUE(in.good());
            const auto size = static_cast<size_t>(in.tellg());
            ASSERT_GE(size, 8u);
            in.seekg(0, std::ios::beg);
            std::vector<uint8_t> bytes(size);
            in.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(size));
            ASSERT_TRUE(in.good());
            const uint32_t bad_version = 99;
            std::memcpy(bytes.data() + 4, &bad_version, sizeof(bad_version));
            std::ofstream out(dir / "upstream_v99_bad.spz", std::ios::binary | std::ios::trunc);
            out.write(reinterpret_cast<const char*>(bytes.data()),
                      static_cast<std::streamsize>(bytes.size()));
            ASSERT_TRUE(out.good());
        }
    }

    static void SetUpTestSuite() {
        const auto dir = fixture_dir();
        fs::create_directories(dir);
        write_upstream_fixtures(dir);
    }

    static void TearDownTestSuite() {
        fs::remove_all(fs::temp_directory_path() / "lfs_spz_test");
    }

    void SetUp() override {
        fs::create_directories(temp_dir);
        fs::create_directories(fixture_dir());
    }

    void TearDown() override {
        std::error_code ec;
        std::vector<fs::path> stale;
        for (const auto& entry : fs::directory_iterator(temp_dir, ec)) {
            if (entry.path().filename() != "fixtures") {
                stale.push_back(entry.path());
            }
        }
        for (const auto& path : stale) {
            fs::remove_all(path, ec);
        }
    }

    // Create SplatData with known values for testing
    static SplatData create_test_splat(size_t num_points, int sh_degree) {
        constexpr int SH_COEFFS[] = {0, 3, 8, 15};
        const size_t sh_coeffs = sh_degree > 0 ? SH_COEFFS[sh_degree] : 0;

        auto means = Tensor::empty({num_points, 3}, Device::CPU, DataType::Float32);
        auto sh0 = Tensor::empty({num_points, 1, 3}, Device::CPU, DataType::Float32);
        auto scaling = Tensor::empty({num_points, 3}, Device::CPU, DataType::Float32);
        auto rotation = Tensor::empty({num_points, 4}, Device::CPU, DataType::Float32);
        auto opacity = Tensor::empty({num_points, 1}, Device::CPU, DataType::Float32);

        Tensor shN;
        if (sh_coeffs > 0) {
            shN = Tensor::empty({num_points, sh_coeffs, 3}, Device::CPU, DataType::Float32);
        }

        auto* means_ptr = static_cast<float*>(means.data_ptr());
        auto* sh0_ptr = static_cast<float*>(sh0.data_ptr());
        auto* scaling_ptr = static_cast<float*>(scaling.data_ptr());
        auto* rotation_ptr = static_cast<float*>(rotation.data_ptr());
        auto* opacity_ptr = static_cast<float*>(opacity.data_ptr());

        for (size_t i = 0; i < num_points; ++i) {
            // Positions: spread out in space
            means_ptr[i * 3 + 0] = static_cast<float>(i % 10);
            means_ptr[i * 3 + 1] = static_cast<float>((i / 10) % 10);
            means_ptr[i * 3 + 2] = static_cast<float>(i / 100);

            // SH0 colors: values in typical SH range [-2, 2]
            sh0_ptr[i * 3 + 0] = 0.5f + 0.1f * static_cast<float>(i % 5);
            sh0_ptr[i * 3 + 1] = 0.3f + 0.1f * static_cast<float>((i + 1) % 5);
            sh0_ptr[i * 3 + 2] = 0.4f + 0.1f * static_cast<float>((i + 2) % 5);

            // Scales: log scale in valid SPZ range [-10, 6]
            scaling_ptr[i * 3 + 0] = -3.0f + 0.01f * static_cast<float>(i % 100);
            scaling_ptr[i * 3 + 1] = -3.0f + 0.01f * static_cast<float>((i + 1) % 100);
            scaling_ptr[i * 3 + 2] = -3.0f + 0.01f * static_cast<float>((i + 2) % 100);

            // Rotation: nontrivial normalized quaternions (wxyz format)
            constexpr float inv_sqrt_two = 0.70710678118f;
            const std::array<std::array<float, 4>, 4> rotations = {{
                {1.0f, 0.0f, 0.0f, 0.0f},
                {inv_sqrt_two, inv_sqrt_two, 0.0f, 0.0f},
                {inv_sqrt_two, 0.0f, inv_sqrt_two, 0.0f},
                {inv_sqrt_two, 0.0f, 0.0f, inv_sqrt_two},
            }};
            std::copy(rotations[i % rotations.size()].begin(),
                      rotations[i % rotations.size()].end(),
                      rotation_ptr + i * 4);

            // Opacity: logit values in SPZ-safe range (avoids inf from sigmoid)
            opacity_ptr[i] = -2.0f + 0.04f * static_cast<float>(i % 100);
        }

        // Fill higher order SH if present
        if (sh_coeffs > 0) {
            auto* shN_ptr = static_cast<float*>(shN.data_ptr());
            for (size_t i = 0; i < num_points * sh_coeffs * 3; ++i) {
                shN_ptr[i] = 0.1f * static_cast<float>(static_cast<int>(i % 10) - 5);
            }
        }

        return SplatData(
            sh_degree,
            std::move(means),
            std::move(sh0),
            std::move(shN),
            std::move(scaling),
            std::move(rotation),
            std::move(opacity),
            0.5f);
    }

    static std::vector<uint8_t> make_spz_header(
        const uint32_t point_count,
        const uint8_t sh_degree,
        const uint8_t fractional_bits,
        const uint32_t version = 3) {
        std::vector<uint8_t> bytes;
        const auto append_u32 = [&](const uint32_t value) {
            bytes.push_back(static_cast<uint8_t>(value));
            bytes.push_back(static_cast<uint8_t>(value >> 8));
            bytes.push_back(static_cast<uint8_t>(value >> 16));
            bytes.push_back(static_cast<uint8_t>(value >> 24));
        };
        append_u32(0x5053474e);
        append_u32(version);
        append_u32(point_count);
        bytes.push_back(sh_degree);
        bytes.push_back(fractional_bits);
        bytes.push_back(0);
        bytes.push_back(0);
        return bytes;
    }

    static bool write_gzipped_spz(
        const fs::path& path,
        const std::vector<uint8_t>& unpacked) {
        std::vector<uint8_t> compressed;
        if (!spz::compressGzipped(
                unpacked.data(), unpacked.size(), &compressed)) {
            return false;
        }
        std::ofstream stream(path, std::ios::binary | std::ios::trunc);
        stream.write(reinterpret_cast<const char*>(compressed.data()),
                     static_cast<std::streamsize>(compressed.size()));
        return stream.good();
    }

    static fs::path fixture_path(const char* filename) {
        return fixture_dir() / filename;
    }

    static std::vector<uint8_t> read_file_prefix(const fs::path& path, size_t n) {
        std::ifstream stream(path, std::ios::binary);
        std::vector<uint8_t> bytes(n, 0);
        stream.read(reinterpret_cast<char*>(bytes.data()),
                    static_cast<std::streamsize>(n));
        bytes.resize(static_cast<size_t>(stream.gcount()));
        return bytes;
    }

    static void expect_near_tensors(
        const Tensor& actual,
        const Tensor& expected,
        float tol,
        const char* label) {
        const auto a = actual.contiguous().to(Device::CPU);
        const auto e = expected.contiguous().to(Device::CPU);
        ASSERT_EQ(a.numel(), e.numel()) << label << " numel mismatch";
        const auto* a_ptr = static_cast<const float*>(a.data_ptr());
        const auto* e_ptr = static_cast<const float*>(e.data_ptr());
        for (size_t i = 0; i < a.numel(); ++i) {
            EXPECT_NEAR(a_ptr[i], e_ptr[i], tol) << label << " at " << i;
        }
    }

    static void expect_near_rotations(
        const Tensor& actual,
        const Tensor& expected,
        float tol,
        const char* label) {
        const auto a = actual.contiguous().to(Device::CPU);
        const auto e = expected.contiguous().to(Device::CPU);
        ASSERT_EQ(a.numel(), e.numel()) << label << " numel mismatch";
        ASSERT_EQ(a.numel() % 4, 0u) << label << " not quaternion-sized";
        const auto* a_ptr = static_cast<const float*>(a.data_ptr());
        const auto* e_ptr = static_cast<const float*>(e.data_ptr());
        const size_t n = a.numel() / 4;
        for (size_t i = 0; i < n; ++i) {
            const float dot = a_ptr[i * 4 + 0] * e_ptr[i * 4 + 0] +
                              a_ptr[i * 4 + 1] * e_ptr[i * 4 + 1] +
                              a_ptr[i * 4 + 2] * e_ptr[i * 4 + 2] +
                              a_ptr[i * 4 + 3] * e_ptr[i * 4 + 3];
            EXPECT_NEAR(std::abs(dot), 1.0f, tol) << label << " at point " << i;
        }
    }

    static void expect_all_finite(const Tensor& tensor, const char* label) {
        const auto cpu = tensor.contiguous().to(Device::CPU);
        const auto* ptr = static_cast<const float*>(cpu.data_ptr());
        for (size_t i = 0; i < cpu.numel(); ++i) {
            EXPECT_TRUE(std::isfinite(ptr[i])) << label << " non-finite at " << i;
        }
    }

    static void expect_splat_near(
        const SplatData& actual,
        const SplatData& expected,
        float tol) {
        EXPECT_EQ(actual.size(), expected.size());
        EXPECT_EQ(actual.get_max_sh_degree(), expected.get_max_sh_degree());
        expect_near_tensors(actual.means(), expected.means(), tol, "means");
        expect_near_tensors(actual.sh0(), expected.sh0(), tol, "sh0");
        expect_near_tensors(actual.scaling_raw(), expected.scaling_raw(), tol, "scale");
        expect_near_tensors(actual.opacity_raw(), expected.opacity_raw(), tol, "opacity");
        expect_near_rotations(actual.rotation_raw(), expected.rotation_raw(), tol, "rotation");
        if (expected.shN().is_valid() && expected.shN().numel() > 0) {
            ASSERT_TRUE(actual.shN().is_valid());
            expect_near_tensors(actual.shN_raw(), expected.shN_raw(), tol, "shN");
        }
    }
};

TEST_F(SpzFormatTest, RejectsOversizedHeaderBeforePayloadAllocation) {
    const fs::path path = temp_dir / "oversized_header.spz";
    ASSERT_TRUE(write_gzipped_spz(
        path,
        make_spz_header(spz::kMaxSpzPoints + 1, 3, 12)));

    const auto result = load_spz(path);

    EXPECT_FALSE(result.has_value());
}

TEST_F(SpzFormatTest, RejectsInvalidFractionalBitsBeforePayloadAllocation) {
    const fs::path path = temp_dir / "invalid_fractional_bits.spz";
    ASSERT_TRUE(write_gzipped_spz(
        path,
        make_spz_header(1, 0, spz::kMaxSpzFractionalBits + 1)));

    const auto result = load_spz(path);

    EXPECT_FALSE(result.has_value());
}

TEST_F(SpzFormatTest, RejectsTruncatedPayloadBeforeAttributeAllocation) {
    const fs::path path = temp_dir / "truncated_payload.spz";
    ASSERT_TRUE(write_gzipped_spz(path, make_spz_header(1, 0, 12)));

    const auto result = load_spz(path);

    EXPECT_FALSE(result.has_value());
}

TEST_F(SpzFormatTest, ExtremeEncodedOpacityLoadsAsFinite) {
    auto original = create_test_splat(1, 0);
    original.opacity_raw().ptr<float>()[0] = 1000.0f;
    const fs::path path = temp_dir / "finite_opacity.spz";
    ASSERT_TRUE(save_spz(original, {.output_path = path}).has_value());

    const auto result = load_spz(path);

    ASSERT_TRUE(result.has_value()) << result.error();
    const auto opacity = result->opacity_raw().cpu();
    EXPECT_TRUE(std::isfinite(opacity.ptr<float>()[0]));
}

// CRITICAL: Verify sh0 tensor shape is [N, 1, 3] - this caught our color bug
TEST_F(SpzFormatTest, Sh0TensorShapeIsCorrect) {
    auto original = create_test_splat(100, 1);

    const fs::path spz_path = temp_dir / "shape_test.spz";
    ASSERT_TRUE(save_spz(original, {.output_path = spz_path}).has_value());

    auto loader = Loader::create();
    const auto result = loader->load(spz_path);
    ASSERT_TRUE(result.has_value());

    const auto* splat_ptr = std::get_if<std::shared_ptr<SplatData>>(&result->data);
    ASSERT_NE(splat_ptr, nullptr);
    const auto& loaded = **splat_ptr;

    // sh0 MUST be [N, 1, 3] to match PLY loader - wrong shape causes color corruption
    EXPECT_EQ(loaded.sh0().ndim(), 3);
    EXPECT_EQ(loaded.sh0().size(0), 100);
    EXPECT_EQ(loaded.sh0().size(1), 1);
    EXPECT_EQ(loaded.sh0().size(2), 3);
}

// Verify all tensor shapes match between original and loaded
TEST_F(SpzFormatTest, AllTensorShapesPreserved) {
    auto original = create_test_splat(100, 1);

    const fs::path spz_path = temp_dir / "shapes.spz";
    ASSERT_TRUE(save_spz(original, {.output_path = spz_path}).has_value());

    auto loader = Loader::create();
    const auto result = loader->load(spz_path);
    ASSERT_TRUE(result.has_value());

    const auto* splat_ptr = std::get_if<std::shared_ptr<SplatData>>(&result->data);
    ASSERT_NE(splat_ptr, nullptr);
    const auto& loaded = **splat_ptr;

    EXPECT_EQ(loaded.size(), original.size());
    EXPECT_EQ(loaded.get_max_sh_degree(), original.get_max_sh_degree());

    // Check all tensor dimensions match
    EXPECT_EQ(loaded.means().ndim(), original.means().ndim());
    EXPECT_EQ(loaded.sh0().ndim(), original.sh0().ndim());
    EXPECT_EQ(loaded.scaling_raw().ndim(), original.scaling_raw().ndim());
    EXPECT_EQ(loaded.rotation_raw().ndim(), original.rotation_raw().ndim());
    EXPECT_EQ(loaded.opacity_raw().ndim(), original.opacity_raw().ndim());

    if (original.shN().is_valid()) {
        EXPECT_TRUE(loaded.shN().is_valid());
        EXPECT_EQ(loaded.shN().ndim(), original.shN().ndim());
    }
}

// Roundtrip test: values should be preserved within SPZ compression tolerance
TEST_F(SpzFormatTest, RoundtripPreservesValues) {
    auto original = create_test_splat(100, 1);

    const fs::path spz_path = temp_dir / "roundtrip.spz";
    ASSERT_TRUE(save_spz(original, {.output_path = spz_path}).has_value());

    auto loader = Loader::create();
    const auto result = loader->load(spz_path);
    ASSERT_TRUE(result.has_value());

    const auto* splat_ptr = std::get_if<std::shared_ptr<SplatData>>(&result->data);
    ASSERT_NE(splat_ptr, nullptr);
    const auto& loaded = **splat_ptr;

    const auto orig_means = original.means().contiguous().to(Device::CPU);
    const auto orig_sh0 = original.sh0().contiguous().to(Device::CPU);
    const auto orig_shN = original.shN_raw().contiguous().to(Device::CPU);
    const auto orig_scaling = original.scaling_raw().contiguous().to(Device::CPU);
    const auto orig_opacity = original.opacity_raw().contiguous().to(Device::CPU);

    const auto load_means = loaded.means().contiguous().to(Device::CPU);
    const auto load_sh0 = loaded.sh0().contiguous().to(Device::CPU);
    const auto load_shN = loaded.shN_raw().contiguous().to(Device::CPU);
    const auto load_scaling = loaded.scaling_raw().contiguous().to(Device::CPU);
    const auto load_opacity = loaded.opacity_raw().contiguous().to(Device::CPU);

    const auto* orig_means_ptr = static_cast<const float*>(orig_means.data_ptr());
    const auto* orig_sh0_ptr = static_cast<const float*>(orig_sh0.data_ptr());
    const auto* orig_shN_ptr = static_cast<const float*>(orig_shN.data_ptr());
    const auto* orig_scaling_ptr = static_cast<const float*>(orig_scaling.data_ptr());
    const auto* orig_opacity_ptr = static_cast<const float*>(orig_opacity.data_ptr());

    const auto* load_means_ptr = static_cast<const float*>(load_means.data_ptr());
    const auto* load_sh0_ptr = static_cast<const float*>(load_sh0.data_ptr());
    const auto* load_shN_ptr = static_cast<const float*>(load_shN.data_ptr());
    const auto* load_scaling_ptr = static_cast<const float*>(load_scaling.data_ptr());
    const auto* load_opacity_ptr = static_cast<const float*>(load_opacity.data_ptr());

    // Check positions (24-bit fixed point, high precision)
    for (size_t i = 0; i < 100 * 3; ++i) {
        EXPECT_NEAR(load_means_ptr[i], orig_means_ptr[i], 0.01f) << "Position mismatch at " << i;
    }

    // Check SH0 colors (8-bit quantization)
    for (size_t i = 0; i < 100 * 3; ++i) {
        EXPECT_NEAR(load_sh0_ptr[i], orig_sh0_ptr[i], SPZ_TOLERANCE) << "SH0 mismatch at " << i;
    }

    ASSERT_EQ(load_shN.numel(), orig_shN.numel());
    for (size_t i = 0; i < orig_shN.numel(); ++i) {
        EXPECT_NEAR(load_shN_ptr[i], orig_shN_ptr[i], SPZ_TOLERANCE) << "SHN mismatch at " << i;
    }

    // Check scales (8-bit quantization, range [-10, 6])
    for (size_t i = 0; i < 100 * 3; ++i) {
        EXPECT_NEAR(load_scaling_ptr[i], orig_scaling_ptr[i], SPZ_TOLERANCE) << "Scale mismatch at " << i;
    }

    // Check opacity
    for (size_t i = 0; i < 100; ++i) {
        EXPECT_NEAR(load_opacity_ptr[i], orig_opacity_ptr[i], SPZ_TOLERANCE) << "Opacity mismatch at " << i;
    }
}

// Verify rotation quaternion conversion (SPZ xyzw <-> SplatData wxyz)
TEST_F(SpzFormatTest, RotationQuaternionConversion) {
    auto original = create_test_splat(10, 0);

    const fs::path spz_path = temp_dir / "rotation.spz";
    ASSERT_TRUE(save_spz(original, {.output_path = spz_path}).has_value());

    auto loader = Loader::create();
    const auto result = loader->load(spz_path);
    ASSERT_TRUE(result.has_value());

    const auto* splat_ptr = std::get_if<std::shared_ptr<SplatData>>(&result->data);
    ASSERT_NE(splat_ptr, nullptr);
    const auto& loaded = **splat_ptr;

    const auto orig_rotation = original.rotation_raw().contiguous().to(Device::CPU);
    const auto load_rotation = loaded.rotation_raw().contiguous().to(Device::CPU);
    const auto* orig_rot_ptr = static_cast<const float*>(orig_rotation.data_ptr());
    const auto* load_rot_ptr = static_cast<const float*>(load_rotation.data_ptr());

    // Verify quaternions represent same rotation (q and -q are equivalent)
    for (size_t i = 0; i < 10; ++i) {
        const float dot = orig_rot_ptr[i * 4 + 0] * load_rot_ptr[i * 4 + 0] +
                          orig_rot_ptr[i * 4 + 1] * load_rot_ptr[i * 4 + 1] +
                          orig_rot_ptr[i * 4 + 2] * load_rot_ptr[i * 4 + 2] +
                          orig_rot_ptr[i * 4 + 3] * load_rot_ptr[i * 4 + 3];
        EXPECT_NEAR(std::abs(dot), 1.0f, SPZ_TOLERANCE) << "Rotation mismatch at point " << i;
    }
}

// Test SH degree 0 (no higher-order coefficients)
TEST_F(SpzFormatTest, ShDegree0) {
    auto original = create_test_splat(50, 0);

    const fs::path spz_path = temp_dir / "sh0.spz";
    ASSERT_TRUE(save_spz(original, {.output_path = spz_path}).has_value());

    auto loader = Loader::create();
    const auto result = loader->load(spz_path);
    ASSERT_TRUE(result.has_value());

    const auto* splat_ptr = std::get_if<std::shared_ptr<SplatData>>(&result->data);
    ASSERT_NE(splat_ptr, nullptr);
    const auto& loaded = **splat_ptr;

    EXPECT_EQ(loaded.get_max_sh_degree(), 0);
    // SplatData keeps a valid zero-length swizzled buffer at degree 0 so every
    // renderer can use one storage contract regardless of SH degree.
    EXPECT_TRUE(loaded.shN().is_valid());
    EXPECT_EQ(loaded.shN().numel(), 0);
}

// Test with real PLY file if available
TEST_F(SpzFormatTest, RealPlyRoundtrip) {
    if (!fs::exists(test_ply)) {
        GTEST_SKIP() << "windmill.ply not found";
    }

    auto loader = Loader::create();

    // Load PLY
    const auto ply_result = loader->load(test_ply);
    ASSERT_TRUE(ply_result.has_value());
    const auto* ply_ptr = std::get_if<std::shared_ptr<SplatData>>(&ply_result->data);
    ASSERT_NE(ply_ptr, nullptr);
    const auto& ply_splat = **ply_ptr;

    // Export to SPZ
    const fs::path spz_path = temp_dir / "real.spz";
    ASSERT_TRUE(save_spz(ply_splat, {.output_path = spz_path}).has_value());

    // Load SPZ
    const auto spz_result = loader->load(spz_path);
    ASSERT_TRUE(spz_result.has_value());
    const auto* spz_ptr = std::get_if<std::shared_ptr<SplatData>>(&spz_result->data);
    ASSERT_NE(spz_ptr, nullptr);
    const auto& spz_splat = **spz_ptr;

    // Verify structure matches
    EXPECT_EQ(spz_splat.size(), ply_splat.size());
    EXPECT_EQ(spz_splat.get_max_sh_degree(), ply_splat.get_max_sh_degree());

    // CRITICAL: sh0 shape must match PLY loader
    EXPECT_EQ(spz_splat.sh0().ndim(), ply_splat.sh0().ndim());
    EXPECT_EQ(spz_splat.sh0().size(1), ply_splat.sh0().size(1)); // Must be 1
}

// Default export writes SPZ v4 (NGSP/zstd container)
TEST_F(SpzFormatTest, DefaultExportWritesV4) {
    auto original = create_test_splat(10, 0);
    const fs::path path = temp_dir / "default_v4.spz";
    ASSERT_TRUE(save_spz(original, {.output_path = path}).has_value());

    const auto prefix = read_file_prefix(path, 8);
    ASSERT_GE(prefix.size(), 8u);
    EXPECT_EQ(prefix[0], static_cast<uint8_t>('N'));
    EXPECT_EQ(prefix[1], static_cast<uint8_t>('G'));
    EXPECT_EQ(prefix[2], static_cast<uint8_t>('S'));
    EXPECT_EQ(prefix[3], static_cast<uint8_t>('P'));

    uint32_t version = 0;
    std::memcpy(&version, prefix.data() + 4, sizeof(version));
    EXPECT_EQ(version, 4u);
}

// v3 escape hatch: explicit version=3 writes legacy gzip
TEST_F(SpzFormatTest, V3EscapeHatchRoundtrip) {
    auto original = create_test_splat(50, 1);
    const fs::path path = temp_dir / "escape_v3.spz";
    ASSERT_TRUE(save_spz(original, {.output_path = path, .version = 3}).has_value());

    const auto prefix = read_file_prefix(path, 2);
    ASSERT_GE(prefix.size(), 2u);
    EXPECT_EQ(prefix[0], 0x1f);
    EXPECT_EQ(prefix[1], 0x8b);

    const auto loaded = load_spz(path);
    ASSERT_TRUE(loaded.has_value()) << loaded.error();
    expect_splat_near(*loaded, original, SPZ_TOLERANCE);
}

// v3 fixture in -> default (v4) out round trip
TEST_F(SpzFormatTest, V3InV4OutRoundtrip) {
    const auto path_in = fixture_path("upstream_v3_sh3.spz");
    ASSERT_TRUE(fs::exists(path_in)) << path_in;

    const auto original = load_spz(path_in);
    ASSERT_TRUE(original.has_value()) << original.error();

    const fs::path path_out = temp_dir / "v3_to_v4.spz";
    ASSERT_TRUE(save_spz(*original, {.output_path = path_out}).has_value());

    const auto prefix = read_file_prefix(path_out, 4);
    ASSERT_GE(prefix.size(), 4u);
    EXPECT_EQ(std::string(prefix.begin(), prefix.begin() + 4), "NGSP");

    const auto reloaded = load_spz(path_out);
    ASSERT_TRUE(reloaded.has_value()) << reloaded.error();
    expect_splat_near(*reloaded, *original, SPZ_TOLERANCE);
}

// v4 fixture in -> default (v4) out round trip
TEST_F(SpzFormatTest, V4InV4OutRoundtrip) {
    const auto path_in = fixture_path("upstream_v4_sh3.spz");
    ASSERT_TRUE(fs::exists(path_in)) << path_in;

    const auto original = load_spz(path_in);
    ASSERT_TRUE(original.has_value()) << original.error();

    const fs::path path_out = temp_dir / "v4_to_v4.spz";
    ASSERT_TRUE(save_spz(*original, {.output_path = path_out}).has_value());

    const auto reloaded = load_spz(path_out);
    ASSERT_TRUE(reloaded.has_value()) << reloaded.error();
    expect_splat_near(*reloaded, *original, SPZ_TOLERANCE);
}

// Upstream v4 fixtures import cleanly
TEST_F(SpzFormatTest, UpstreamV4FixturesImport) {
    struct Case {
        const char* name;
        int sh_degree;
    };
    const Case cases[] = {
        {"upstream_v4_sh0.spz", 0},
        {"upstream_v4_sh1.spz", 1},
        {"upstream_v4_sh2.spz", 2},
        {"upstream_v4_sh3.spz", 3},
        {"upstream_v4_noext.spz", 3},
    };

    for (const auto& c : cases) {
        const auto path = fixture_path(c.name);
        ASSERT_TRUE(fs::exists(path)) << path;

        const auto loaded = load_spz(path);
        ASSERT_TRUE(loaded.has_value()) << c.name << ": " << loaded.error();
        EXPECT_EQ(loaded->size(), 100u) << c.name;
        EXPECT_EQ(loaded->get_max_sh_degree(), c.sh_degree) << c.name;

        expect_all_finite(loaded->means(), "means");
        expect_all_finite(loaded->sh0(), "sh0");
        expect_all_finite(loaded->scaling_raw(), "scale");
        expect_all_finite(loaded->rotation_raw(), "rotation");
        expect_all_finite(loaded->opacity_raw(), "opacity");
        if (loaded->shN().is_valid() && loaded->shN().numel() > 0) {
            expect_all_finite(loaded->shN_raw(), "shN");
        }
    }
}

// Same source clouds written as v3 and v4 must decode to matching attributes
TEST_F(SpzFormatTest, V3V4Parity) {
    for (int degree = 0; degree <= 3; ++degree) {
        const auto v3_name = std::string("upstream_v3_sh") + std::to_string(degree) + ".spz";
        const auto v4_name = std::string("upstream_v4_sh") + std::to_string(degree) + ".spz";
        const auto path_v3 = fixture_path(v3_name.c_str());
        const auto path_v4 = fixture_path(v4_name.c_str());
        ASSERT_TRUE(fs::exists(path_v3)) << path_v3;
        ASSERT_TRUE(fs::exists(path_v4)) << path_v4;

        const auto a = load_spz(path_v3);
        const auto b = load_spz(path_v4);
        ASSERT_TRUE(a.has_value()) << v3_name << ": " << a.error();
        ASSERT_TRUE(b.has_value()) << v4_name << ": " << b.error();
        expect_splat_near(*a, *b, SPZ_TOLERANCE);
    }
}

// Coordinate-system extension declaration is honoured on load
TEST_F(SpzFormatTest, CoordinateSystemExtensionHonoured) {
    const auto path_rdf = fixture_path("upstream_v4_declRDF.spz");
    const auto path_rub = fixture_path("upstream_v4_declRUB.spz");
    ASSERT_TRUE(fs::exists(path_rdf)) << path_rdf;
    ASSERT_TRUE(fs::exists(path_rub)) << path_rub;

    const auto rdf = load_spz(path_rdf);
    const auto rub = load_spz(path_rub);
    ASSERT_TRUE(rdf.has_value()) << rdf.error();
    ASSERT_TRUE(rub.has_value()) << rub.error();

    constexpr float kCoordTol = 1e-4f;
    EXPECT_EQ(rdf->size(), rub->size());
    expect_near_tensors(rdf->means(), rub->means(), kCoordTol, "means");
    expect_near_tensors(rdf->scaling_raw(), rub->scaling_raw(), kCoordTol, "scale");
    expect_near_rotations(rdf->rotation_raw(), rub->rotation_raw(), kCoordTol, "rotation");
}

// Unknown SPZ versions must be rejected without crashing
TEST_F(SpzFormatTest, UnknownVersionRejected) {
    const auto path_v99 = fixture_path("upstream_v99_bad.spz");
    ASSERT_TRUE(fs::exists(path_v99)) << path_v99;

    const auto result_v99 = load_spz(path_v99);
    EXPECT_FALSE(result_v99.has_value());
    if (!result_v99.has_value()) {
        const auto& err = result_v99.error();
        EXPECT_NE(err.find("version"), std::string::npos) << err;
        EXPECT_NE(err.find("99"), std::string::npos) << err;
    }

    const fs::path path_gzip_v9 = temp_dir / "forged_gzip_v9.spz";
    ASSERT_TRUE(write_gzipped_spz(
        path_gzip_v9,
        make_spz_header(1, 0, 12, /*version=*/9)));

    const auto result_v9 = load_spz(path_gzip_v9);
    EXPECT_FALSE(result_v9.has_value());
    if (!result_v9.has_value()) {
        const auto& err = result_v9.error();
        EXPECT_NE(err.find("version"), std::string::npos) << err;
        EXPECT_NE(err.find("9"), std::string::npos) << err;
    }
}

// v3 export must not write an NGSP container
TEST_F(SpzFormatTest, V3ExportCarriesNoExtensions) {
    auto original = create_test_splat(20, 1);
    const fs::path path = temp_dir / "v3_no_ngsp.spz";
    ASSERT_TRUE(save_spz(original, {.output_path = path, .version = 3}).has_value());

    const auto prefix = read_file_prefix(path, 4);
    ASSERT_GE(prefix.size(), 2u);
    EXPECT_EQ(prefix[0], 0x1f);
    EXPECT_EQ(prefix[1], 0x8b);
    if (prefix.size() >= 4u) {
        EXPECT_NE(std::string(prefix.begin(), prefix.begin() + 4), "NGSP");
    }

    const auto loaded = load_spz(path);
    ASSERT_TRUE(loaded.has_value()) << loaded.error();
    EXPECT_EQ(loaded->size(), original.size());
}

// Tiny-window membuf: byte reads, bulk reads that cross window boundaries, EOF.
// deserializePackedGaussians seeks (tellg / end / restore); that path is covered
// by the seekg(0) reset and the size probe below.
TEST(SpzMembufTest, TinyWindowByteAndBulkReads) {
    constexpr std::size_t n = 50;
    std::vector<uint8_t> pattern(n);
    for (std::size_t i = 0; i < n; ++i) {
        pattern[i] = static_cast<uint8_t>((i * 131u) & 0xffu);
    }
    spz::membuf buffer(pattern.data(), pattern.size(), 7);
    std::istream stream(&buffer);

    for (std::size_t i = 0; i < n; ++i) {
        const int ch = stream.get();
        ASSERT_NE(ch, std::char_traits<char>::eof()) << i;
        EXPECT_EQ(static_cast<unsigned char>(ch), pattern[i]) << i;
    }
    EXPECT_EQ(stream.get(), std::char_traits<char>::eof());

    stream.clear();
    stream.seekg(0);
    ASSERT_TRUE(stream);
    std::array<char, 20> chunk{};
    stream.read(chunk.data(), 20);
    ASSERT_EQ(stream.gcount(), 20);
    ASSERT_TRUE(stream);
    for (std::size_t i = 0; i < chunk.size(); ++i) {
        EXPECT_EQ(static_cast<unsigned char>(chunk[i]), pattern[i]) << i;
    }

    stream.clear();
    const std::streampos cur = stream.tellg();
    ASSERT_NE(cur, std::streampos(-1));
    stream.seekg(0, std::ios::end);
    const std::streampos end = stream.tellg();
    ASSERT_EQ(end, std::streampos(static_cast<std::streamoff>(n)));
    stream.seekg(cur);
    ASSERT_TRUE(stream);

    stream.clear();
    stream.seekg(static_cast<std::streamoff>(n - 5));
    ASSERT_TRUE(stream);
    std::array<char, 16> tail{};
    stream.read(tail.data(), 16);
    EXPECT_EQ(stream.gcount(), 5);
    EXPECT_TRUE(stream.eof());
    for (std::size_t i = 0; i < 5; ++i) {
        EXPECT_EQ(static_cast<unsigned char>(tail[i]), pattern[n - 5 + i]) << i;
    }
}
