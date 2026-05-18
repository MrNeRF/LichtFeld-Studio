
/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#ifdef _WIN32
#define NOMINMAX
#endif

#include <archive.h>
#include <archive_entry.h>
#include <gtest/gtest.h>
#include <nlohmann/json.hpp>
#include <pxr/base/gf/half.h>
#include <zlib.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <span>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "core/splat_data.hpp"
#include "io/exporter.hpp"
#include "io/formats/nurec_usdz.hpp"
#include "io/formats/ply.hpp"
#include "io/loaders/usd_loader.hpp"
#include "tinyply.hpp"

namespace fs = std::filesystem;
using namespace lfs::core;
using namespace lfs::io;

namespace {

#ifdef _WIN32
    using ssize_t = std::ptrdiff_t;
#endif

    using Json = nlohmann::ordered_json;

    class NurecUsdzFormatTest : public ::testing::Test {
    protected:
        static constexpr float EPSILON = 2e-3f;
        static constexpr float PLY_EPSILON = 2e-3f;

        const fs::path temp_dir = fs::temp_directory_path() / "lfs_nurec_usdz_test";

        void SetUp() override {
            fs::create_directories(temp_dir);
        }

        void TearDown() override {
            fs::remove_all(temp_dir);
        }

        void write_text_file(const fs::path& path, const std::string& contents) const {
            fs::create_directories(path.parent_path());
            std::ofstream out(path, std::ios::binary);
            ASSERT_TRUE(out.is_open()) << "Failed to open " << path;
            out << contents;
            out.close();
            ASSERT_TRUE(out.good()) << "Failed to write " << path;
        }

        std::string read_text_file(const fs::path& path) const {
            std::ifstream in(path, std::ios::binary);
            EXPECT_TRUE(in.is_open()) << "Failed to open " << path;
            return {
                std::istreambuf_iterator<char>(in),
                std::istreambuf_iterator<char>()};
        }

        void expect_existing_target_and_no_temp_files(const fs::path& output_path, const std::string& existing_contents) const {
            EXPECT_EQ(read_text_file(output_path), existing_contents);

            const auto temp_prefix = output_path.stem().string() + ".";
            for (const auto& entry : fs::directory_iterator(output_path.parent_path())) {
                if (entry.path() == output_path) {
                    continue;
                }
                EXPECT_FALSE(entry.path().filename().string().starts_with(temp_prefix))
                    << "Temporary export file was not removed: " << entry.path();
            }
        }

        static void expect_progress_completed(const std::vector<float>& updates) {
            ASSERT_FALSE(updates.empty());
            EXPECT_FLOAT_EQ(updates.front(), 0.0f);
            EXPECT_FLOAT_EQ(updates.back(), 1.0f);
            for (size_t i = 1; i < updates.size(); ++i) {
                EXPECT_GE(updates[i], updates[i - 1]) << "Progress regressed at update " << i;
            }
        }

        static SplatData create_test_splat(const size_t num_points, const int sh_degree) {
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

            auto* const means_ptr = static_cast<float*>(means.data_ptr());
            auto* const sh0_ptr = static_cast<float*>(sh0.data_ptr());
            auto* const scaling_ptr = static_cast<float*>(scaling.data_ptr());
            auto* const rotation_ptr = static_cast<float*>(rotation.data_ptr());
            auto* const opacity_ptr = static_cast<float*>(opacity.data_ptr());

            for (size_t i = 0; i < num_points; ++i) {
                means_ptr[i * 3 + 0] = static_cast<float>(i) * 0.125f - 1.25f;
                means_ptr[i * 3 + 1] = static_cast<float>(i % 7) * 0.33f;
                means_ptr[i * 3 + 2] = static_cast<float>(i % 5) * -0.42f;

                sh0_ptr[i * 3 + 0] = 0.15f * static_cast<float>(i + 1);
                sh0_ptr[i * 3 + 1] = -0.07f * static_cast<float>(i + 2);
                sh0_ptr[i * 3 + 2] = 0.11f * static_cast<float>(i + 3);

                scaling_ptr[i * 3 + 0] = -2.3f + 0.02f * static_cast<float>(i);
                scaling_ptr[i * 3 + 1] = -1.9f + 0.015f * static_cast<float>(i);
                scaling_ptr[i * 3 + 2] = -1.5f + 0.01f * static_cast<float>(i);

                const float angle = 0.08f * static_cast<float>(i);
                rotation_ptr[i * 4 + 0] = std::cos(angle * 0.5f);
                rotation_ptr[i * 4 + 1] = 0.1f * std::sin(angle * 0.25f);
                rotation_ptr[i * 4 + 2] = std::sin(angle * 0.5f);
                rotation_ptr[i * 4 + 3] = -0.05f * std::sin(angle * 0.125f);

                const float norm = std::sqrt(rotation_ptr[i * 4 + 0] * rotation_ptr[i * 4 + 0] +
                                             rotation_ptr[i * 4 + 1] * rotation_ptr[i * 4 + 1] +
                                             rotation_ptr[i * 4 + 2] * rotation_ptr[i * 4 + 2] +
                                             rotation_ptr[i * 4 + 3] * rotation_ptr[i * 4 + 3]);
                for (size_t j = 0; j < 4; ++j) {
                    rotation_ptr[i * 4 + j] /= norm;
                }

                opacity_ptr[i] = -1.8f + 0.09f * static_cast<float>(i);
            }

            if (sh_coeffs > 0) {
                auto* const shN_ptr = static_cast<float*>(shN.data_ptr());
                for (size_t i = 0; i < num_points * sh_coeffs * 3; ++i) {
                    shN_ptr[i] = 0.02f * static_cast<float>((static_cast<int>(i) % 23) - 11);
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
                1.0f);
        }

        static std::vector<float> tensor_to_vector(const Tensor& tensor) {
            const auto cpu = tensor.contiguous().to(Device::CPU);
            const auto* const ptr = static_cast<const float*>(cpu.data_ptr());
            return {ptr, ptr + cpu.numel()};
        }

        static std::expected<std::vector<uint8_t>, std::string> gunzip_bytes(const std::span<const uint8_t> input) {
            z_stream stream{};
            if (inflateInit2(&stream, 16 + MAX_WBITS) != Z_OK) {
                return std::unexpected("inflateInit2 failed");
            }

            stream.next_in = const_cast<Bytef*>(reinterpret_cast<const Bytef*>(input.data()));
            stream.avail_in = static_cast<uInt>(input.size());

            std::vector<uint8_t> output;
            std::array<uint8_t, 16384> chunk{};

            int status = Z_OK;
            while (status != Z_STREAM_END) {
                stream.next_out = chunk.data();
                stream.avail_out = static_cast<uInt>(chunk.size());
                status = inflate(&stream, Z_NO_FLUSH);
                if (status != Z_OK && status != Z_STREAM_END) {
                    inflateEnd(&stream);
                    return std::unexpected("inflate failed");
                }

                const size_t produced = chunk.size() - stream.avail_out;
                output.insert(output.end(), chunk.begin(), chunk.begin() + static_cast<std::ptrdiff_t>(produced));
            }

            inflateEnd(&stream);
            return output;
        }

        static std::unordered_map<std::string, std::vector<uint8_t>> read_usdz_entries(const fs::path& path) {
            struct archive* archive = archive_read_new();
            if (!archive) {
                throw std::runtime_error("Failed to allocate archive reader");
            }
            archive_read_support_format_zip(archive);
            archive_read_support_filter_all(archive);

#ifdef _WIN32
            if (archive_read_open_filename_w(archive, path.wstring().c_str(), 10240) != ARCHIVE_OK) {
                const std::string error = archive_error_string(archive) ? archive_error_string(archive) : "unknown error";
                archive_read_free(archive);
                throw std::runtime_error(error);
            }
#else
            if (archive_read_open_filename(archive, path.c_str(), 10240) != ARCHIVE_OK) {
                const std::string error = archive_error_string(archive) ? archive_error_string(archive) : "unknown error";
                archive_read_free(archive);
                throw std::runtime_error(error);
            }
#endif

            std::unordered_map<std::string, std::vector<uint8_t>> files;
            struct archive_entry* entry = nullptr;
            while (archive_read_next_header(archive, &entry) == ARCHIVE_OK) {
                const char* name = archive_entry_pathname(entry);
                if (!name) {
                    archive_read_free(archive);
                    throw std::runtime_error("Archive entry missing name");
                }
                const size_t size = static_cast<size_t>(archive_entry_size(entry));
                std::vector<uint8_t> data(size);
                if (size > 0) {
                    const ssize_t read = archive_read_data(archive, data.data(), size);
                    if (read != static_cast<ssize_t>(size)) {
                        archive_read_free(archive);
                        throw std::runtime_error("Failed to read archive entry");
                    }
                }
                files.emplace(name, std::move(data));
            }

            archive_read_free(archive);
            return files;
        }

        static Json read_nurec_payload(const fs::path& path) {
            const auto files = read_usdz_entries(path);
            if (!files.contains("default.usda") || !files.contains("gauss.usda")) {
                throw std::runtime_error("USDZ archive is missing required USDA layers");
            }

            auto nurec_it = std::find_if(files.begin(), files.end(), [](const auto& item) {
                return fs::path(item.first).extension() == ".nurec";
            });
            if (nurec_it == files.end()) {
                throw std::runtime_error("USDZ archive is missing a .nurec payload");
            }

            auto decompressed = gunzip_bytes(nurec_it->second);
            if (!decompressed) {
                throw std::runtime_error(decompressed.error());
            }
            return Json::from_msgpack(*decompressed, true, true);
        }

        static std::vector<float> decode_half_field(const Json& state_dict, const std::string& key) {
            const auto& binary = state_dict.at(key).get_binary();
            std::vector<float> result(binary.size() / sizeof(uint16_t), 0.0f);
            for (size_t i = 0; i < result.size(); ++i) {
                const uint16_t bits = static_cast<uint16_t>(binary[i * 2 + 0]) |
                                      (static_cast<uint16_t>(binary[i * 2 + 1]) << 8u);
                pxr::GfHalf half_value;
                half_value.setBits(bits);
                result[i] = static_cast<float>(half_value);
            }
            return result;
        }

        static int64_t decode_int64_field(const Json& state_dict, const std::string& key) {
            const auto& binary = state_dict.at(key).get_binary();
            int64_t value = 0;
            std::memcpy(&value, binary.data(), sizeof(int64_t));
            return value;
        }

        static std::vector<float> read_ply_float_property(const fs::path& path, const std::string& property_name) {
            std::ifstream file(path, std::ios::binary);
            if (!file.is_open()) {
                throw std::runtime_error("Failed to open PLY file");
            }

            tinyply::PlyFile ply;
            ply.parse_header(file);
            auto property = ply.request_properties_from_element("vertex", {property_name});
            ply.read(file);

            std::vector<float> values(property->count);
            std::memcpy(values.data(), property->buffer.get(), values.size() * sizeof(float));
            return values;
        }

        static void expect_vectors_near(const std::vector<float>& actual,
                                        const std::vector<float>& expected,
                                        const float tolerance) {
            ASSERT_EQ(actual.size(), expected.size());
            for (size_t i = 0; i < actual.size(); ++i) {
                EXPECT_NEAR(actual[i], expected[i], tolerance) << "Mismatch at element " << i;
            }
        }
    };

    TEST_F(NurecUsdzFormatTest, RoundtripPreservesValuesWithinHalfPrecision) {
        auto original = create_test_splat(32, 2);
        const fs::path usdz_path = temp_dir / "roundtrip.usdz";
        std::vector<float> updates;

        ASSERT_TRUE(save_nurec_usdz(original, {.output_path = usdz_path,
                                               .progress_callback = [&](float progress, const std::string&) {
                                                   updates.push_back(progress);
                                                   return true;
                                               }})
                        .has_value());
        ASSERT_TRUE(fs::exists(usdz_path));
        expect_progress_completed(updates);

        auto loaded_result = load_nurec_usdz(usdz_path);
        ASSERT_TRUE(loaded_result.has_value()) << loaded_result.error();
        const auto& loaded = *loaded_result;

        EXPECT_EQ(loaded.size(), original.size());
        EXPECT_EQ(loaded.get_max_sh_degree(), original.get_max_sh_degree());

        expect_vectors_near(tensor_to_vector(loaded.means()), tensor_to_vector(original.means()), EPSILON);
        expect_vectors_near(tensor_to_vector(loaded.sh0()), tensor_to_vector(original.sh0()), EPSILON);
        expect_vectors_near(tensor_to_vector(loaded.scaling_raw()), tensor_to_vector(original.scaling_raw()), EPSILON);
        expect_vectors_near(tensor_to_vector(loaded.opacity_raw()), tensor_to_vector(original.opacity_raw()), EPSILON);
        expect_vectors_near(tensor_to_vector(loaded.shN()), tensor_to_vector(original.shN()), EPSILON);

        const auto loaded_rotations = tensor_to_vector(loaded.rotation_raw());
        const auto original_rotations = tensor_to_vector(original.get_rotation());
        ASSERT_EQ(loaded_rotations.size(), original_rotations.size());
        for (size_t i = 0; i < original.size(); ++i) {
            const float dot = original_rotations[i * 4 + 0] * loaded_rotations[i * 4 + 0] +
                              original_rotations[i * 4 + 1] * loaded_rotations[i * 4 + 1] +
                              original_rotations[i * 4 + 2] * loaded_rotations[i * 4 + 2] +
                              original_rotations[i * 4 + 3] * loaded_rotations[i * 4 + 3];
            EXPECT_NEAR(std::abs(dot), 1.0f, EPSILON) << "Rotation mismatch at point " << i;
        }
    }

    TEST_F(NurecUsdzFormatTest, SaveCancellationKeepsExistingTarget) {
        auto original = create_test_splat(8, 1);
        const fs::path usdz_path = temp_dir / "keep_existing_on_cancel.usdz";
        const std::string existing_contents = "existing usdz data";
        write_text_file(usdz_path, existing_contents);

        auto result = save_nurec_usdz(original, {.output_path = usdz_path,
                                                 .progress_callback = [](float progress, const std::string&) {
                                                     return progress < 1.0f;
                                                 }});
        ASSERT_FALSE(result.has_value());
        EXPECT_EQ(result.error().code, ErrorCode::CANCELLED);
        expect_existing_target_and_no_temp_files(usdz_path, existing_contents);
    }

    TEST_F(NurecUsdzFormatTest, UsdLoaderLoadsNurecUsdz) {
        auto original = create_test_splat(12, 1);
        const fs::path usdz_path = temp_dir / "loader.usdz";

        ASSERT_TRUE(save_nurec_usdz(original, {.output_path = usdz_path}).has_value());

        USDLoader loader;
        auto result = loader.load(usdz_path);
        ASSERT_TRUE(result.has_value()) << result.error().format();
        EXPECT_EQ(result->loader_used, "NuRec USDZ");

        const auto* const data = std::get_if<std::shared_ptr<SplatData>>(&result->data);
        ASSERT_NE(data, nullptr);
        ASSERT_TRUE(*data);
        EXPECT_EQ((*data)->size(), original.size());
        EXPECT_EQ((*data)->get_max_sh_degree(), original.get_max_sh_degree());
    }

    TEST_F(NurecUsdzFormatTest, ExportMatchesPlyToUsdPayloadLayout) {
        auto original = create_test_splat(6, 3);
        const fs::path usdz_path = temp_dir / "payload_layout.usdz";

        ASSERT_TRUE(save_nurec_usdz(original, {.output_path = usdz_path}).has_value());

        const auto payload = read_nurec_payload(usdz_path);
        const auto& root = payload.at("nre_data");
        const auto& state_dict = root.at("state_dict");

        EXPECT_EQ(root.at("version"), "0.2.576");
        EXPECT_EQ(root.at("model"), "nre");
        EXPECT_EQ(root.at("config").at("layers").at("gaussians").at("name"), "sh-gaussians");
        EXPECT_EQ(root.at("config").at("layers").at("gaussians").at("device"), "cuda");
        EXPECT_EQ(root.at("config").at("layers").at("gaussians").at("particle").at("radiance_sph_degree"), 3);
        EXPECT_EQ(decode_int64_field(state_dict, ".gaussians_nodes.gaussians.n_active_features"), 3);

        expect_vectors_near(
            decode_half_field(state_dict, ".gaussians_nodes.gaussians.positions"),
            tensor_to_vector(original.means()),
            EPSILON);
        expect_vectors_near(
            decode_half_field(state_dict, ".gaussians_nodes.gaussians.scales"),
            tensor_to_vector(original.scaling_raw()),
            EPSILON);
        expect_vectors_near(
            decode_half_field(state_dict, ".gaussians_nodes.gaussians.densities"),
            tensor_to_vector(original.opacity_raw()),
            EPSILON);

        const auto expected_albedo = tensor_to_vector(original.sh0().reshape({static_cast<int>(original.size()), 3}));
        expect_vectors_near(
            decode_half_field(state_dict, ".gaussians_nodes.gaussians.features_albedo"),
            expected_albedo,
            EPSILON);

        const auto expected_specular = tensor_to_vector(original.shN().reshape({static_cast<int>(original.size()), static_cast<int>(original.shN().size(1) * 3)}));
        expect_vectors_near(
            decode_half_field(state_dict, ".gaussians_nodes.gaussians.features_specular"),
            expected_specular,
            EPSILON);

        expect_vectors_near(
            decode_half_field(state_dict, ".gaussians_nodes.gaussians.rotations"),
            tensor_to_vector(original.get_rotation()),
            EPSILON);
    }

    TEST_F(NurecUsdzFormatTest, PlyRoundtripAfterImportMatchesOriginalProperties) {
        auto original = create_test_splat(10, 2);
        const fs::path original_ply = temp_dir / "original.ply";
        const fs::path usdz_path = temp_dir / "roundtrip_ply.usdz";
        const fs::path loaded_ply = temp_dir / "loaded.ply";

        ASSERT_TRUE(save_ply(original, {.output_path = original_ply, .binary = true}).has_value());
        ASSERT_TRUE(save_nurec_usdz(original, {.output_path = usdz_path}).has_value());

        auto loaded_result = load_nurec_usdz(usdz_path);
        ASSERT_TRUE(loaded_result.has_value()) << loaded_result.error();
        ASSERT_TRUE(save_ply(*loaded_result, {.output_path = loaded_ply, .binary = true}).has_value());

        for (const auto& property_name : get_ply_attribute_names(original)) {
            const auto original_values = read_ply_float_property(original_ply, property_name);
            const auto loaded_values = read_ply_float_property(loaded_ply, property_name);
            expect_vectors_near(loaded_values, original_values, PLY_EPSILON);
        }
    }

} // namespace
