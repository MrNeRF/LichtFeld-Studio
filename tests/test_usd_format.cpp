/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <iterator>
#include <string>
#include <vector>

#include "core/splat_data.hpp"
#include "io/exporter.hpp"
#include "io/formats/usd.hpp"
#include "io/formats/usd_flat/crate.hpp"
#include "io/formats/usd_flat/flat_stage.hpp"
#include "io/formats/usd_flat/half.hpp"
#include "io/formats/usd_flat/usda_read.hpp"
#include "io/formats/usd_flat/usda_write.hpp"
#include "io/formats/usd_flat/usdz.hpp"
#include "io/loaders/usd_loader.hpp"
#include "io/project_document.hpp"

namespace fs = std::filesystem;
using namespace lfs::core;
using namespace lfs::io;
using namespace lfs::io::usd_flat;

namespace {

    std::uint16_t reference_float_to_half(const float value) {
#if defined(__FLT16_MANT_DIG__)
        const _Float16 converted = static_cast<_Float16>(value);
        std::uint16_t bits = 0;
        std::memcpy(&bits, &converted, sizeof(bits));
        return bits;
#else
        std::uint32_t bits = 0;
        std::memcpy(&bits, &value, sizeof(bits));
        const std::uint32_t sign = (bits >> 16u) & 0x8000u;
        const std::uint32_t exponent = (bits >> 23u) & 0xffu;
        const std::uint32_t fraction = bits & 0x7fffffu;
        if (exponent == 0xffu)
            return static_cast<std::uint16_t>(sign | 0x7c00u | (fraction ? 0x0200u : 0));
        if (exponent < 102u)
            return static_cast<std::uint16_t>(sign);
        const auto round = [](const std::uint32_t mantissa, const unsigned shift) {
            const std::uint32_t truncated = mantissa >> shift;
            const std::uint32_t remainder = mantissa & ((1u << shift) - 1u);
            const std::uint32_t halfway = 1u << (shift - 1u);
            return truncated + (remainder > halfway || (remainder == halfway && (truncated & 1u)));
        };
        if (exponent < 113u)
            return static_cast<std::uint16_t>(sign | round(fraction | 0x800000u, 126u - exponent));
        if (exponent > 142u)
            return static_cast<std::uint16_t>(sign | 0x7c00u);
        const auto rounded = round(fraction | 0x800000u, 13u);
        if (rounded == 0x400u) {
            const auto half_exponent = exponent - 111u;
            return static_cast<std::uint16_t>(sign | (half_exponent << 10u));
        }
        const auto half_exponent = exponent - 112u;
        return static_cast<std::uint16_t>(sign | (half_exponent << 10u) | rounded);
#endif
    }

    class UsdFormatTest : public ::testing::Test {
    protected:
        static constexpr float EPSILON = 1e-4f;

        const fs::path temp_dir = fs::temp_directory_path() / "lfs_usd_test";

        void SetUp() override { fs::create_directories(temp_dir); }

        void TearDown() override { fs::remove_all(temp_dir); }

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
            return {std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
        }

        void expect_existing_target_and_no_temp_files(const fs::path& output_path, const std::string& existing_contents) const {
            EXPECT_EQ(read_text_file(output_path), existing_contents);
            const auto temp_prefix = output_path.stem().string() + ".";
            for (const auto& entry : fs::directory_iterator(output_path.parent_path())) {
                if (entry.path() != output_path) {
                    EXPECT_FALSE(entry.path().filename().string().starts_with(temp_prefix))
                        << "Temporary export file was not removed: " << entry.path();
                }
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

        static FlatPrim positions_prim(const std::string& path, const std::vector<float>& positions,
                                       const std::string& type = "ParticleField3DGaussianSplat") {
            FlatPrim prim;
            prim.path = path;
            prim.type_name = type;
            prim.local_transform = identity_matrix();
            prim.attributes.emplace("positions", FlatAttribute{"point3f[]", positions, 3, true});
            return prim;
        }

        static void write_stage(const fs::path& path, FlatStage stage) {
            auto result = write_usda(stage, path);
            ASSERT_TRUE(result.has_value()) << lfs::format_for_developer(result.error());
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
                means_ptr[i * 3 + 0] = static_cast<float>(i) * 0.25f - 1.0f;
                means_ptr[i * 3 + 1] = static_cast<float>(i % 5) * 0.5f;
                means_ptr[i * 3 + 2] = static_cast<float>(i % 3) * -0.75f;
                sh0_ptr[i * 3 + 0] = 0.1f * static_cast<float>(i + 1);
                sh0_ptr[i * 3 + 1] = -0.05f * static_cast<float>(i + 2);
                sh0_ptr[i * 3 + 2] = 0.08f * static_cast<float>(i + 3);
                scaling_ptr[i * 3 + 0] = -2.0f + 0.03f * static_cast<float>(i);
                scaling_ptr[i * 3 + 1] = -1.5f + 0.02f * static_cast<float>(i);
                scaling_ptr[i * 3 + 2] = -1.0f + 0.01f * static_cast<float>(i);
                const float angle = 0.1f * static_cast<float>(i);
                rotation_ptr[i * 4 + 0] = std::cos(angle * 0.5f);
                rotation_ptr[i * 4 + 1] = 0.0f;
                rotation_ptr[i * 4 + 2] = std::sin(angle * 0.5f);
                rotation_ptr[i * 4 + 3] = 0.0f;
                opacity_ptr[i] = -1.5f + 0.1f * static_cast<float>(i);
            }
            if (sh_coeffs > 0) {
                auto* const shN_ptr = static_cast<float*>(shN.data_ptr());
                for (size_t i = 0; i < num_points * sh_coeffs * 3; ++i) {
                    shN_ptr[i] = 0.01f * static_cast<float>((static_cast<int>(i) % 17) - 8);
                }
            }
            return SplatData(sh_degree, std::move(means), std::move(sh0), std::move(shN), std::move(scaling),
                             std::move(rotation), std::move(opacity), 0.5f);
        }
    };

    TEST_F(UsdFormatTest, RoundtripPreservesValues) {
        auto original = create_test_splat(16, 2);
        const fs::path usd_path = temp_dir / "roundtrip.usda";
        std::vector<float> updates;
        ASSERT_TRUE(save_usd(original, {.output_path = usd_path,
                                        .progress_callback = [&](float progress, const std::string&) {
                                            updates.push_back(progress);
                                            return true;
                                        }})
                        .has_value());
        expect_progress_completed(updates);
        auto loaded_result = load_usd(usd_path);
        ASSERT_TRUE(loaded_result.has_value()) << loaded_result.error();
        const auto& loaded = *loaded_result;
        EXPECT_EQ(loaded.size(), original.size());
        EXPECT_EQ(loaded.get_max_sh_degree(), original.get_max_sh_degree());

        const auto orig_means = original.means().contiguous().to(Device::CPU);
        const auto load_means = loaded.means().contiguous().to(Device::CPU);
        const auto orig_sh0 = original.sh0().contiguous().to(Device::CPU);
        const auto load_sh0 = loaded.sh0().contiguous().to(Device::CPU);
        const auto orig_scaling = original.scaling_raw().contiguous().to(Device::CPU);
        const auto load_scaling = loaded.scaling_raw().contiguous().to(Device::CPU);
        const auto orig_rotation = original.rotation_raw().contiguous().to(Device::CPU);
        const auto load_rotation = loaded.rotation_raw().contiguous().to(Device::CPU);
        const auto orig_opacity = original.opacity_raw().contiguous().to(Device::CPU);
        const auto load_opacity = loaded.opacity_raw().contiguous().to(Device::CPU);
        const auto orig_shN = original.shN_canonical_cpu().contiguous();
        const auto load_shN = loaded.shN_canonical_cpu().contiguous();
        const auto* const orig_means_ptr = static_cast<const float*>(orig_means.data_ptr());
        const auto* const load_means_ptr = static_cast<const float*>(load_means.data_ptr());
        const auto* const orig_sh0_ptr = static_cast<const float*>(orig_sh0.data_ptr());
        const auto* const load_sh0_ptr = static_cast<const float*>(load_sh0.data_ptr());
        const auto* const orig_scaling_ptr = static_cast<const float*>(orig_scaling.data_ptr());
        const auto* const load_scaling_ptr = static_cast<const float*>(load_scaling.data_ptr());
        const auto* const orig_rotation_ptr = static_cast<const float*>(orig_rotation.data_ptr());
        const auto* const load_rotation_ptr = static_cast<const float*>(load_rotation.data_ptr());
        const auto* const orig_opacity_ptr = static_cast<const float*>(orig_opacity.data_ptr());
        const auto* const load_opacity_ptr = static_cast<const float*>(load_opacity.data_ptr());
        const auto* const orig_shN_ptr = static_cast<const float*>(orig_shN.data_ptr());
        const auto* const load_shN_ptr = static_cast<const float*>(load_shN.data_ptr());
        for (size_t i = 0; i < original.size() * 3; ++i) {
            EXPECT_NEAR(load_means_ptr[i], orig_means_ptr[i], EPSILON);
            EXPECT_NEAR(load_scaling_ptr[i], orig_scaling_ptr[i], EPSILON);
            EXPECT_NEAR(load_sh0_ptr[i], orig_sh0_ptr[i], EPSILON);
        }
        for (size_t i = 0; i < original.size(); ++i) {
            EXPECT_NEAR(load_opacity_ptr[i], orig_opacity_ptr[i], EPSILON);
            const float dot = orig_rotation_ptr[i * 4 + 0] * load_rotation_ptr[i * 4 + 0] +
                              orig_rotation_ptr[i * 4 + 1] * load_rotation_ptr[i * 4 + 1] +
                              orig_rotation_ptr[i * 4 + 2] * load_rotation_ptr[i * 4 + 2] +
                              orig_rotation_ptr[i * 4 + 3] * load_rotation_ptr[i * 4 + 3];
            EXPECT_NEAR(std::abs(dot), 1.0f, EPSILON);
        }
        ASSERT_TRUE(loaded.shN().is_valid());
        for (size_t i = 0; i < original.size() * original.active_sh_coeffs_rest() * 3; ++i) {
            EXPECT_NEAR(load_shN_ptr[i], orig_shN_ptr[i], EPSILON);
        }
    }

    TEST_F(UsdFormatTest, RejectsUnsupportedExtension) {
        auto result = save_usd(create_test_splat(4, 0), {.output_path = temp_dir / "packed.usdz"});
        ASSERT_FALSE(result.has_value());
        EXPECT_EQ(result.error().code, ErrorCode::UNSUPPORTED_FORMAT);
    }

    TEST_F(UsdFormatTest, SaveCancellationKeepsExistingTarget) {
        const fs::path usd_path = temp_dir / "keep_existing_on_cancel.usda";
        write_text_file(usd_path, "existing usd data");
        auto result = save_usd(create_test_splat(4, 0), {.output_path = usd_path,
                                                         .progress_callback = [](float progress, const std::string&) {
                                                             return progress < 1.0f;
                                                         }});
        ASSERT_FALSE(result.has_value());
        EXPECT_EQ(result.error().code, ErrorCode::CANCELLED);
        expect_existing_target_and_no_temp_files(usd_path, "existing usd data");
    }

    TEST_F(UsdFormatTest, ExportAuthorsStageMetricsAndExtent) {
        const fs::path usd_path = temp_dir / "metadata.usda";
        ASSERT_TRUE(save_usd(create_test_splat(4, 0), {.output_path = usd_path}).has_value());
        auto stage = read_usda(usd_path);
        ASSERT_TRUE(stage.has_value()) << lfs::format_for_developer(stage.error());
        EXPECT_EQ(stage->up_axis, "Y");
        EXPECT_DOUBLE_EQ(stage->meters_per_unit, 1.0);
        ASSERT_EQ(stage->default_prim, "/GaussianSplats");
        const auto& extent = stage->prims.front().attributes.at("extent").values;
        ASSERT_EQ(extent.size(), 6u);
        EXPECT_FLOAT_EQ(extent[0], -1.0f);
        EXPECT_FLOAT_EQ(extent[1], 0.0f);
        EXPECT_FLOAT_EQ(extent[2], -1.5f);
        EXPECT_FLOAT_EQ(extent[3], -0.25f);
        EXPECT_FLOAT_EQ(extent[4], 1.5f);
        EXPECT_FLOAT_EQ(extent[5], 0.0f);
    }

    TEST_F(UsdFormatTest, ImportPrefersDefaultPrimSubtreeAndAppliesStageUnits) {
        const fs::path usd_path = temp_dir / "default_prim.usda";
        FlatStage stage;
        stage.default_prim = "/Asset";
        stage.meters_per_unit = 2.0;
        stage.prims.push_back(FlatPrim{"/Asset", "Xform", {}, identity_matrix()});
        stage.prims.push_back(positions_prim("/Asset/Chosen", {1.0f, 2.0f, 3.0f}));
        stage.prims.push_back(positions_prim("/Other", {100.0f, 200.0f, 300.0f}));
        write_stage(usd_path, std::move(stage));
        auto loaded_result = load_usd(usd_path);
        ASSERT_TRUE(loaded_result.has_value()) << loaded_result.error();
        const auto means = loaded_result->means().contiguous().to(Device::CPU);
        const auto* const means_ptr = static_cast<const float*>(means.data_ptr());
        EXPECT_FLOAT_EQ(means_ptr[0], 2.0f);
        EXPECT_FLOAT_EQ(means_ptr[1], 4.0f);
        EXPECT_FLOAT_EQ(means_ptr[2], 6.0f);
    }

    TEST_F(UsdFormatTest, RejectsAmbiguousMultiPrimStagesWithoutDefaultPrim) {
        const fs::path usd_path = temp_dir / "ambiguous.usda";
        FlatStage stage;
        stage.prims.push_back(positions_prim("/First", {1.0f, 0.0f, 0.0f}));
        stage.prims.push_back(positions_prim("/Second", {2.0f, 0.0f, 0.0f}));
        write_stage(usd_path, std::move(stage));
        auto loaded_result = load_usd(usd_path);
        ASSERT_FALSE(loaded_result.has_value());
        EXPECT_NE(loaded_result.error().find("multiple OpenUSD ParticleField prims"), std::string::npos);
    }

    TEST_F(UsdFormatTest, ShortShPayloadFallsBackToDegreeZero) {
        const fs::path usd_path = temp_dir / "short_sh.usda";
        FlatStage stage;
        stage.default_prim = "/GaussianSplats";
        auto prim = positions_prim("/GaussianSplats", {1.0f, 2.0f, 3.0f});
        prim.attributes.emplace("radiance:sphericalHarmonicsDegree", FlatAttribute{"int", {3.0f}, 1, true});
        prim.attributes.emplace("radiance:sphericalHarmonicsCoefficients", FlatAttribute{"float3[]", {0.25f, -0.5f, 0.75f}, 3, true});
        stage.prims.push_back(std::move(prim));
        write_stage(usd_path, std::move(stage));
        auto loaded_result = load_usd(usd_path);
        ASSERT_TRUE(loaded_result.has_value()) << loaded_result.error();
        EXPECT_EQ(loaded_result->get_max_sh_degree(), 0);
        const auto sh0 = loaded_result->sh0().contiguous().to(Device::CPU);
        const auto* const sh0_ptr = static_cast<const float*>(sh0.data_ptr());
        EXPECT_FLOAT_EQ(sh0_ptr[0], 0.0f);
        EXPECT_FLOAT_EQ(sh0_ptr[1], 0.0f);
        EXPECT_FLOAT_EQ(sh0_ptr[2], 0.0f);
    }

    TEST_F(UsdFormatTest, LoaderValidateOnlyUsesLightweightStageValidation) {
        const fs::path usd_path = temp_dir / "validate_only.usda";
        FlatStage stage;
        stage.default_prim = "/GaussianSplats";
        stage.prims.push_back(positions_prim("/GaussianSplats", {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}));
        write_stage(usd_path, std::move(stage));
        USDLoader loader;
        auto result = loader.load(usd_path, {.validate_only = true});
        ASSERT_TRUE(result.has_value()) << result.error().message;
        EXPECT_EQ(result->loader_used, "OpenUSD");
        const auto splat_data = std::get_if<std::shared_ptr<SplatData>>(&result->data);
        ASSERT_NE(splat_data, nullptr);
        EXPECT_EQ(*splat_data, nullptr);
        EXPECT_EQ(result->scene_center.numel(), 3);
    }

    TEST_F(UsdFormatTest, LoaderGeoreferenceCapturesIntoProjectAndSurvivesReload) {
        const fs::path usd_path = temp_dir / "georeference.usda";
        ASSERT_TRUE(save_usd(create_test_splat(2, 0), {.output_path = usd_path}).has_value());
        auto stage = read_usda(usd_path);
        ASSERT_TRUE(stage.has_value()) << lfs::format_for_developer(stage.error());
        constexpr double METERS_PER_UNIT = 0.3048;
        stage->meters_per_unit = METERS_PER_UNIT;
        ASSERT_TRUE(write_usda(*stage, usd_path).has_value());

        USDLoader loader;
        auto loaded = loader.load(usd_path);
        ASSERT_TRUE(loaded.has_value()) << loaded.error().message;
        ASSERT_TRUE(loaded->georeference);
        EXPECT_DOUBLE_EQ(loaded->georeference->world_unit_scale, METERS_PER_UNIT);
        const auto project_uuid = Uuid::from_string("76000000-0000-4000-8000-000000000001");
        ASSERT_TRUE(project_uuid);
        auto document = lfs::io::project::ProjectDocument::create(*project_uuid, 100);
        ASSERT_TRUE(document) << lfs::format_for_developer(document.error());
        auto captured = document->capture_georeference(*loaded);
        ASSERT_TRUE(captured) << lfs::format_for_developer(captured.error());
        const fs::path project_path = temp_dir / "georeference.licht";
        auto saved = document->save(project_path, lfs::io::project::ProjectDocumentSaveOptions{
                                                      .commit = {.kind = lfs::io::project::CommitKind::Explicit,
                                                                 .commit_uuid = *Uuid::from_string("76000000-0000-4000-8000-000000000002"),
                                                                 .snapshot_uuid = *Uuid::from_string("76000000-0000-4000-8000-000000000003"),
                                                                 .wallclock_unix_ns = 200},
                                                      .file_uuid = *Uuid::from_string("76000000-0000-4000-8000-000000000004"),
                                                      .index_compression = lfs::io::project::IndexCompression::StoredForDeterministicTests,
                                                      .disk_reserve_bytes = 0});
        ASSERT_TRUE(saved) << lfs::format_for_developer(saved.error());
        auto reopened = lfs::io::project::ProjectDocument::open(project_path);
        ASSERT_TRUE(reopened) << lfs::format_for_developer(reopened.error());
        const auto georeference = reopened->project().georeference();
        ASSERT_TRUE(georeference);
        EXPECT_DOUBLE_EQ(georeference->world_unit_scale, METERS_PER_UNIT);
        EXPECT_EQ(georeference->world_origin, loaded->georeference->world_origin);
        EXPECT_EQ(georeference->world_origin_provenance, lfs::io::project::WorldOriginProvenance::Import);
    }

    TEST_F(UsdFormatTest, CrateRoundtripPreservesArraysAndMetadata) {
        FlatStage stage;
        stage.default_prim = "/GaussianSplats";
        stage.up_axis = "Y";
        stage.meters_per_unit = 0.3048;
        stage.custom_layer_data = {{"creator", "LichtFeld Studio"}, {"source", "test"}};
        FlatPrim prim = positions_prim("/GaussianSplats", {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
        prim.attributes.emplace("orientations", FlatAttribute{"quatf[]", {1, 0, 0, 0, 0.5f, 0.5f, 0.5f, 0.5f}, 4, true});
        prim.attributes.emplace("scales", FlatAttribute{"float3[]", {1, 2, 3, 4, 5, 6}, 3, true});
        prim.attributes.emplace("opacities", FlatAttribute{"float[]", {0.2f, 0.8f}, 1, true});
        prim.attributes.emplace("radiance:sphericalHarmonicsCoefficients",
                                FlatAttribute{"float3[]", {0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f}, 3, true});
        prim.attributes.emplace("radiance:sphericalHarmonicsDegree", FlatAttribute{"int", {1}, 1, true});
        prim.attributes.emplace("extent", FlatAttribute{"float3[]", {-1, -2, -3, 4, 5, 6}, 3, true});
        stage.prims.push_back(prim);
        const auto path = temp_dir / "roundtrip.usdc";
        ASSERT_TRUE(write_usdc(stage, path).has_value());
        auto loaded = read_usdc(path);
        ASSERT_TRUE(loaded.has_value()) << lfs::format_for_developer(loaded.error());
        ASSERT_EQ(loaded->default_prim, stage.default_prim);
        ASSERT_EQ(loaded->custom_layer_data, stage.custom_layer_data);
        ASSERT_EQ(loaded->prims.size(), 1u);
        for (const auto& item : prim.attributes) {
            const auto& expected = item.second.values;
            const auto& actual = loaded->prims.front().attributes.at(item.first).values;
            ASSERT_EQ(actual.size(), expected.size()) << item.first;
            for (size_t i = 0; i < expected.size(); ++i) {
                EXPECT_FLOAT_EQ(actual[i], expected[i]) << item.first << "[" << i << "]";
            }
        }
    }

    TEST_F(UsdFormatTest, HalfFixtureWidensToFloat) {
        const auto path = fs::path(PROJECT_ROOT_PATH) / "tests/data/usd/half_variants.usdc";
        auto stage = read_usdc(path);
        ASSERT_TRUE(stage.has_value()) << lfs::format_for_developer(stage.error());
        const auto& attrs = stage->prims.back().attributes;
        EXPECT_FLOAT_EQ(attrs.at("positionsh").values[0], -0.5f);
        EXPECT_FLOAT_EQ(attrs.at("scalesh").values[4], 1.099609375f);
        EXPECT_FLOAT_EQ(attrs.at("opacitiesh").values[1], 0.5f);
        EXPECT_FLOAT_EQ(attrs.at("orientationsh").values[0], 1.0f);
    }

    TEST_F(UsdFormatTest, HalfFixtureUsdaWidensToFloat) {
        const auto usda_path = fs::path(PROJECT_ROOT_PATH) / "tests/data/usd/half_variants.usda";
        const auto usdc_path = fs::path(PROJECT_ROOT_PATH) / "tests/data/usd/half_variants.usdc";
        auto usda = read_usda(usda_path);
        ASSERT_TRUE(usda.has_value()) << lfs::format_for_developer(usda.error());
        auto usdc = read_usdc(usdc_path);
        ASSERT_TRUE(usdc.has_value()) << lfs::format_for_developer(usdc.error());
        ASSERT_FALSE(usda->prims.empty());
        ASSERT_FALSE(usdc->prims.empty());

        const auto& usda_attributes = usda->prims.back().attributes;
        const auto& usdc_attributes = usdc->prims.back().attributes;
        for (const std::string_view name : {"positionsh", "scalesh", "opacitiesh", "orientationsh",
                                            "radiance:sphericalHarmonicsCoefficientsh"}) {
            ASSERT_TRUE(usda_attributes.contains(std::string{name})) << name;
            ASSERT_TRUE(usdc_attributes.contains(std::string{name})) << name;
            const auto& usda_attribute = usda_attributes.at(std::string{name});
            const auto& usdc_attribute = usdc_attributes.at(std::string{name});
            ASSERT_EQ(usda_attribute.components, usdc_attribute.components) << name;
            ASSERT_EQ(usda_attribute.values.size(), usdc_attribute.values.size()) << name;
            for (std::size_t index = 0; index < usda_attribute.values.size(); ++index) {
                EXPECT_FLOAT_EQ(usda_attribute.values[index], usdc_attribute.values[index]) << name << "[" << index << "]";
            }
        }

        auto loaded = load_usd(usda_path);
        ASSERT_TRUE(loaded.has_value()) << loaded.error();
    }

    TEST_F(UsdFormatTest, HalfRoleFixtureParses) {
        const auto path = fs::path(PROJECT_ROOT_PATH) / "tests/data/usd/half_roles.usda";
        auto stage = read_usda(path);
        ASSERT_TRUE(stage.has_value()) << lfs::format_for_developer(stage.error());
        ASSERT_FALSE(stage->prims.empty());
        const auto& attributes = stage->prims.back().attributes;
        ASSERT_TRUE(attributes.contains("positionsh"));
        auto loaded = load_usd(path);
        ASSERT_TRUE(loaded.has_value()) << loaded.error();
    }

    TEST(UsdHalfTest, RoundTripAndRoundingMatchIEEE754) {
        for (std::uint32_t bits = 0; bits < 0x10000u; ++bits) {
            if ((bits & 0x7c00u) != 0x7c00u || (bits & 0x03ffu) == 0) {
                EXPECT_EQ(float_to_half(half_to_float(static_cast<std::uint16_t>(bits))), bits)
                    << "half bits 0x" << std::hex << bits;
            }
        }
        for (std::uint32_t bits = 0; bits <= 0x7f800000u; bits += 0x2000u) {
            for (const std::uint32_t sign : {0u, 0x80000000u}) {
                std::uint32_t signed_bits = bits | sign;
                float value = 0.0f;
                std::memcpy(&value, &signed_bits, sizeof(value));
                EXPECT_EQ(float_to_half(value), reference_float_to_half(value))
                    << "float bits 0x" << std::hex << signed_bits;
            }
        }
        for (std::uint32_t exponent = 0; exponent < 255; ++exponent) {
            for (const std::uint32_t fraction : {0x1000u, 0x5000u, 0x7f7000u}) {
                const std::uint32_t bits = (exponent << 23u) | fraction;
                float value = 0.0f;
                std::memcpy(&value, &bits, sizeof(value));
                if (std::isnan(value)) {
                    EXPECT_TRUE((float_to_half(value) & 0x7c00u) == 0x7c00u);
                    EXPECT_NE(float_to_half(value) & 0x03ffu, 0u);
                } else {
                    EXPECT_EQ(float_to_half(value), reference_float_to_half(value));
                }
            }
        }
    }

    TEST_F(UsdFormatTest, XformFixtureEvaluatesUSDAAndUSDC) {
        const auto root = fs::path(PROJECT_ROOT_PATH) / "tests/data/usd/";
        auto usda = read_usda(root / "xform_stage.usda");
        auto usdc = read_usdc(root / "xform_stage.usdc");
        ASSERT_TRUE(usda.has_value()) << lfs::format_for_developer(usda.error());
        ASSERT_TRUE(usdc.has_value()) << lfs::format_for_developer(usdc.error());
        ASSERT_EQ(usda->prims.size(), usdc->prims.size());
        ASSERT_EQ(usda->prims.front().path, "/Parent");
        for (size_t i = 0; i < 16; ++i) {
            EXPECT_DOUBLE_EQ(usda->prims.front().local_transform[i], usdc->prims.front().local_transform[i]);
        }
        EXPECT_DOUBLE_EQ(usda->prims.front().local_transform[12], 10.0);
        EXPECT_DOUBLE_EQ(usda->prims.front().local_transform[13], 20.0);
        EXPECT_DOUBLE_EQ(usda->prims.front().local_transform[14], 30.0);
    }

    TEST_F(UsdFormatTest, XformFixtureAppliesWorldTransform) {
        const auto root = fs::path(PROJECT_ROOT_PATH) / "tests/data/usd/";
        for (const auto& extension : {"usda", "usdc", "usd"}) {
            auto result = load_usd(root / (std::string("xform_stage.") + extension));
            ASSERT_TRUE(result.has_value()) << result.error();
            const auto means = result->means().contiguous().to(Device::CPU);
            const auto* values = static_cast<const float*>(means.data_ptr());
            const float expected[] = {9, 23, 38, 16, 8, 52, 22, 41, -2};
            for (size_t index = 0; index < std::size(expected); ++index) {
                EXPECT_FLOAT_EQ(values[index], expected[index]) << extension << "[" << index << "]";
            }
        }
    }

    TEST_F(UsdFormatTest, NestedXformUsesUsdOrder) {
        const auto root = fs::path(PROJECT_ROOT_PATH) / "tests/data/usd/";
        for (const auto& extension : {"usda", "usdc"}) {
            auto result = load_usd(root / (std::string("nested_xform.") + extension));
            ASSERT_TRUE(result.has_value()) << result.error();
            const auto means = result->means().contiguous().to(Device::CPU);
            const auto* values = static_cast<const float*>(means.data_ptr());
            EXPECT_NEAR(values[0], 0.0f, EPSILON);
            EXPECT_NEAR(values[1], 1.0f, EPSILON);
            EXPECT_NEAR(values[3], 0.0f, EPSILON);
            EXPECT_NEAR(values[4], 2.0f, EPSILON);
        }
    }

    TEST_F(UsdFormatTest, XformOpsAndSpecialTokensLoad) {
        const auto root = fs::path(PROJECT_ROOT_PATH) / "tests/data/usd/";
        for (const auto& extension : {"usda", "usdc"}) {
            auto ops = load_usd(root / (std::string("xform_ops.") + extension));
            ASSERT_TRUE(ops.has_value()) << ops.error();
            const auto ops_means = ops->means().contiguous().to(Device::CPU);
            const auto* ops_position = static_cast<const float*>(ops_means.data_ptr());
            EXPECT_NEAR(ops_position[0], 1.7260047f, EPSILON);
            EXPECT_NEAR(ops_position[1], 0.2686399f, EPSILON);
            EXPECT_NEAR(ops_position[2], 3.3088882f, EPSILON);
            const auto ops_rotation = ops->rotation_raw().contiguous().to(Device::CPU);
            const auto* rotation = static_cast<const float*>(ops_rotation.data_ptr());
            const float expected_rotation[] = {0.43433788f, 0.43292007f, 0.31106988f, 0.72606224f};
            float rotation_dot = 0.0f;
            for (size_t index = 0; index < 4; ++index) {
                rotation_dot += rotation[index] * expected_rotation[index];
            }
            EXPECT_NEAR(std::abs(rotation_dot), 1.0f, EPSILON);
            const auto ops_scaling = ops->scaling_raw().contiguous().to(Device::CPU);
            const auto* scaling = static_cast<const float*>(ops_scaling.data_ptr());
            EXPECT_FLOAT_EQ(scaling[0], 0.0f);
            EXPECT_FLOAT_EQ(scaling[1], 0.0f);
            EXPECT_FLOAT_EQ(scaling[2], 0.0f);
            auto inverse = load_usd(root / (std::string("xform_invert.") + extension));
            ASSERT_TRUE(inverse.has_value()) << inverse.error();
            const auto inverse_means = inverse->means().contiguous().to(Device::CPU);
            EXPECT_FLOAT_EQ(static_cast<const float*>(inverse_means.data_ptr())[0], 7.0f);
            auto reset = load_usd(root / (std::string("xform_reset.") + extension));
            ASSERT_TRUE(reset.has_value()) << reset.error();
            const auto reset_means = reset->means().contiguous().to(Device::CPU);
            EXPECT_FLOAT_EQ(static_cast<const float*>(reset_means.data_ptr())[0], 3.0f);
        }
    }

    TEST_F(UsdFormatTest, ProvenanceWordDoesNotTriggerCompositionRejection) {
        FlatStage stage;
        stage.default_prim = "/GaussianSplats";
        stage.custom_layer_data = {
            {"_private", "underscore identifiers stay bare"},
            {"9starts", "digit-leading keys require quotes"},
            {"provenance", "references are recorded as plain text"},
            {"with space", "whitespace requires quotes"},
        };
        stage.prims.push_back(positions_prim("/GaussianSplats", {1.0f, 2.0f, 3.0f}));
        const auto path = temp_dir / "provenance.usda";
        ASSERT_TRUE(write_usda(stage, path).has_value());
        const auto text = read_text_file(path);
        EXPECT_NE(text.find("string _private = "), std::string::npos);
        EXPECT_NE(text.find("string provenance = "), std::string::npos);
        EXPECT_NE(text.find("string \"9starts\" = "), std::string::npos);
        EXPECT_NE(text.find("string \"with space\" = "), std::string::npos);
        auto loaded = read_usda(path);
        ASSERT_TRUE(loaded.has_value()) << lfs::format_for_developer(loaded.error());
        EXPECT_EQ(loaded->custom_layer_data, stage.custom_layer_data);
    }

    TEST_F(UsdFormatTest, TripleQuotedMetadataDoesNotDesynchronizeFastPath) {
        FlatStage stage;
        stage.default_prim = "/GaussianSplats";
        stage.custom_layer_data["doc"] = "placeholder";
        const std::vector<float> authored_positions = {1.25f, -2.5f, 3.75f, 4.5f, 5.25f, -6.0f};
        stage.prims.push_back(positions_prim("/GaussianSplats", authored_positions));
        const auto path = temp_dir / "triple_quoted_metadata.usda";
        ASSERT_TRUE(write_usda(stage, path).has_value());

        auto text = read_text_file(path);
        const auto placeholder = std::string{"string doc = \"placeholder\""};
        const auto replacement = std::string{"string doc = \"\"\"metadata { \"quoted\" }\ncontinued\"\"\""};
        const auto position = text.find(placeholder);
        ASSERT_NE(position, std::string::npos);
        text.replace(position, placeholder.size(), replacement);
        write_text_file(path, text);

        auto loaded = load_usd(path);
        ASSERT_TRUE(loaded.has_value()) << loaded.error();
        const auto means = loaded->means().contiguous().to(Device::CPU);
        const auto* const values = static_cast<const float*>(means.data_ptr());
        for (size_t index = 0; index < authored_positions.size(); ++index) {
            EXPECT_FLOAT_EQ(values[index], authored_positions[index]);
        }

        const auto unterminated_path = temp_dir / "unterminated_triple_quote.usda";
        const auto closing = text.find("continued\"\"\"");
        ASSERT_NE(closing, std::string::npos);
        text.replace(closing, std::string{"continued\"\"\""}.size(), "continued\"\"");
        write_text_file(unterminated_path, text);
        auto unterminated = load_usd(unterminated_path);
        if (unterminated.has_value()) {
            const auto unterminated_means = unterminated->means().contiguous().to(Device::CPU);
            const auto* const unterminated_values = static_cast<const float*>(unterminated_means.data_ptr());
            for (size_t index = 0; index < authored_positions.size(); ++index) {
                EXPECT_FLOAT_EQ(unterminated_values[index], authored_positions[index]);
            }
        }
    }

    TEST_F(UsdFormatTest, CompositionIsRejectedExplicitly) {
        const auto path = fs::path(PROJECT_ROOT_PATH) / "tests/data/usd/reference_stage.usda";
        auto result = read_usda(path);
        ASSERT_FALSE(result.has_value());
        EXPECT_NE(std::string(result.error().user_message()).find("USD composition is unsupported"), std::string::npos);
    }

    TEST_F(UsdFormatTest, UsdzReadsFlatCrateRoot) {
        const auto path = fs::path(PROJECT_ROOT_PATH) / "tests/data/usd/flat_root.usdz";
        auto stage = read_usdz(path);
        ASSERT_TRUE(stage.has_value()) << lfs::format_for_developer(stage.error());
        ASSERT_FALSE(stage->prims.empty());
        const auto& positions = stage->prims.back().attributes.at("positions").values;
        EXPECT_EQ(positions.size(), 9u);
    }

} // namespace
