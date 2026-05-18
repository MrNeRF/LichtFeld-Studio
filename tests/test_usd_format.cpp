/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include <cmath>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <iterator>
#include <string>
#include <vector>

#include "core/splat_data.hpp"
#include "io/exporter.hpp"
#include "io/formats/usd.hpp"
#include "io/loaders/usd_loader.hpp"
#include <pxr/usd/sdf/path.h>
#include <pxr/usd/usd/stage.h>
#include <pxr/usd/usdGeom/boundable.h>
#include <pxr/usd/usdGeom/metrics.h>
#include <pxr/usd/usdGeom/xform.h>
#include <pxr/usd/usdVol/particleField3DGaussianSplat.h>

namespace fs = std::filesystem;
using namespace lfs::core;
using namespace lfs::io;

namespace {

    class UsdFormatTest : public ::testing::Test {
    protected:
        static constexpr float EPSILON = 1e-4f;

        const fs::path temp_dir = fs::temp_directory_path() / "lfs_usd_test";

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

        static void author_positions_only_field(const pxr::UsdStageRefPtr& stage,
                                                const std::string& prim_path,
                                                const std::vector<pxr::GfVec3f>& positions) {
            auto splat = pxr::UsdVolParticleField3DGaussianSplat::Define(stage, pxr::SdfPath(prim_path));
            ASSERT_TRUE(splat);
            ASSERT_TRUE(splat.CreatePositionsAttr().Set(pxr::VtArray<pxr::GfVec3f>(positions.begin(), positions.end())));
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
        ASSERT_TRUE(fs::exists(usd_path));
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
        const auto orig_shN = original.shN().contiguous().to(Device::CPU);
        const auto load_shN = loaded.shN().contiguous().to(Device::CPU);

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
            EXPECT_NEAR(load_means_ptr[i], orig_means_ptr[i], EPSILON) << "Means mismatch at " << i;
            EXPECT_NEAR(load_scaling_ptr[i], orig_scaling_ptr[i], EPSILON) << "Scaling mismatch at " << i;
            EXPECT_NEAR(load_sh0_ptr[i], orig_sh0_ptr[i], EPSILON) << "SH0 mismatch at " << i;
        }

        for (size_t i = 0; i < original.size(); ++i) {
            EXPECT_NEAR(load_opacity_ptr[i], orig_opacity_ptr[i], EPSILON) << "Opacity mismatch at " << i;

            const float dot = orig_rotation_ptr[i * 4 + 0] * load_rotation_ptr[i * 4 + 0] +
                              orig_rotation_ptr[i * 4 + 1] * load_rotation_ptr[i * 4 + 1] +
                              orig_rotation_ptr[i * 4 + 2] * load_rotation_ptr[i * 4 + 2] +
                              orig_rotation_ptr[i * 4 + 3] * load_rotation_ptr[i * 4 + 3];
            EXPECT_NEAR(std::abs(dot), 1.0f, EPSILON) << "Rotation mismatch at point " << i;
        }

        ASSERT_TRUE(loaded.shN().is_valid());
        for (size_t i = 0; i < original.size() * static_cast<size_t>(original.shN().size(1)) * 3; ++i) {
            EXPECT_NEAR(load_shN_ptr[i], orig_shN_ptr[i], EPSILON) << "SHN mismatch at " << i;
        }
    }

    TEST_F(UsdFormatTest, RejectsUnsupportedExtension) {
        auto splat = create_test_splat(4, 0);
        auto result = save_usd(splat, {.output_path = temp_dir / "packed.usdz"});
        ASSERT_FALSE(result.has_value());
        EXPECT_EQ(result.error().code, ErrorCode::UNSUPPORTED_FORMAT);
    }

    TEST_F(UsdFormatTest, SaveCancellationKeepsExistingTarget) {
        auto splat = create_test_splat(4, 0);
        const fs::path usd_path = temp_dir / "keep_existing_on_cancel.usda";
        const std::string existing_contents = "existing usd data";
        write_text_file(usd_path, existing_contents);

        auto result = save_usd(splat, {.output_path = usd_path,
                                       .progress_callback = [](float progress, const std::string&) {
                                           return progress < 1.0f;
                                       }});
        ASSERT_FALSE(result.has_value());
        EXPECT_EQ(result.error().code, ErrorCode::CANCELLED);
        expect_existing_target_and_no_temp_files(usd_path, existing_contents);
    }

    TEST_F(UsdFormatTest, ExportAuthorsStageMetricsAndExtent) {
        auto splat = create_test_splat(4, 0);
        const fs::path usd_path = temp_dir / "metadata.usda";

        ASSERT_TRUE(save_usd(splat, {.output_path = usd_path}).has_value());

        const auto stage = pxr::UsdStage::Open(usd_path.string());
        ASSERT_TRUE(stage);
        EXPECT_EQ(pxr::UsdGeomGetStageUpAxis(stage), pxr::UsdGeomTokens->y);
        EXPECT_FLOAT_EQ(static_cast<float>(pxr::UsdGeomGetStageMetersPerUnit(stage)), 1.0f);

        const auto prim = stage->GetDefaultPrim();
        ASSERT_TRUE(prim);

        pxr::UsdGeomBoundable boundable(prim);
        auto extent_attr = boundable.GetExtentAttr();
        ASSERT_TRUE(extent_attr);

        pxr::VtArray<pxr::GfVec3f> extent;
        ASSERT_TRUE(extent_attr.Get(&extent));
        ASSERT_EQ(extent.size(), 2u);

        EXPECT_FLOAT_EQ(extent[0][0], -1.0f);
        EXPECT_FLOAT_EQ(extent[0][1], 0.0f);
        EXPECT_FLOAT_EQ(extent[0][2], -1.5f);
        EXPECT_FLOAT_EQ(extent[1][0], -0.25f);
        EXPECT_FLOAT_EQ(extent[1][1], 1.5f);
        EXPECT_FLOAT_EQ(extent[1][2], 0.0f);
    }

    TEST_F(UsdFormatTest, ImportPrefersDefaultPrimSubtreeAndAppliesStageUnits) {
        const fs::path usd_path = temp_dir / "default_prim.usda";
        auto stage = pxr::UsdStage::CreateNew(usd_path.string());
        ASSERT_TRUE(stage);

        pxr::UsdGeomSetStageUpAxis(stage, pxr::UsdGeomTokens->y);
        pxr::UsdGeomSetStageMetersPerUnit(stage, 2.0);

        auto asset_root = pxr::UsdGeomXform::Define(stage, pxr::SdfPath("/Asset"));
        ASSERT_TRUE(asset_root);
        stage->SetDefaultPrim(asset_root.GetPrim());

        author_positions_only_field(stage, "/Asset/Chosen", {pxr::GfVec3f(1.0f, 2.0f, 3.0f)});
        author_positions_only_field(stage, "/Other", {pxr::GfVec3f(100.0f, 200.0f, 300.0f)});

        ASSERT_TRUE(stage->GetRootLayer()->Save());

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
        auto stage = pxr::UsdStage::CreateNew(usd_path.string());
        ASSERT_TRUE(stage);

        author_positions_only_field(stage, "/First", {pxr::GfVec3f(1.0f, 0.0f, 0.0f)});
        author_positions_only_field(stage, "/Second", {pxr::GfVec3f(2.0f, 0.0f, 0.0f)});
        ASSERT_TRUE(stage->GetRootLayer()->Save());

        auto loaded_result = load_usd(usd_path);
        ASSERT_FALSE(loaded_result.has_value());
        EXPECT_NE(loaded_result.error().find("multiple OpenUSD ParticleField prims"), std::string::npos);
    }

    TEST_F(UsdFormatTest, ShortShPayloadFallsBackToDegreeZero) {
        const fs::path usd_path = temp_dir / "short_sh.usda";
        auto stage = pxr::UsdStage::CreateNew(usd_path.string());
        ASSERT_TRUE(stage);

        auto splat = pxr::UsdVolParticleField3DGaussianSplat::Define(stage, pxr::SdfPath("/GaussianSplats"));
        ASSERT_TRUE(splat);
        stage->SetDefaultPrim(splat.GetPrim());

        ASSERT_TRUE(splat.CreatePositionsAttr().Set(pxr::VtArray<pxr::GfVec3f>{pxr::GfVec3f(1.0f, 2.0f, 3.0f)}));
        ASSERT_TRUE(splat.CreateRadianceSphericalHarmonicsDegreeAttr().Set(3));
        ASSERT_TRUE(splat.CreateRadianceSphericalHarmonicsCoefficientsAttr().Set(
            pxr::VtArray<pxr::GfVec3f>{pxr::GfVec3f(0.25f, -0.5f, 0.75f)}));
        ASSERT_TRUE(stage->GetRootLayer()->Save());

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
        auto stage = pxr::UsdStage::CreateNew(usd_path.string());
        ASSERT_TRUE(stage);

        auto splat = pxr::UsdVolParticleField3DGaussianSplat::Define(stage, pxr::SdfPath("/GaussianSplats"));
        ASSERT_TRUE(splat);
        stage->SetDefaultPrim(splat.GetPrim());
        ASSERT_TRUE(splat.CreatePositionsAttr().Set(
            pxr::VtArray<pxr::GfVec3f>{pxr::GfVec3f(1.0f, 2.0f, 3.0f), pxr::GfVec3f(4.0f, 5.0f, 6.0f)}));
        ASSERT_TRUE(stage->GetRootLayer()->Save());

        USDLoader loader;
        auto result = loader.load(usd_path, {.validate_only = true});
        ASSERT_TRUE(result.has_value()) << result.error().message;
        EXPECT_EQ(result->loader_used, "OpenUSD");
        const auto splat_data = std::get_if<std::shared_ptr<SplatData>>(&result->data);
        ASSERT_NE(splat_data, nullptr);
        EXPECT_EQ(*splat_data, nullptr);
        EXPECT_EQ(result->scene_center.numel(), 3);
    }

} // namespace
