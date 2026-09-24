/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/tensor.hpp"
#include "training/components/ppisp_file.hpp"

#include "core/tensor_backend.hpp"
#include "rendering/appearance_tensor_model.hpp"
#include "rendering/viewport_appearance_correction.hpp"
#include "scene/scene_manager.hpp"
#include "tensor_test_support.hpp"
#include <cmath>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#if LFS_BUILD_TRAINER
#include "training/components/ppisp.hpp"
#include "training/components/ppisp_controller_pool.hpp"
#endif

#include <chrono>
#include <gtest/gtest.h>
#include <optional>

namespace {
    using namespace lfs::core;
    using namespace lfs::vis;

    void writeAppearanceFixture(const std::filesystem::path& path, const bool controller) {
        using namespace lfs::core;
        std::ofstream output(path, std::ios::binary);
        const auto write = [&output](const auto& value) {
            output.write(reinterpret_cast<const char*>(&value), sizeof(value));
        };
        lfs::training::PPISPFileHeader header;
        header.num_cameras = 1;
        header.num_frames = 2;
        header.flags = controller ? static_cast<uint32_t>(lfs::training::PPISPFileFlags::HAS_CONTROLLER) : 0;
        write(header);
        write(uint32_t{0x4C465049});
        write(uint32_t{1});
        write(int{1});
        write(int{2});
        output << Tensor::from_vector({0.35f, -0.25f}, {2}, Device::CPU);
        output << Tensor::from_vector({0.03f, -0.02f, -0.3f, -0.1f, 0.03f,
                                       -0.01f, 0.04f, -0.2f, -0.07f, 0.01f,
                                       0.02f, 0.01f, -0.25f, -0.05f, 0.02f},
                                      {15}, Device::CPU);
        output << Tensor::from_vector({0.1f, -0.1f, 0.03f, 0.04f, 0.06f, -0.02f, 0.09f, -0.05f,
                                       -0.05f, 0.09f, -0.02f, 0.07f, 0.03f, 0.01f, -0.04f, 0.08f},
                                      {16}, Device::CPU);
        output << Tensor::from_vector({0.1f, -0.03f, 0.4f, -0.2f,
                                       -0.05f, 0.2f, 0.3f, 0.1f,
                                       0.2f, -0.1f, 0.5f, 0.3f},
                                      {12}, Device::CPU);
        if (controller) {
            write(uint32_t{0x4C464349});
            write(uint32_t{1});
            write(int{1});
            const auto matrix = [&](const size_t rows, const size_t columns) {
                std::vector<float> values(rows * columns);
                const float scale = 0.15f * std::sqrt(6.0f / static_cast<float>(columns));
                for (size_t i = 0; i < values.size(); ++i)
                    values[i] = scale * std::sin(static_cast<float>(i + 1) * 1.31f);
                output << Tensor::from_vector(values, {rows, columns}, Device::CPU);
                output << Tensor::full({rows}, 0.02f, Device::CPU);
            };
            matrix(16, 3);
            matrix(32, 16);
            matrix(64, 32);
            matrix(128, 1601);
            matrix(128, 128);
            matrix(128, 128);
            matrix(9, 128);
        }
        if (!output)
            throw std::runtime_error("Cannot write appearance test fixture");
    }

    Tensor picture(const int height = 211, const int width = 317) {
        std::vector<float> values(3 * height * width);
        for (size_t i = 0; i < values.size(); ++i)
            values[i] = static_cast<float>(static_cast<int>(i % 301) - 20) / 270.0f;
        return Tensor::from_vector(values, {3, static_cast<size_t>(height), static_cast<size_t>(width)}, Device::CPU);
    }

    PPISPOverrides overrides() {
        PPISPOverrides result;
        result.exposure_offset = 0.3f;
        result.vignette_strength = 0.7f;
        result.wb_temperature = 0.05f;
        result.wb_tint = -0.03f;
        result.color_blue_x = 0.03f;
        result.color_green_y = -0.02f;
        result.gamma_multiplier = 1.3f;
        result.gamma_red = -0.1f;
        result.crf_toe = 0.2f;
        result.crf_shoulder = -0.15f;
        return result;
    }

#if LFS_BUILD_TRAINER
    lfs::training::PPISPRenderOverrides referenceOverrides() {
        const auto values = overrides();
        lfs::training::PPISPRenderOverrides result;
        result.exposure_offset = values.exposure_offset;
        result.vignette_strength = values.vignette_strength;
        result.wb_temperature = values.wb_temperature;
        result.wb_tint = values.wb_tint;
        result.color_blue_x = values.color_blue_x;
        result.color_green_y = values.color_green_y;
        result.gamma_multiplier = values.gamma_multiplier;
        result.gamma_red = values.gamma_red;
        result.crf_toe = values.crf_toe;
        result.crf_shoulder = values.crf_shoulder;
        return result;
    }

#endif
    void expectClose(const Tensor& actual, const std::vector<float>& expected, const float tolerance = 3e-5f) {
        const auto values = actual.to_vector();
        ASSERT_EQ(values.size(), expected.size());
        float error = 0;
        for (size_t i = 0; i < values.size(); ++i) {
            ASSERT_TRUE(std::isfinite(values[i])) << i;
            error = std::max(error, std::abs(values[i] - expected[i]));
        }
        EXPECT_LE(error, tolerance);
    }

    class ViewerAppearance : public ::testing::TestWithParam<GpuBackend> {
    protected:
        std::filesystem::path path;
        std::optional<GpuBackendScope> scope;
        void SetUp() override {
            if (!gpu_backend_available(GetParam()))
                GTEST_SKIP() << "Requested backend unavailable";
            scope.emplace(GetParam());
            path = std::filesystem::temp_directory_path() /
                   ("viewer-appearance-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) + ".ppisp");
            writeAppearanceFixture(path, true);
        }
        void TearDown() override {
            std::error_code error;
            if (!path.empty())
                std::filesystem::remove(path, error);
            if (GetParam() == GpuBackend::Vulkan) {
                EXPECT_TRUE(shutdown_gpu_backend(GpuBackend::Vulkan));
                for (const auto& message : lfs::test::vulkan_validation_messages())
                    ADD_FAILURE() << message;
            }
        }
    };

#if LFS_BUILD_TRAINER
    TEST_P(ViewerAppearance, MatchesCudaKnownFramesAndOverrides) {
        if (!gpu_backend_available(GpuBackend::CUDA))
            GTEST_SKIP() << "Requires the CUDA reference";
        const auto cpu = picture(533, 541);
        const auto loaded = AppearanceTensorModel::load(path, GetParam());
        ASSERT_TRUE(loaded) << lfs::format_for_developer(loaded.error());
        for (const int uid : {0, 1}) {
            for (const bool manual : {false, true}) {
                std::vector<float> expected;
                {
                    const GpuBackendScope reference_backend(GpuBackend::CUDA);
                    lfs::training::PPISP reference(1);
                    lfs::training::PPISPControllerPool pool(1, 1);
                    ASSERT_TRUE(lfs::training::load_ppisp_file(path, reference, &pool));
                    const auto input = cpu.gpu();
                    expected = (manual ? reference.apply_with_overrides(input, 0, uid, referenceOverrides())
                                       : reference.apply(input, 0, uid))
                                   .to_vector();
                }
                expectClose((*loaded)->apply(cpu.gpu(), uid, manual ? overrides() : PPISPOverrides{}, false), expected);
            }
        }
    }

    TEST_P(ViewerAppearance, NovelViewWithoutControllerMatchesCudaNeutralParameters) {
        if (!gpu_backend_available(GpuBackend::CUDA))
            GTEST_SKIP() << "Requires the CUDA reference";
        writeAppearanceFixture(path, false);
        const auto cpu = picture(17, 31);
        const auto loaded = AppearanceTensorModel::load(path, GetParam());
        ASSERT_TRUE(loaded);
        for (const bool manual : {false, true}) {
            std::vector<float> expected;
            {
                const GpuBackendScope reference_backend(GpuBackend::CUDA);
                lfs::training::PPISP reference(1);
                ASSERT_TRUE(lfs::training::load_ppisp_file(path, reference));
                const auto neutral = Tensor::zeros({1, 9}, Device::GPU);
                expected = (manual ? reference.apply_with_controller_params_and_overrides(cpu.gpu(), neutral, 0, referenceOverrides())
                                   : reference.apply_with_controller_params(cpu.gpu(), neutral, 0))
                               .to_vector();
            }
            expectClose((*loaded)->apply(cpu.gpu(), -1, manual ? overrides() : PPISPOverrides{}, true), expected);
        }
    }

    TEST_P(ViewerAppearance, ControllerPredictionAndCorrectionMatchCuda) {
        if (!gpu_backend_available(GpuBackend::CUDA))
            GTEST_SKIP() << "Requires the CUDA reference";
        const auto cpu = picture(63, 79);
        std::vector<float> parameters, corrected;
        {
            const GpuBackendScope reference_backend(GpuBackend::CUDA);
            lfs::training::PPISP reference(1);
            lfs::training::PPISPControllerPool pool(1, 1);
            ASSERT_TRUE(lfs::training::load_ppisp_file(path, reference, &pool));
            pool.allocate_buffers(63, 79);
            const auto input = cpu.gpu();
            const auto prediction = pool.predict(0, input.unsqueeze(0));
            parameters = prediction.to_vector();
            corrected = reference.apply_with_controller_params_and_overrides(input, prediction, 0, referenceOverrides()).to_vector();
        }
        const auto loaded = AppearanceTensorModel::load(path, GetParam());
        ASSERT_TRUE(loaded) << lfs::format_for_developer(loaded.error());
        const auto input = cpu.gpu();
        const auto prediction = (*loaded)->predict(input);
        expectClose(prediction, parameters, 1e-5f);
        expectClose((*loaded)->apply(input, -1, overrides(), true), corrected);
    }

#endif
    TEST_P(ViewerAppearance, FullImageAndBandsAgreeWithoutCudaOrAController) {
        writeAppearanceFixture(path, false);
        const auto loaded = AppearanceTensorModel::load(path, GetParam());
        ASSERT_TRUE(loaded) << lfs::format_for_developer(loaded.error());
        EXPECT_FALSE((*loaded)->hasController());
        const auto input = picture().gpu();
        const auto full = (*loaded)->apply(input, -1, overrides(), false);
        auto params = (*loaded)->parameters(-1, overrides(), {});
        params.full_height = 211;
        auto banded = Tensor::empty(input.shape(), Device::GPU);
        for (int row = 0; row < 211; row += 19) {
            const int end = std::min(row + 19, 211);
            const auto band = input.slice(1, row, end).contiguous();
            params.y_offset = row;
            banded.slice(1, row, end).copy_from(ppisp_apply(band, params));
        }
        expectClose(banded, full.to_vector(), 1e-6f);
        EXPECT_EQ(gpu_backend_of(full), GetParam());
    }

    TEST_P(ViewerAppearance, OwnsControllerOutputsAcrossPredictions) {
        const auto loaded = AppearanceTensorModel::load(path, GetParam());
        ASSERT_TRUE(loaded) << lfs::format_for_developer(loaded.error());
        const auto input = picture(63, 79).gpu();
        const auto first = (*loaded)->predict(input);
        const auto saved = first.to_vector();
        const auto second = (*loaded)->predict(input.mul(0.1f));
        EXPECT_EQ(first.to_vector(), saved);
        EXPECT_NE(second.to_vector(), saved);
        EXPECT_EQ(gpu_backend_of(first), GetParam());
    }

    TEST_P(ViewerAppearance, RejectsTruncatedWeightsAndTinyImages) {
        const auto loaded = AppearanceTensorModel::load(path, GetParam());
        ASSERT_TRUE(loaded) << lfs::format_for_developer(loaded.error());
        const auto input = picture(7, 9).gpu();
        EXPECT_THROW((void)(*loaded)->predict(input.slice(1, 0, 2).contiguous()), lfs::Exception);
        std::filesystem::resize_file(path, 51);
        EXPECT_FALSE(AppearanceTensorModel::load(path, GetParam()));
    }

    TEST_P(ViewerAppearance, SavedModelCorrectsNovelViewsAndPreservesAlpha) {
        writeAppearanceFixture(path, false);
        auto model = AppearanceTensorModel::load(path, GetParam());
        ASSERT_TRUE(model);
        SceneManager scene;
        scene.setAppearanceModel(std::move(*model));
        const auto oracle = AppearanceTensorModel::load(path, GetParam());
        ASSERT_TRUE(oracle) << lfs::format_for_developer(oracle.error());
        const auto rgb = picture().clamp(0.0f, 1.0f);
        auto rgba = Tensor::empty({4, 211, 317}, Device::CPU);
        rgba.slice(0, 0, 3).copy_from(rgb);
        rgba.slice(0, 3, 4).fill_(0.37f);
        const auto input = std::make_shared<Tensor>(rgba.gpu());
        RenderSettings settings;
        settings.apply_appearance_correction = true;
        settings.ppisp_mode = RenderSettings::PPISPMode::MANUAL;
        settings.ppisp_overrides = overrides();
        const auto result = applyViewportAppearanceCorrection(input, &scene, settings, -1);
        ASSERT_NE(result.get(), input.get());
        expectClose(result->slice(0, 0, 3), (*oracle)->apply(rgb.gpu(), -1, overrides(), false).to_vector());
        EXPECT_EQ(result->slice(0, 3, 4).to_vector(), rgba.slice(0, 3, 4).to_vector());
        const auto export_input = rgba.permute({1, 2, 0}).contiguous().mul(255.0f).to(DataType::UInt8);
        const auto exported = applyExportPostProcess(export_input.clone(), &scene, settings, -1,
                                                     ExportPostProcessMode::Transparent, {});
        ASSERT_TRUE(exported) << exported.error();
        EXPECT_EQ(exported->slice(2, 3, 4).to_vector(), export_input.slice(2, 3, 4).to_vector());
        EXPECT_NE(exported->slice(2, 0, 3).to_vector(), export_input.slice(2, 0, 3).to_vector());
    }

    INSTANTIATE_TEST_SUITE_P(Backends, ViewerAppearance, ::testing::ValuesIn(kGpuBackends),
                             [](const ::testing::TestParamInfo<GpuBackend>& info) { return info.param == GpuBackend::CUDA ? "CUDA" : "Vulkan"; });
} // namespace
