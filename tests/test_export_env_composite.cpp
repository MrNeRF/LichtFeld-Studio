/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

// The CUDA export environment composite must match the shared CPU reference
// math (environment_math.hpp) that also drives the software video composite,
// and banded application must equal a single full-image pass.

#include <cmath>
#include <cstdint>
#include <cstring>
#include <cuda_runtime.h>
#include <filesystem>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <gtest/gtest.h>
#include <vector>

#include "core/tensor.hpp"
#include "environment_image.hpp"
#include "environment_math.hpp"
#include "rendering/coordinate_conventions.hpp"
#include "rendering/export_post_process.hpp"

namespace {

    using lfs::core::Tensor;
    namespace envmath = lfs::rendering::envmath;

    constexpr const char* kEnvironmentAsset = "resources/assets/environments/alps_field_1k.hdr";
    constexpr int WIDTH = 96;
    constexpr int HEIGHT = 64;
    constexpr float FOCAL_LENGTH_MM = 26.0f;
    constexpr float EXPOSURE_EV = 0.7f;
    constexpr float ROTATION_DEGREES = 33.0f;

    [[nodiscard]] glm::mat3 testCameraRotation() {
        return glm::mat3(glm::rotate(glm::mat4(1.0f), glm::radians(25.0f), glm::vec3(0.3f, 0.9f, 0.1f)));
    }

    [[nodiscard]] std::vector<float> makeRgbChw() {
        std::vector<float> rgb(3u * HEIGHT * WIDTH);
        for (int y = 0; y < HEIGHT; ++y) {
            for (int x = 0; x < WIDTH; ++x) {
                const size_t idx = static_cast<size_t>(y) * WIDTH + x;
                rgb[idx] = static_cast<float>(x) / (WIDTH - 1);
                rgb[HEIGHT * WIDTH + idx] = static_cast<float>(y) / (HEIGHT - 1);
                rgb[2u * HEIGHT * WIDTH + idx] = 0.25f + 0.5f * static_cast<float>((x + y) % 7) / 6.0f;
            }
        }
        return rgb;
    }

    [[nodiscard]] std::vector<float> makeAlpha() {
        std::vector<float> alpha(static_cast<size_t>(HEIGHT) * WIDTH);
        for (size_t i = 0; i < alpha.size(); ++i) {
            alpha[i] = static_cast<float>(i % 256) / 255.0f;
        }
        return alpha;
    }

    [[nodiscard]] uint8_t quantizeU8(const float value) {
        const float clamped = std::min(std::max(value, 0.0f), 1.0f);
        return static_cast<uint8_t>(clamped * 255.0f + 0.5f);
    }

    [[nodiscard]] std::vector<uint8_t> cpuReferenceComposite(const lfs::rendering::EnvironmentImage& env,
                                                             const std::vector<float>& rgb_chw,
                                                             const std::vector<float>& alpha,
                                                             const bool equirectangular_view) {
        const glm::mat3 rotation = testCameraRotation();
        const auto [focal_x, focal_y] =
            lfs::rendering::computePixelFocalLengths({WIDTH, HEIGHT}, FOCAL_LENGTH_MM);
        const float exposure_factor = std::exp2(EXPOSURE_EV);
        const float rotation_radians = glm::radians(ROTATION_DEGREES);

        const auto fetch = [&](const int px, const int py) -> envmath::Vec3 {
            const size_t index =
                (static_cast<size_t>(py) * static_cast<size_t>(env.width) + static_cast<size_t>(px)) * 3u;
            return {env.pixels[index], env.pixels[index + 1], env.pixels[index + 2]};
        };

        std::vector<uint8_t> out(static_cast<size_t>(HEIGHT) * WIDTH * 3u);
        const size_t plane = static_cast<size_t>(HEIGHT) * WIDTH;
        for (int y = 0; y < HEIGHT; ++y) {
            for (int x = 0; x < WIDTH; ++x) {
                envmath::Vec3 dir = envmath::environmentWorldDirection(
                    static_cast<float>(x), static_cast<float>(y),
                    static_cast<float>(WIDTH), static_cast<float>(HEIGHT),
                    equirectangular_view, focal_x, focal_y,
                    static_cast<float>(WIDTH) * 0.5f, static_cast<float>(HEIGHT) * 0.5f,
                    &rotation[0][0]);
                dir = envmath::normalized(envmath::rotateAroundY(dir, rotation_radians));
                const auto uv = envmath::equirectUvForDirection(dir);
                const envmath::Vec3 hdr =
                    envmath::sampleEnvironmentBilinear(fetch, uv.u, uv.v, env.width, env.height);
                const envmath::Vec3 background = envmath::shadeEnvironmentRadiance(hdr, exposure_factor);

                const size_t idx = static_cast<size_t>(y) * WIDTH + x;
                const envmath::Vec3 rgb{rgb_chw[idx], rgb_chw[plane + idx], rgb_chw[2u * plane + idx]};
                const envmath::Vec3 blended = envmath::mix(background, rgb, alpha[idx]);
                out[idx * 3u] = quantizeU8(blended.x);
                out[idx * 3u + 1u] = quantizeU8(blended.y);
                out[idx * 3u + 2u] = quantizeU8(blended.z);
            }
        }
        return out;
    }

    [[nodiscard]] lfs::rendering::EnvironmentCompositeBandParams makeParams(const bool equirectangular_view) {
        const auto [focal_x, focal_y] =
            lfs::rendering::computePixelFocalLengths({WIDTH, HEIGHT}, FOCAL_LENGTH_MM);
        lfs::rendering::EnvironmentCompositeBandParams params;
        params.camera_rotation = testCameraRotation();
        params.full_size = {WIDTH, HEIGHT};
        params.y_offset = 0;
        params.focal_x = focal_x;
        params.focal_y = focal_y;
        params.center_x = static_cast<float>(WIDTH) * 0.5f;
        params.center_y = static_cast<float>(HEIGHT) * 0.5f;
        params.equirectangular_view = equirectangular_view;
        params.exposure = EXPOSURE_EV;
        params.rotation_degrees = ROTATION_DEGREES;
        return params;
    }

    class ExportEnvCompositeTest : public ::testing::Test {
    protected:
        void SetUp() override {
            if (!std::filesystem::exists(kEnvironmentAsset)) {
                GTEST_SKIP() << "environment asset not available: " << kEnvironmentAsset;
            }
            auto loaded = lfs::rendering::getOrLoadCudaEnvironmentMap(kEnvironmentAsset);
            ASSERT_TRUE(loaded.has_value()) << loaded.error();
            env_map_ = *loaded;
            auto cpu = lfs::rendering::loadEnvironmentImage(kEnvironmentAsset);
            ASSERT_TRUE(cpu.has_value()) << cpu.error();
            env_cpu_ = std::move(*cpu);

            rgb_host_ = makeRgbChw();
            alpha_host_ = makeAlpha();
            rgb_ = Tensor::from_vector(rgb_host_, {3, HEIGHT, WIDTH}, lfs::core::Device::CUDA);
            alpha_ = Tensor::from_vector(alpha_host_, {HEIGHT, WIDTH}, lfs::core::Device::CUDA);
            ASSERT_TRUE(rgb_.is_valid());
            ASSERT_TRUE(alpha_.is_valid());
        }

        void TearDown() override {
            lfs::rendering::releaseCudaEnvironmentMapCache();
        }

        [[nodiscard]] std::vector<uint8_t> runCudaComposite(const bool equirectangular_view, const int band_rows) {
            std::vector<uint8_t> out(static_cast<size_t>(HEIGHT) * WIDTH * 3u);
            auto params = makeParams(equirectangular_view);
            for (int y0 = 0; y0 < HEIGHT; y0 += band_rows) {
                const int band_height = std::min(band_rows, HEIGHT - y0);
                const auto rgb_band = rgb_.slice(1, y0, y0 + band_height).contiguous();
                const auto alpha_band = alpha_.slice(0, y0, y0 + band_height).contiguous();
                params.y_offset = y0;
                Tensor band_u8;
                auto composited = lfs::rendering::compositeEnvironmentBand(
                    *env_map_, params, rgb_band, alpha_band, band_u8);
                EXPECT_TRUE(composited.has_value()) << (composited ? "" : composited.error());
                const auto band_cpu = band_u8.cpu();
                std::memcpy(out.data() + static_cast<size_t>(y0) * WIDTH * 3u,
                            band_cpu.ptr<uint8_t>(),
                            static_cast<size_t>(band_height) * WIDTH * 3u);
            }
            return out;
        }

        std::shared_ptr<const lfs::rendering::CudaEnvironmentMap> env_map_;
        lfs::rendering::EnvironmentImage env_cpu_;
        std::vector<float> rgb_host_;
        std::vector<float> alpha_host_;
        Tensor rgb_;
        Tensor alpha_;
    };

    TEST_F(ExportEnvCompositeTest, PerspectiveMatchesCpuReference) {
        const auto gpu = runCudaComposite(false, HEIGHT);
        const auto cpu = cpuReferenceComposite(env_cpu_, rgb_host_, alpha_host_, false);

        ASSERT_EQ(gpu.size(), cpu.size());
        int max_diff = 0;
        for (size_t i = 0; i < gpu.size(); ++i) {
            max_diff = std::max(max_diff, std::abs(static_cast<int>(gpu[i]) - static_cast<int>(cpu[i])));
        }
        EXPECT_LE(max_diff, 1);
    }

    TEST_F(ExportEnvCompositeTest, EquirectangularViewMatchesCpuReference) {
        const auto gpu = runCudaComposite(true, HEIGHT);
        const auto cpu = cpuReferenceComposite(env_cpu_, rgb_host_, alpha_host_, true);

        ASSERT_EQ(gpu.size(), cpu.size());
        int max_diff = 0;
        for (size_t i = 0; i < gpu.size(); ++i) {
            max_diff = std::max(max_diff, std::abs(static_cast<int>(gpu[i]) - static_cast<int>(cpu[i])));
        }
        EXPECT_LE(max_diff, 1);
    }

    TEST_F(ExportEnvCompositeTest, BandedEqualsFullImage) {
        const auto full = runCudaComposite(false, HEIGHT);
        const auto banded = runCudaComposite(false, 13);

        EXPECT_EQ(full, banded);
    }

    TEST_F(ExportEnvCompositeTest, UnpackPackRoundtripIsExact) {
        std::vector<uint8_t> pixels(static_cast<size_t>(HEIGHT) * WIDTH * 4u);
        for (size_t i = 0; i < pixels.size(); ++i) {
            pixels[i] = static_cast<uint8_t>((i * 7u + 3u) % 256u);
        }
        auto band = Tensor::from_blob(pixels.data(), {HEIGHT, WIDTH, 4},
                                      lfs::core::Device::CPU, lfs::core::DataType::UInt8)
                        .cuda();

        Tensor rgb_chw;
        Tensor alpha;
        auto unpacked = lfs::rendering::unpackU8HwcBandToChwFloat(band, rgb_chw, &alpha);
        ASSERT_TRUE(unpacked.has_value()) << (unpacked ? "" : unpacked.error());

        Tensor repacked;
        auto packed = lfs::rendering::packChwFloatBandToU8Hwc(rgb_chw, &alpha, repacked);
        ASSERT_TRUE(packed.has_value()) << (packed ? "" : packed.error());

        const auto original_cpu = band.cpu();
        const auto repacked_cpu = repacked.cpu();
        ASSERT_EQ(repacked_cpu.numel(), original_cpu.numel());
        EXPECT_EQ(std::memcmp(original_cpu.ptr<uint8_t>(), repacked_cpu.ptr<uint8_t>(), pixels.size()), 0);
    }

} // namespace
