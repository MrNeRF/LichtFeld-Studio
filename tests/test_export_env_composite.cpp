/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

// Each GPU export environment composite must match the shared CPU reference
// math (environment_math.hpp) that also drives the software video composite,
// and banded application must equal a single full-image pass.

#include "cuda_backend_test.hpp"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <cuda_runtime.h>
#include <filesystem>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <gtest/gtest.h>
#include <vector>

#include "core/environment_math.hpp"
#include "core/image_io.hpp"
#include "core/path_utils.hpp"
#include "core/tensor.hpp"
#include "core/tensor_backend.hpp"
#include "core/tensor_cuda_interop.hpp"
#include "environment_image.hpp"
#include "rendering/coordinate_conventions.hpp"
#include "rendering/export_post_process.hpp"
#include "tensor_test_support.hpp"
#include <optional>

namespace {

    using lfs::core::Tensor;
    namespace envmath = lfs::core::envmath;

    TEST(EnvironmentImageLoadTest, ExtractsRgbFromRgbaSource) {
        const auto path = std::filesystem::temp_directory_path() / "lfs_rgba_environment_test.png";
        const std::vector<uint8_t> rgba = {10, 20, 30, 255, 40, 50, 60, 128};
        ASSERT_TRUE(lfs::core::save_png(path, rgba.data(), 2, 1, 4, 8, 0));

        const auto loaded = lfs::rendering::loadEnvironmentImage(path);
        std::error_code ec;
        std::filesystem::remove(path, ec);
        if (!loaded)
            FAIL() << loaded.error();
        ASSERT_EQ(loaded->pixels, (std::vector<float>{10.0f / 255.0f, 20.0f / 255.0f,
                                                      30.0f / 255.0f, 40.0f / 255.0f,
                                                      50.0f / 255.0f, 60.0f / 255.0f}));
    }

    TEST(EnvironmentImageLoadTest, PreservesUnicodePath) {
        const auto directory = std::filesystem::temp_directory_path() /
                               lfs::core::utf8_to_path("lfs_environment_環境");
        const auto path = directory / lfs::core::utf8_to_path("背景_é.png");
        std::error_code ec;
        std::filesystem::remove_all(directory, ec);
        ASSERT_TRUE(std::filesystem::create_directories(directory, ec)) << ec.message();

        const std::vector<uint8_t> rgb = {10, 20, 30};
        ASSERT_TRUE(lfs::core::save_png(path, rgb.data(), 1, 1, 3, 8, 0));

        const auto loaded = lfs::rendering::loadEnvironmentImage(path);
        lfs::rendering::releaseEnvironmentImageCache();
        std::filesystem::remove_all(directory, ec);

        ASSERT_TRUE(loaded.has_value()) << (loaded ? "" : loaded.error());
        EXPECT_EQ(loaded->path, path);
        EXPECT_EQ(loaded->pixels, (std::vector<float>{10.0f / 255.0f,
                                                      20.0f / 255.0f,
                                                      30.0f / 255.0f}));
    }

    TEST(EnvironmentImageLoadTest, ReportsMissingUnicodePathAsUtf8) {
        const auto path = std::filesystem::temp_directory_path() /
                          lfs::core::utf8_to_path("missing_environment_環境.hdr");
        std::error_code ec;
        std::filesystem::remove(path, ec);

        const auto loaded = lfs::rendering::loadEnvironmentImage(path);

        ASSERT_FALSE(loaded.has_value());
        EXPECT_NE(loaded.error().find(lfs::core::path_to_utf8(path)), std::string::npos);
    }

    const auto kEnvironmentAsset = std::filesystem::path(PROJECT_ROOT_PATH) /
                                   "src/visualizer/gui/assets/environments/alps_field_1k.hdr";
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

    class ExportEnvCompositeTest : public ::testing::TestWithParam<lfs::core::GpuBackend> {
    protected:
        void SetUp() override {
            if (!lfs::core::gpu_backend_available(GetParam()))
                GTEST_SKIP() << "Requested GPU backend unavailable";
            backend_scope_.emplace(GetParam());
            ASSERT_TRUE(std::filesystem::exists(kEnvironmentAsset)) << kEnvironmentAsset;
            auto loaded = lfs::rendering::getOrLoadEnvironmentMap(kEnvironmentAsset, GetParam());
            ASSERT_TRUE(loaded.has_value()) << loaded.error();
            env_map_ = *loaded;
            auto cpu = lfs::rendering::loadEnvironmentImage(kEnvironmentAsset);
            ASSERT_TRUE(cpu.has_value()) << cpu.error();
            env_cpu_ = std::move(*cpu);

            rgb_host_ = makeRgbChw();
            alpha_host_ = makeAlpha();
            rgb_ = Tensor::from_vector(rgb_host_, {3, HEIGHT, WIDTH}, lfs::core::Device::GPU);
            alpha_ = Tensor::from_vector(alpha_host_, {HEIGHT, WIDTH}, lfs::core::Device::GPU);
            ASSERT_TRUE(rgb_.is_valid());
            ASSERT_TRUE(alpha_.is_valid());
        }

        void TearDown() override {
            lfs::rendering::releaseEnvironmentMapCaches();
            env_map_.reset();
            rgb_ = {};
            alpha_ = {};
            if (GetParam() == lfs::core::GpuBackend::Vulkan) {
                EXPECT_TRUE(lfs::core::shutdown_gpu_backend(lfs::core::GpuBackend::Vulkan));
                for (const auto& message : lfs::test::vulkan_validation_messages())
                    ADD_FAILURE() << message;
            }
        }

        [[nodiscard]] std::vector<uint8_t> runComposite(const bool equirectangular_view, const int band_rows) {
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

        std::shared_ptr<const lfs::rendering::EnvironmentMap> env_map_;
        lfs::rendering::EnvironmentImage env_cpu_;
        std::vector<float> rgb_host_;
        std::vector<float> alpha_host_;
        Tensor rgb_;
        Tensor alpha_;
        std::optional<lfs::core::GpuBackendScope> backend_scope_;
    };

    TEST_P(ExportEnvCompositeTest, PerspectiveMatchesCpuReference) {
        const auto gpu = runComposite(false, HEIGHT);
        const auto cpu = cpuReferenceComposite(env_cpu_, rgb_host_, alpha_host_, false);

        ASSERT_EQ(gpu.size(), cpu.size());
        int max_diff = 0;
        for (size_t i = 0; i < gpu.size(); ++i) {
            max_diff = std::max(max_diff, std::abs(static_cast<int>(gpu[i]) - static_cast<int>(cpu[i])));
        }
        EXPECT_LE(max_diff, 1);
    }

    TEST_P(ExportEnvCompositeTest, EquirectangularViewMatchesCpuReference) {
        const auto gpu = runComposite(true, HEIGHT);
        const auto cpu = cpuReferenceComposite(env_cpu_, rgb_host_, alpha_host_, true);

        ASSERT_EQ(gpu.size(), cpu.size());
        int max_diff = 0;
        for (size_t i = 0; i < gpu.size(); ++i) {
            max_diff = std::max(max_diff, std::abs(static_cast<int>(gpu[i]) - static_cast<int>(cpu[i])));
        }
        EXPECT_LE(max_diff, 1);
    }

    TEST_P(ExportEnvCompositeTest, BandedEqualsFullImage) {
        const auto full = runComposite(false, HEIGHT);
        const auto banded = runComposite(false, 13);

        EXPECT_EQ(full, banded);
    }

    TEST_P(ExportEnvCompositeTest, UnpackPackRoundtripIsExact) {
        std::vector<uint8_t> pixels(static_cast<size_t>(HEIGHT) * WIDTH * 4u);
        for (size_t i = 0; i < pixels.size(); ++i) {
            pixels[i] = static_cast<uint8_t>((i * 7u + 3u) % 256u);
        }
        auto band = Tensor::from_blob(pixels.data(), {HEIGHT, WIDTH, 4},
                                      lfs::core::Device::CPU, lfs::core::DataType::UInt8)
                        .gpu();

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

    TEST_P(ExportEnvCompositeTest, RetainsStorageBackendWhenTheDefaultChanges) {
        const lfs::core::GpuBackendScope other(GetParam() == lfs::core::GpuBackend::CUDA
                                                   ? lfs::core::GpuBackend::Vulkan
                                                   : lfs::core::GpuBackend::CUDA);
        Tensor output;
        const auto status = lfs::rendering::compositeEnvironmentBand(*env_map_, makeParams(false), rgb_, alpha_, output);
        ASSERT_TRUE(status) << status.error();
        EXPECT_EQ(lfs::core::gpu_backend_of(output), GetParam());
        const auto expected = cpuReferenceComposite(env_cpu_, rgb_host_, alpha_host_, false);
        const auto actual = output.to_vector_uint8();
        ASSERT_EQ(expected.size(), actual.size());
        int max_diff = 0;
        for (size_t i = 0; i < actual.size(); ++i)
            max_diff = std::max(max_diff, std::abs(static_cast<int>(actual[i]) - expected[i]));
        EXPECT_LE(max_diff, 1);
    }

    TEST_P(ExportEnvCompositeTest, RejectsMalformedEnvironmentAndBand) {
        auto malformed = *env_map_;
        malformed.width += 1;
        Tensor output;
        EXPECT_FALSE(lfs::rendering::compositeEnvironmentBand(malformed, makeParams(false), rgb_, alpha_, output));
        auto params = makeParams(false);
        params.y_offset = HEIGHT;
        EXPECT_FALSE(lfs::rendering::compositeEnvironmentBand(*env_map_, params, rgb_, alpha_, output));
    }

    class ExportEnvironmentCudaCache : public lfs::test::CudaBackendTest {};

    TEST_F(ExportEnvironmentCudaCache, PublishedMapOutlivesItsUploadStream) {
        lfs::rendering::releaseEnvironmentMapCaches();
        cudaStream_t producer = nullptr;
        ASSERT_EQ(cudaStreamCreateWithFlags(&producer, cudaStreamNonBlocking), cudaSuccess);
        std::shared_ptr<const lfs::rendering::EnvironmentMap> map;
        {
            const lfs::core::CUDAStreamGuard guard(producer);
            const auto loaded = lfs::rendering::getOrLoadEnvironmentMap(kEnvironmentAsset, lfs::core::GpuBackend::CUDA);
            ASSERT_TRUE(loaded);
            map = *loaded;
        }
        EXPECT_EQ(map->pixels.stream(), nullptr);
        lfs::core::release_cuda_stream(producer);
        ASSERT_EQ(cudaStreamDestroy(producer), cudaSuccess);
        auto rgb = Tensor::full({3, 1, 1}, 0.2f, lfs::core::Device::GPU);
        auto alpha = Tensor::ones({1, 1}, lfs::core::Device::GPU);
        lfs::rendering::EnvironmentCompositeBandParams params;
        params.full_size = {1, 1};
        params.focal_x = params.focal_y = 1;
        Tensor output;
        ASSERT_TRUE(lfs::rendering::compositeEnvironmentBand(*map, params, rgb, alpha, output));
        EXPECT_EQ(output.to_vector_uint8(), (std::vector<uint8_t>{51, 51, 51}));
        lfs::rendering::releaseEnvironmentMapCaches();
    }

    TEST(ExportEnvironmentCache, KeepsCudaAndVulkanMapsSeparate) {
        using namespace lfs::core;
        if (!gpu_backend_available(GpuBackend::CUDA) || !gpu_backend_available(GpuBackend::Vulkan))
            GTEST_SKIP() << "Requires both GPU backends";
        auto cuda = lfs::rendering::getOrLoadEnvironmentMap(kEnvironmentAsset, GpuBackend::CUDA);
        auto vulkan = lfs::rendering::getOrLoadEnvironmentMap(kEnvironmentAsset, GpuBackend::Vulkan);
        ASSERT_TRUE(cuda) << cuda.error();
        ASSERT_TRUE(vulkan) << vulkan.error();
        EXPECT_NE(*cuda, *vulkan);
        EXPECT_EQ(gpu_backend_of((*cuda)->pixels), GpuBackend::CUDA);
        EXPECT_EQ(gpu_backend_of((*vulkan)->pixels), GpuBackend::Vulkan);
        EXPECT_EQ((*cuda)->pixels.to_vector(), (*vulkan)->pixels.to_vector());
        EXPECT_EQ(*lfs::rendering::getOrLoadEnvironmentMap(kEnvironmentAsset, GpuBackend::CUDA), *cuda);
        EXPECT_EQ(*lfs::rendering::getOrLoadEnvironmentMap(kEnvironmentAsset, GpuBackend::Vulkan), *vulkan);
        lfs::rendering::releaseEnvironmentMapCaches();
        // Active users retain their maps across a cache release.
        EXPECT_EQ((*cuda)->pixels.to_vector(), (*vulkan)->pixels.to_vector());
    }

    INSTANTIATE_TEST_SUITE_P(Backends, ExportEnvCompositeTest,
                             ::testing::ValuesIn(lfs::core::kGpuBackends),
                             [](const ::testing::TestParamInfo<lfs::core::GpuBackend>& info) {
                                 return info.param == lfs::core::GpuBackend::CUDA ? "CUDA" : "Vulkan";
                             });
} // namespace
