/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/cuda/lanczos_resize/lanczos_resize.hpp"
#include "core/cuda/undistort/undistort.hpp"
#include "core/image_io.hpp"
#include "core/logger.hpp"
#include "core/tensor_backend.hpp"
#include "io/pipelined_image_loader.hpp"

#include <gtest/gtest.h>

#include <array>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#ifdef _WIN32
#include <process.h>
#else
#include <unistd.h>
#endif

namespace {

    using namespace lfs::core;
    using lfs::io::ImageRequest;
    using lfs::io::PipelinedImageLoader;
    using lfs::io::PipelinedLoaderConfig;

    constexpr int WIDTH = 13;
    constexpr int HEIGHT = 9;
    constexpr float UINT8_SCALE = 1.0f / 255.0f;
    constexpr float UINT16_SCALE = 1.0f / 65535.0f;

    std::filesystem::path fixture_directory() {
#ifdef _WIN32
        const auto pid = static_cast<unsigned long long>(_getpid());
#else
        const auto pid = static_cast<unsigned long long>(getpid());
#endif
        return std::filesystem::temp_directory_path() / ("lfs_loader_backends_" + std::to_string(pid));
    }

    // Deterministic samples that include both ends of the range.
    template <typename T>
    std::vector<T> samples(const int channels, const uint32_t maximum, const uint32_t seed) {
        std::vector<T> values(size_t(WIDTH) * HEIGHT * channels);
        for (size_t i = 0; i < values.size(); ++i)
            values[i] = T((i * 2654435761u + seed) % (maximum + 1));
        values[0] = 0;
        values[values.size() - 1] = T(maximum);
        return values;
    }

    // HWC samples to CHW floats with the loader's normalization.
    template <typename T>
    std::vector<float> planar(const std::vector<T>& values, const int channels, const float scale) {
        std::vector<float> result(values.size());
        for (size_t pixel = 0; pixel < size_t(WIDTH) * HEIGHT; ++pixel)
            for (int c = 0; c < channels; ++c)
                result[size_t(c) * WIDTH * HEIGHT + pixel] = float(values[pixel * channels + c]) * scale;
        return result;
    }

    Tensor host(const std::vector<float>& values, TensorShape shape) {
        return Tensor::from_vector(values, std::move(shape), Device::CPU);
    }

    UndistortParams distorted_camera() {
        UndistortParams p{};
        p.src_width = p.dst_width = WIDTH;
        p.src_height = p.dst_height = HEIGHT;
        p.src_fx = p.dst_fx = float(WIDTH);
        p.src_fy = p.dst_fy = float(HEIGHT);
        p.src_cx = p.dst_cx = WIDTH * 0.5f;
        p.src_cy = p.dst_cy = HEIGHT * 0.5f;
        p.model_type = CameraModelType::PINHOLE;
        p.num_distortion = 4;
        const float distortion[] = {0.12f, -0.03f, 0.004f, 0.002f};
        std::copy_n(distortion, 4, p.distortion);
        return p;
    }

    void expect_close(const Tensor& actual, const Tensor& expected, const float tolerance) {
        ASSERT_EQ(actual.shape(), expected.shape());
        const auto a = actual.cpu().to_vector();
        const auto b = expected.cpu().to_vector();
        for (size_t i = 0; i < a.size(); ++i) {
            if (tolerance == 0.0f)
                ASSERT_EQ(a[i], b[i]) << "element " << i;
            else
                ASSERT_NEAR(a[i], b[i], tolerance) << "element " << i;
        }
    }

    struct Loaded {
        Tensor image;
        Tensor mask;
        Tensor depth;
        Tensor normal;
    };

    class PipelinedLoaderBackends : public ::testing::TestWithParam<GpuBackend> {
    protected:
        static void SetUpTestSuite() {
            directory_ = fixture_directory();
            std::filesystem::create_directories(directory_);
            rgb8_ = samples<uint8_t>(3, 255, 1);
            rgb16_ = samples<uint16_t>(3, 65535, 7);
            rgba_ = samples<uint8_t>(4, 255, 3);
            mask_ = samples<uint8_t>(1, 255, 5);
            depth_ = samples<uint16_t>(1, 65535, 11);
            normal_ = samples<uint8_t>(3, 255, 13);
            ASSERT_TRUE(save_png(path("rgb8.png"), rgb8_.data(), WIDTH, HEIGHT, 3, 8, 1));
            ASSERT_TRUE(save_png(path("rgb16.png"), rgb16_.data(), WIDTH, HEIGHT, 3, 16, 1));
            ASSERT_TRUE(save_png(path("rgba.png"), rgba_.data(), WIDTH, HEIGHT, 4, 8, 1));
            ASSERT_TRUE(save_png(path("mask.png"), mask_.data(), WIDTH, HEIGHT, 1, 8, 1));
            ASSERT_TRUE(save_png(path("depth16.png"), depth_.data(), WIDTH, HEIGHT, 1, 16, 1));
            ASSERT_TRUE(save_png(path("normal.png"), normal_.data(), WIDTH, HEIGHT, 3, 8, 1));
        }

        static void TearDownTestSuite() {
            std::error_code ec;
            std::filesystem::remove_all(directory_, ec);
        }

        void SetUp() override {
            if (!gpu_backend_available(GetParam()))
                GTEST_SKIP() << gpu_backend_name(GetParam()) << " unavailable";
        }

        static std::filesystem::path path(const char* name) { return directory_ / name; }

        static ImageRequest request(const char* image) {
            ImageRequest result;
            result.sequence_id = 0;
            result.path = path(image);
            return result;
        }

        // Host copies are taken while the loader still owns its execution queues.
        Loaded load(const ImageRequest& request, const bool sixteen_bit = false) const {
            PipelinedLoaderConfig config;
            config.backend = GetParam();
            config.jpeg_batch_size = 1;
            config.prefetch_count = 2;
            config.output_queue_size = 2;
            config.decoder_pool_size = 1;
            config.io_threads = 1;
            config.cold_process_threads = 2;
            config.max_cache_bytes = 16u << 20;
            config.use_16bit_color = sixteen_bit;
            PipelinedImageLoader loader(config);
            loader.prefetch({request});
            auto completion = loader.get_completion();
            if (!completion || !completion->outcome) {
                ADD_FAILURE() << "load failed";
                return {};
            }
            auto& ready = *completion->outcome;
            for (auto* event : {&ready.depth_ready_event, &ready.normal_ready_event}) {
                if (*event)
                    (void)TensorFence::adopt(GpuBackend::CUDA, std::exchange(*event, nullptr));
            }
            EXPECT_TRUE(gpu_backend_of(ready.tensor) == GetParam());
            const auto copy = [](const std::optional<Tensor>& tensor) {
                return tensor ? tensor->cpu() : Tensor();
            };
            return {ready.tensor.cpu(), copy(ready.mask), copy(ready.depth), copy(ready.normal)};
        }

        static inline std::filesystem::path directory_;
        static inline std::vector<uint8_t> rgb8_, rgba_, mask_, normal_;
        static inline std::vector<uint16_t> rgb16_, depth_;
    };

    TEST_P(PipelinedLoaderBackends, ImagesMatchHostReference) {
        const TensorShape chw{3, HEIGHT, WIDTH};

        expect_close(load(request("rgb8.png")).image, host(planar(rgb8_, 3, UINT8_SCALE), chw), 0.0f);

        auto compact = request("rgb8.png");
        compact.params.output_uint8 = true;
        const auto bytes = load(compact).image;
        ASSERT_EQ(bytes.dtype(), DataType::UInt8);
        EXPECT_EQ(bytes.to_vector_uint8(), host(planar(rgb8_, 3, 1.0f), chw).to(DataType::UInt8).to_vector_uint8());

        expect_close(load(request("rgb16.png"), true).image, host(planar(rgb16_, 3, UINT16_SCALE), chw), 0.0f);

        auto resized = request("rgb8.png");
        resized.params.max_width = 7;
        auto [data, width, height, channels] = load_image(path("rgb8.png"), 1, 7);
        ASSERT_NE(data, nullptr);
        const Tensor expected = Tensor::from_blob(data, {size_t(height), size_t(width), 3}, Device::CPU, DataType::UInt8)
                                    .permute({2, 0, 1})
                                    .to(DataType::Float32)
                                    .mul(UINT8_SCALE)
                                    .contiguous();
        free_image(data);
        expect_close(load(resized).image, expected, 0.0f);

        auto undistorted = request("rgb8.png");
        const auto camera = distorted_camera();
        undistorted.undistort = &camera;
        undistorted.params.undistort = &camera;
        expect_close(load(undistorted).image,
                     undistort_image(host(planar(rgb8_, 3, UINT8_SCALE), chw), camera, nullptr),
                     1.5e-4f);
    }

    TEST_P(PipelinedLoaderBackends, MasksMatchHostReference) {
        const TensorShape hw{HEIGHT, WIDTH};
        std::vector<float> alpha(size_t(WIDTH) * HEIGHT);
        for (size_t i = 0; i < alpha.size(); ++i)
            alpha[i] = 1.0f - float(rgba_[i * 4 + 3]) * UINT8_SCALE;

        auto rgba = request("rgba.png");
        rgba.extract_alpha_as_mask = true;
        rgba.alpha_mask_params.invert = true;
        const auto split = load(rgba);
        std::vector<uint8_t> rgb(size_t(WIDTH) * HEIGHT * 3);
        for (size_t i = 0; i < size_t(WIDTH) * HEIGHT; ++i)
            std::copy_n(&rgba_[i * 4], 3, &rgb[i * 3]);
        expect_close(split.image, host(planar(rgb, 3, UINT8_SCALE), {3, HEIGHT, WIDTH}), 0.0f);
        expect_close(split.mask, host(alpha, hw), 0.0f);

        auto binary = request("rgb8.png");
        binary.mask_path = path("mask.png");
        binary.mask_params.threshold = 0.5f;
        std::vector<float> thresholded(mask_.size());
        for (size_t i = 0; i < mask_.size(); ++i)
            thresholded[i] = float(mask_[i]) * UINT8_SCALE >= 0.5f ? 1.0f : 0.0f;
        expect_close(load(binary).mask, host(thresholded, hw), 0.0f);

        auto soft = request("rgb8.png");
        soft.mask_path = path("mask.png");
        const auto camera = distorted_camera();
        soft.undistort = &camera;
        soft.params.undistort = &camera;
        expect_close(load(soft).mask,
                     undistort_mask(host(planar(mask_, 1, UINT8_SCALE), hw), camera, nullptr),
                     1.5e-4f);
    }

    TEST_P(PipelinedLoaderBackends, DepthAndNormalPriorsMatchHostReference) {
        auto depth = request("rgb8.png");
        depth.depth_path = path("depth16.png");
        depth.aux_target_width = 7;
        depth.aux_target_height = 5;
        const Tensor source_depth = host(planar(depth_, 1, UINT16_SCALE), {HEIGHT, WIDTH});
        expect_close(load(depth).depth, resize_depth_prior(source_depth, 5, 7), 1e-6f);

        // Orthonormal world-to-camera rotation, applied after the Y/Z flip.
        const std::array<float, 9> rotation{0.36f, 0.48f, -0.8f, -0.8f, 0.6f, 0.0f, 0.48f, 0.64f, 0.6f};
        for (const bool srgb : {false, true}) {
            auto normal = request("rgb8.png");
            normal.normal_path = path("normal.png");
            normal.normal_srgb = srgb;
            normal.normal_flip_yz = true;
            normal.normal_transform_world_to_camera = true;
            normal.normal_world_to_camera = rotation;
            std::vector<float> expected(normal_.size());
            for (size_t pixel = 0; pixel < size_t(WIDTH) * HEIGHT; ++pixel) {
                float n[3];
                for (int c = 0; c < 3; ++c) {
                    float v = float(normal_[pixel * 3 + c]) * UINT8_SCALE;
                    if (srgb)
                        v = srgb_encoding_to_linear(v);
                    n[c] = v * 2.0f - 1.0f;
                }
                n[1] = -n[1];
                n[2] = -n[2];
                for (int r = 0; r < 3; ++r)
                    expected[size_t(r) * WIDTH * HEIGHT + pixel] =
                        rotation[r * 3] * n[0] + rotation[r * 3 + 1] * n[1] + rotation[r * 3 + 2] * n[2];
            }
            expect_close(load(normal).normal,
                         resize_normal_prior(host(expected, {3, HEIGHT, WIDTH}), HEIGHT, WIDTH),
                         srgb ? 2e-5f : 1e-6f);
        }
    }

    INSTANTIATE_TEST_SUITE_P(Backends, PipelinedLoaderBackends,
                             ::testing::Values(GpuBackend::CUDA, GpuBackend::Vulkan),
                             [](const auto& info) { return std::string(gpu_backend_name(info.param)); });

    // Runs in a fresh process: an earlier codec probe elsewhere would otherwise hide one here.
    TEST(PipelinedLoaderVulkan, NeverProbesOrCreatesTheCudaCodec) {
        if (!gpu_backend_available(GpuBackend::Vulkan))
            GTEST_SKIP() << "Vulkan unavailable";
        const std::string style = GTEST_FLAG_GET(death_test_style);
        GTEST_FLAG_SET(death_test_style, "threadsafe");
        EXPECT_EXIT(
            {
                const auto directory = fixture_directory() / "codec";
                std::filesystem::create_directories(directory);
                const auto rgb = samples<uint8_t>(3, 255, 1);
                const auto gray = samples<uint8_t>(1, 255, 5);
                save_png(directory / "image.png", rgb.data(), WIDTH, HEIGHT, 3, 8, 1);
                save_png(directory / "mask.png", gray.data(), WIDTH, HEIGHT, 1, 8, 1);

                std::atomic<bool> codec_touched = false;
                std::atomic<bool> loader_logged = false;
                Logger::get().set_level(LogLevel::Info);
                const auto handler = Logger::get().add_log_handler(
                    [&](LogLevel, const SourceSite&, const std::string_view message) {
                        if (message.find("nvImageCodec") != std::string_view::npos ||
                            message.find("NvCodecImageLoader") != std::string_view::npos)
                            codec_touched = true;
                        if (message.find("[PipelinedImageLoader]") != std::string_view::npos)
                            loader_logged = true;
                    });
                bool delivered = false;
                {
                    PipelinedLoaderConfig config;
                    config.backend = GpuBackend::Vulkan;
                    PipelinedImageLoader loader(config);
                    ImageRequest request;
                    request.sequence_id = 0;
                    request.path = directory / "image.png";
                    request.mask_path = directory / "mask.png";
                    request.depth_path = directory / "mask.png";
                    request.normal_path = directory / "image.png";
                    loader.prefetch({request});
                    auto completion = loader.get_completion();
                    delivered = completion && completion->outcome && completion->outcome->mask &&
                                completion->outcome->depth && completion->outcome->normal;
                    (void)loader.load_image_immediate(request.path, {});
                }
                Logger::get().remove_log_handler(handler);
                std::filesystem::remove_all(directory);
                std::_Exit(delivered && loader_logged && !codec_touched ? 0 : 1);
            },
            ::testing::ExitedWithCode(0), "");
        GTEST_FLAG_SET(death_test_style, style);
    }

} // namespace
