/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/color_transfer.h"
#include "core/cuda/lanczos_resize/lanczos_resize.hpp"
#include "core/cuda/undistort/undistort.hpp"
#include "core/image_codecs.hpp"
#include "core/image_exr.hpp"
#include "core/image_io.hpp"
#include "core/logger.hpp"
#include "core/path_utils.hpp"
#include "core/shared_image_ops.hpp"
#include "core/tensor_backend.hpp"
#include "io/cuda/masked_image_fill.hpp"
#include "io/pipelined_image_loader.hpp"

#include <gtest/gtest.h>

#include <array>
#include <atomic>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <future>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#ifdef _WIN32
#include <process.h>
#else
#include <unistd.h>
#endif

namespace lfs::io {
    struct PipelinedPreparedCacheTestAccess {
        static std::string key(const PipelinedImageLoader& loader, const std::filesystem::path& path, const LoadParams& params) {
            return loader.make_cache_key(path, params);
        }
        static auto blob(PipelinedImageLoader& loader, const std::string& key) { return loader.load_cached_jpeg_blob(key); }
        static auto backend(const PipelinedImageLoader& loader) { return loader.config_.backend; }
        static auto png_encoded_limit() { return PipelinedImageLoader::PORTABLE_PNG_CACHE_MAX_BYTES; }
        static bool eligible(const PipelinedImageLoader& loader, const LoadParams& params) {
            return loader.uses_portable_prepared_cache(params);
        }
        static void put(PipelinedImageLoader& loader, const std::string& key, std::vector<uint8_t> bytes) {
            loader.put_in_jpeg_cache(key, std::make_shared<std::vector<uint8_t>>(std::move(bytes)));
        }
        static void encode(PipelinedImageLoader& loader, const std::string& key, const lfs::core::Tensor& image, size_t limit, bool lossless = false) {
            loader.cache_portable_prepared_image(key, image, limit, lossless);
        }
    };
} // namespace lfs::io

namespace {

    using namespace lfs::core;
    using lfs::io::ImageRequest;
    using lfs::io::PipelinedImageLoader;
    using lfs::io::PipelinedLoaderConfig;

    constexpr int WIDTH = 13;
    constexpr int HEIGHT = 9;
    constexpr float UINT8_SCALE = 1.0f / 255.0f;
    constexpr float UINT16_SCALE = 1.0f / 65535.0f;

    class DebugLogHandler {
    public:
        explicit DebugLogHandler(LogHandler handler)
            : level_(Logger::get().level()), token_(Logger::get().add_log_handler(std::move(handler))) {
            Logger::get().set_level(LogLevel::Debug);
        }
        ~DebugLogHandler() {
            Logger::get().remove_log_handler(token_);
            Logger::get().set_level(level_);
        }
        DebugLogHandler(const DebugLogHandler&) = delete;
        DebugLogHandler& operator=(const DebugLogHandler&) = delete;

    private:
        LogLevel level_;
        LogHandlerToken token_;
    };

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
            ASSERT_TRUE(image_codecs::write_tiff_u16(
                path("rgb16.tiff"),
                {reinterpret_cast<const uint8_t*>(rgb16_.data()), rgb16_.size() * sizeof(uint16_t)},
                WIDTH, HEIGHT, 3, WIDTH * 3 * sizeof(uint16_t), std::nullopt,
                {.overwrite = true, .durable = false}));
            ASSERT_TRUE(save_png(path("rgba.png"), rgba_.data(), WIDTH, HEIGHT, 4, 8, 1));
            ASSERT_TRUE(save_png(path("mask.png"), mask_.data(), WIDTH, HEIGHT, 1, 8, 1));
            ASSERT_TRUE(save_png(path("depth16.png"), depth_.data(), WIDTH, HEIGHT, 1, 16, 1));
            ASSERT_TRUE(save_png(path("normal.png"), normal_.data(), WIDTH, HEIGHT, 3, 8, 1));
            std::vector<float> floating(rgb16_.size());
            for (size_t i = 0; i < floating.size(); ++i)
                floating[i] = float(rgb16_[i]) / 16384.f - 0.5f;
            image_codecs::ExrEncodeOptions options;
            options.precision = image_codecs::ExrSampleType::Float;
            std::vector<uint8_t> exr;
            ASSERT_TRUE(image_codecs::encode_exr_memory(
                {WIDTH, HEIGHT, 3, WIDTH * 3 * sizeof(float), {reinterpret_cast<const uint8_t*>(floating.data()), floating.size() * sizeof(float)}},
                exr, options));
            std::ofstream output(path("rgbfloat.exr"), std::ios::binary);
            output.write(reinterpret_cast<const char*>(exr.data()), exr.size());
            ASSERT_TRUE(output);
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
            auto completion = loader.try_get_completion_for(std::chrono::seconds(20));
            if (!completion || !completion->outcome) {
                ADD_FAILURE() << "load failed";
                return {};
            }
            auto& ready = *completion->outcome;
            for (auto* fence : {&ready.depth_ready, &ready.normal_ready}) {
                if (*fence)
                    (*fence)->wait();
            }
            EXPECT_TRUE(gpu_backend_of(ready.tensor) == GetParam());
            // Exercise a consumer GPU operation on a different thread before readback.
            const auto copy = [](const Tensor& tensor) {
                return gpu_backend_of(tensor) == GpuBackend::Vulkan ? tensor.clone().cpu() : tensor.cpu();
            };
            return {copy(ready.tensor), ready.mask ? copy(*ready.mask) : Tensor(),
                    ready.depth ? copy(*ready.depth) : Tensor(), ready.normal ? copy(*ready.normal) : Tensor()};
        }

        static inline std::filesystem::path directory_;
        static inline std::vector<uint8_t> rgb8_, rgba_, mask_, normal_;
        static inline std::vector<uint16_t> rgb16_, depth_;
    };

    TEST_P(PipelinedLoaderBackends, EvaluationFloatReferencePreservesGraySamples) {
        const GpuBackendScope scope(GetParam());
        // Minimal one-channel Float32 TIFF: values are off both integer grids.
        std::vector<uint8_t> bytes{'I', 'I', 42, 0, 8, 0, 0, 0};
        const auto append = [&](uint32_t value, int count) {
            for (int i = 0; i < count; ++i)
                bytes.push_back(static_cast<uint8_t>(value >> (8 * i)));
        };
        append(10, 2);
        const auto tag = [&](uint16_t key, uint16_t type, uint32_t value) {
            append(key, 2);
            append(type, 2);
            append(1, 4);
            append(value, 4);
        };
        tag(256, 3, 2);
        tag(257, 3, 1);
        tag(258, 3, 32);
        tag(259, 3, 1);
        tag(262, 3, 1);
        tag(273, 4, 134);
        tag(277, 3, 1);
        tag(278, 4, 1);
        tag(279, 4, 8);
        tag(339, 3, 3);
        append(0, 4);
        constexpr float a = 0.123456f, b = 0.654321f;
        append(std::bit_cast<uint32_t>(a), 4);
        append(std::bit_cast<uint32_t>(b), 4);
        const auto file = path("eval-float-gray.tiff");
        {
            std::ofstream out(file, std::ios::binary);
            out.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
            ASSERT_TRUE(out.good());
        }
        PipelinedLoaderConfig config;
        config.backend = GetParam();
        PipelinedImageLoader loader(config);
        lfs::io::LoadParams params;
        params.decode_float = true;
        params.decode_16bit = true;
        params.skip_blob_cache = true;
        const auto image = loader.load_image_immediate(file, params);
        ASSERT_EQ(image.dtype(), DataType::Float32);
        expect_close(image, host({a, b, a, b, a, b}, {3, 1, 2}), 0.0f);
    }

    TEST_P(PipelinedLoaderBackends, DecodeAheadMatchesImmediateOnEachBackend) {
        const GpuBackendScope scope(GetParam());
        auto distortion = distorted_camera();
        for (const bool sixteen_bit : {false, true}) {
            for (const bool undistort : {false, true}) {
                PipelinedLoaderConfig config;
                config.backend = GetParam();
                config.use_16bit_color = sixteen_bit;
                config.io_threads = 1;
                config.cold_process_threads = 1;
                config.decoder_pool_size = 1;
                lfs::io::LoadParams params;
                params.max_width = WIDTH / 2;
                params.output_uint8 = false;
                params.undistort = undistort ? &distortion : nullptr;
                PipelinedImageLoader plain(config), ahead(config);
                const auto expected = plain.load_image_immediate(path(sixteen_bit ? "rgb16.png" : "rgb8.png"), params).cpu().contiguous();
                ahead.decode_ahead(path(sixteen_bit ? "rgb16.png" : "rgb8.png"), params);
                const auto actual = ahead.load_image_immediate(path(sixteen_bit ? "rgb16.png" : "rgb8.png"), params).cpu().contiguous();
                ASSERT_EQ(actual.shape(), expected.shape());
                ASSERT_EQ(actual.dtype(), expected.dtype());
                EXPECT_EQ(actual.to_vector(), expected.to_vector());
            }
        }
    }

    TEST_P(PipelinedLoaderBackends, ImagesMatchHostReference) {
        const TensorShape chw{3, HEIGHT, WIDTH};

        expect_close(load(request("rgb8.png")).image, host(planar(rgb8_, 3, UINT8_SCALE), chw), 0.0f);

        auto compact = request("rgb8.png");
        compact.params.output_uint8 = true;
        const auto bytes = load(compact).image;
        ASSERT_EQ(bytes.dtype(), DataType::UInt8);
        EXPECT_EQ(bytes.to_vector_uint8(), host(planar(rgb8_, 3, 1.0f), chw).to(DataType::UInt8).to_vector_uint8());

        expect_close(load(request("rgb16.png"), true).image, host(planar(rgb16_, 3, UINT16_SCALE), chw), 0.0f);

        // CPU-decoded images are downscaled on the GPU with a size-2 Lanczos kernel.
        auto resized = request("rgb8.png");
        resized.params.max_width = 7;
        const auto [target_width, target_height] = resized_image_dimensions(WIDTH, HEIGHT, 1, 7);
        Tensor expected;
        {
            GpuBackendScope scope(GetParam());
            const auto source = Tensor::from_blob(rgb8_.data(), {size_t(HEIGHT), size_t(WIDTH), 3}, Device::CPU,
                                                  DataType::UInt8)
                                    .to(Device::GPU);
            expected = shared_image_ops(GetParam())
                           ->resize(source, target_height, target_width, lfs::gpu_ops::Resample::LanczosRGB, 2)
                           .cpu();
        }
        expect_close(load(resized).image, expected, 0.0f);

        auto undistorted = request("rgb8.png");
        const auto camera = distorted_camera();
        undistorted.undistort = &camera;
        undistorted.params.undistort = &camera;
        // Warped RGB is bounded and quantized to the lossless 16-bit cache grid.
        const auto warped = undistort_image(host(planar(rgb8_, 3, UINT8_SCALE), chw), camera, nullptr)
                                .clamp(0.0f, 1.0f)
                                .mul(65535.0f)
                                .add(0.5f)
                                .to(DataType::Int32)
                                .to(DataType::Float32)
                                .mul(UINT16_SCALE);
        expect_close(load(undistorted).image, warped, 1.5e-4f);
    }

    TEST_P(PipelinedLoaderBackends, MasksMatchHostReference) {
        const TensorShape hw{HEIGHT, WIDTH};
        std::vector<float> alpha(size_t(WIDTH) * HEIGHT);
        for (size_t i = 0; i < alpha.size(); ++i)
            alpha[i] = 1.0f - float(rgba_[i * 4 + 3]) * UINT8_SCALE;

        auto rgba = request("rgba.png");
        rgba.extract_alpha_as_mask = true;
        rgba.alpha_mask_params.invert = true;
        rgba.fill_masked_rgb = false;
        const auto split = load(rgba);
        std::vector<uint8_t> rgb(size_t(WIDTH) * HEIGHT * 3);
        for (size_t i = 0; i < size_t(WIDTH) * HEIGHT; ++i)
            std::copy_n(&rgba_[i * 4], 3, &rgb[i * 3]);
        expect_close(split.image, host(planar(rgb, 3, UINT8_SCALE), {3, HEIGHT, WIDTH}), 0.0f);
        // A driver may contract 1 - a / 255 into one fma (MoltenVK does); the
        // host reference rounds twice, so allow the last bit.
        expect_close(split.mask, host(alpha, hw), 1e-6f);

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
        Tensor expected_soft_mask;
        {
            const GpuBackendScope scope(GetParam());
            expected_soft_mask = undistort_mask(
                host(planar(mask_, 1, UINT8_SCALE), hw).to(Device::GPU), camera, nullptr);
            shared_image_ops(GetParam())->mask(expected_soft_mask, lfs::gpu_ops::MaskTransform::QuantizeByteGrid, 0.0f);
        }
        expect_close(load(soft).mask, expected_soft_mask.cpu(), 0.0f);
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
            normal.aux_target_width = 7;
            normal.aux_target_height = 5;
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
                         resize_normal_prior(host(expected, {3, HEIGHT, WIDTH}), 5, 7),
                         2e-5f);
        }
    }

    using CacheAccess = lfs::io::PipelinedPreparedCacheTestAccess;

    Tensor decoded_prepared_blob(PipelinedImageLoader& loader, const std::filesystem::path& file,
                                 const lfs::io::LoadParams& params);

    TEST_P(PipelinedLoaderBackends, JxlSourcesUseExistingLoaderAndPreparedCachePolicies) {
        const GpuBackendScope scope(GetParam());
        for (const bool wide : {false, true}) {
            const auto file = path(wide ? "source16.jxl" : "source8.jxl");
            std::filesystem::copy_file(std::filesystem::path(LFS_JXL_FIXTURES) / (wide ? "rgb16.jxl" : "rgb8.jxl"), file,
                                       std::filesystem::copy_options::overwrite_existing);
            PipelinedLoaderConfig config;
            config.min_spill_free_bytes = 0; // spill regardless of free space in the temp dir
            config.backend = GetParam();
            PipelinedImageLoader loader(config);
            lfs::io::LoadParams params;
            params.decode_16bit = wide;
            const auto image = loader.load_image_immediate(file, params).cpu();
            ASSERT_EQ(image.shape(), (TensorShape{3, 2, 2}));
            EXPECT_NEAR(image.ptr<float>()[0], 1.f, 2e-7f);
            EXPECT_NEAR(image.ptr<float>()[4], wide ? 32768.f / 65535.f : 0.f, 2e-7f);
            EXPECT_EQ(image.ptr<float>()[8], 0.f);
            if (GetParam() != GpuBackend::CUDA) {
                auto camera = distorted_camera();
                params.undistort = &camera;
                const auto cold = loader.load_image_immediate(file, params).cpu();
                // The existing RGB8 prepared cache is JPEG; wide samples use
                // lossless PNG16. Compare warm/spill with the actual cache
                // representation, not an assumption that JPEG is lossless.
                const auto cached = decoded_prepared_blob(loader, file, params);
                if (wide)
                    expect_close(cached, cold, 0.f);
                expect_close(loader.load_image_immediate(file, params), cached, 0.f);
                EXPECT_EQ(loader.get_stats().jpeg_cache_entries, 1u);
                EXPECT_EQ(loader.get_stats().hot_path_hits, 1u);
                EXPECT_GT(loader.release_host_cache(std::numeric_limits<size_t>::max()), 0u);
                EXPECT_EQ(loader.get_stats().jpeg_cache_bytes, 0u);
                EXPECT_EQ(loader.get_stats().spill_cache_entries, 1u);
                expect_close(loader.load_image_immediate(file, params), cached, 0.f);
                EXPECT_EQ(loader.get_stats().hot_path_hits, 2u);
            }
        }
    }

    TEST_P(PipelinedLoaderBackends, JxlFloatPreparedCachePreservesWarmAndSpilledSamples) {
        if (GetParam() == GpuBackend::CUDA)
            GTEST_SKIP() << "Float prepared-cache policy belongs to the portable loader";
        const GpuBackendScope scope(GetParam());
        const auto file = path("float.jxl");
        std::filesystem::copy_file(std::filesystem::path(LFS_JXL_FIXTURES) / "rgb32.jxl", file,
                                   std::filesystem::copy_options::overwrite_existing);
        PipelinedLoaderConfig config;
        config.min_spill_free_bytes = 0; // spill regardless of free space in the temp dir
        config.backend = GetParam();
        PipelinedImageLoader loader(config);
        lfs::io::LoadParams params;
        params.decode_float = true;
        const auto original = loader.load_image_immediate(file, params).cpu();
        ASSERT_EQ(original.shape(), (TensorShape{3, 2, 2}));
        // libjxl uses sign-preserving sRGB extension outside the unit range.
        // Its SIMD approximation need not match the host pow implementation
        // in the last bits; cache restoration below remains bit-exact.
        EXPECT_NEAR(original.ptr<float>()[0], lfs::core::color::linearToSrgb(2.f), 1e-4f);
        EXPECT_NEAR(original.ptr<float>()[4], lfs::core::color::linearToSrgb(.5f), 1e-4f);
        EXPECT_NEAR(original.ptr<float>()[8], -lfs::core::color::linearToSrgb(.25f), 1e-4f);
        auto camera = distorted_camera();
        params.undistort = &camera;
        const auto cold = loader.load_image_immediate(file, params).cpu();
        expect_close(loader.load_image_immediate(file, params), cold, 0.f);
        EXPECT_EQ(loader.get_stats().hot_path_hits, 1u);
        EXPECT_GT(loader.release_host_cache(std::numeric_limits<size_t>::max()), 0u);
        EXPECT_EQ(loader.get_stats().spill_cache_entries, 1u);
        expect_close(loader.load_image_immediate(file, params), cold, 0.f);
        EXPECT_EQ(loader.get_stats().hot_path_hits, 2u);
    }

    TEST_P(PipelinedLoaderBackends, CompressedCacheBudgetCountsReservedCapacityAndSpillsOnlyPayload) {
        const GpuBackendScope scope(GetParam());
        std::vector<uint8_t> small{1, 2, 3};
        small.reserve(64);
        const auto budget = small.capacity();
        PipelinedLoaderConfig config;
        config.min_spill_free_bytes = 0; // spill regardless of free space in the temp dir
        config.backend = GetParam();
        config.max_cache_bytes = budget;
        PipelinedImageLoader loader(config);
        std::vector<uint8_t> large{1, 2, 3};
        large.reserve(budget * 8);
        CacheAccess::put(loader, "capacity-test", std::move(large));
        EXPECT_EQ(loader.get_stats().jpeg_cache_bytes, 0u);
        EXPECT_EQ(loader.get_stats().spill_cache_bytes, 3u);
        ASSERT_TRUE(CacheAccess::blob(loader, "capacity-test"));
        EXPECT_EQ(*CacheAccess::blob(loader, "capacity-test"), (std::vector<uint8_t>{1, 2, 3}));
        CacheAccess::put(loader, "capacity-test", std::move(small));
        EXPECT_EQ(loader.get_stats().jpeg_cache_bytes, budget);
        EXPECT_EQ(loader.get_stats().spill_cache_bytes, 0u);
        EXPECT_EQ(loader.release_host_cache(1), budget);
        EXPECT_EQ(loader.get_stats().jpeg_cache_bytes, 0u);
        EXPECT_EQ(loader.get_stats().spill_cache_bytes, 3u);
        for (uintmax_t changed_size : {uintmax_t(1), uintmax_t(1024 * 1024)}) {
            std::vector<uint8_t> bytes{1, 2, 3};
            bytes.reserve(budget * 8);
            CacheAccess::put(loader, "capacity-test", std::move(bytes));
            const auto directory = loader.run_spill_directory();
            const auto entry = std::filesystem::directory_iterator(directory);
            ASSERT_NE(entry, std::filesystem::directory_iterator{});
            std::filesystem::resize_file(entry->path(), changed_size);
            EXPECT_FALSE(CacheAccess::blob(loader, "capacity-test"));
            EXPECT_EQ(loader.get_stats().spill_cache_bytes, 0u);
        }
    }

    TEST_P(PipelinedLoaderBackends, PreparedLosslessCacheBoundsAdditionalStagingBeforeEncoding) {
        if (GetParam() == GpuBackend::CUDA)
            GTEST_SKIP();
        const GpuBackendScope scope(GetParam());
        PipelinedLoaderConfig config;
        config.backend = GetParam();
        PipelinedImageLoader loader(config);
        auto camera = distorted_camera();
        camera.src_width = camera.dst_width = 3840;
        camera.src_height = camera.dst_height = 2160;
        lfs::io::LoadParams params;
        params.undistort = &camera;
        params.decode_16bit = true;
        EXPECT_TRUE(CacheAccess::eligible(loader, params)); // PNG16 staging fits at full 4K.
        params.decode_float = true;
        EXPECT_FALSE(CacheAccess::eligible(loader, params)); // EXR retains Float32 staging.
        params.decode_float = false;
        camera.dst_width = 7680;
        camera.dst_height = 4320;
        EXPECT_FALSE(CacheAccess::eligible(loader, params));
        camera.dst_width = 3840;
        camera.dst_height = 2160;
        params.resize_factor = 2;
        EXPECT_TRUE(CacheAccess::eligible(loader, params));
        params.resize_factor = 1;
        params.max_width = 1920;
        EXPECT_TRUE(CacheAccess::eligible(loader, params));
        params.max_width = 0;
        params.decode_16bit = false;
        EXPECT_TRUE(CacheAccess::eligible(loader, params)); // Existing JPEG policy is unchanged.
    }

    TEST_P(PipelinedLoaderBackends, PreparedLosslessCachesPreserveColdValuesAndOwnWarmImages) {
        if (GetParam() == GpuBackend::CUDA)
            GTEST_SKIP() << "Portable cache; CUDA retains its existing cache policy";
        const GpuBackendScope scope(GetParam());
        for (int mode : {0, 1, 2, 3}) {
            SCOPED_TRACE(mode);
            PipelinedLoaderConfig config;
            config.backend = GetParam();
            config.use_16bit_color = mode == 1;
            PipelinedImageLoader loader(config);
            auto camera = distorted_camera();
            lfs::io::LoadParams params;
            params.undistort = &camera;
            params.decode_16bit = mode == 0;
            params.decode_float = mode >= 2;
            const auto file = path(mode == 2 ? "rgbfloat.exr" : mode == 1 ? "rgb16.tiff"
                                                                          : "rgb16.png");
            params.skip_blob_cache = true;
            const auto reference = loader.load_image_immediate(file, params).cpu();
            EXPECT_EQ(loader.get_stats().jpeg_cache_entries, 0u);
            params.skip_blob_cache = false;
            auto cold = loader.load_image_immediate(file, params);
            expect_close(cold, reference, 0.f);
            const auto blob = CacheAccess::blob(loader, CacheAccess::key(loader, file, params));
            ASSERT_TRUE(blob);
            ASSERT_GE(blob->size(), 4u);
            EXPECT_EQ((*blob)[0], mode >= 2 ? 0x76 : 0x89);
            EXPECT_EQ((*blob)[1], mode >= 2 ? 0x2f : 0x50);
            EXPECT_EQ(blob->capacity(), blob->size());
            EXPECT_EQ(loader.get_stats().jpeg_cache_entries, 1u);
            cold.fill_(0.f);
            auto warm = loader.load_image_immediate(file, params);
            expect_close(warm, reference, 0.f);
            EXPECT_EQ(loader.get_stats().hot_path_hits, 1u);
            warm.fill_(1.f);
            expect_close(loader.load_image_immediate(file, params), reference, 0.f);
            params.output_uint8 = true;
            EXPECT_EQ(loader.load_image_immediate(file, params).dtype(), DataType::UInt8);
            EXPECT_EQ(loader.get_stats().jpeg_cache_entries, 1u); // Precision-to-byte conversion is not cached here.
        }
    }

    TEST_P(PipelinedLoaderBackends, PreparedPngCacheRestoresEveryUInt16ValueWithGpuDivision) {
        if (GetParam() == GpuBackend::CUDA)
            GTEST_SKIP();
        const GpuBackendScope scope(GetParam());
        PipelinedLoaderConfig config;
        config.backend = GetParam();
        PipelinedImageLoader loader(config);
        auto camera = distorted_camera();
        camera.src_width = camera.dst_width = 256;
        camera.src_height = camera.dst_height = 256;
        lfs::io::LoadParams params;
        params.undistort = &camera;
        params.decode_16bit = true;
        std::vector<int32_t> samples(3 * 65536);
        for (size_t i = 0; i < samples.size(); ++i)
            samples[i] = static_cast<uint16_t>(i);
        const auto integers = Tensor::from_blob(samples.data(), {3, 256, 256}, Device::CPU, DataType::Int32).clone().to(GetParam());
        const auto expected = integers.to(DataType::Int32).to(DataType::Float32).div(65535.0f).cpu();
        const auto file = path("rgb16.png");
        CacheAccess::encode(loader, CacheAccess::key(loader, file, params), integers, 64u << 20, true);
        expect_close(loader.load_image_immediate(file, params), expected, 0.f);
        EXPECT_EQ(loader.get_stats().hot_path_hits, 1u);
    }

    TEST_P(PipelinedLoaderBackends, PreparedPngCacheSpillsRejectsDamageAndPreservesHealthyBlobOnFailure) {
        if (GetParam() == GpuBackend::CUDA)
            GTEST_SKIP();
        const GpuBackendScope scope(GetParam());
        PipelinedLoaderConfig config;
        config.min_spill_free_bytes = 0; // spill regardless of free space in the temp dir
        config.backend = GetParam();
        config.max_cache_bytes = 1;
        PipelinedImageLoader loader(config);
        auto camera = distorted_camera();
        lfs::io::LoadParams params;
        params.undistort = &camera;
        params.decode_16bit = true;
        const auto file = path("rgb16.png");
        const auto cold = loader.load_image_immediate(file, params).cpu();
        const auto key = CacheAccess::key(loader, file, params);
        const auto healthy = CacheAccess::blob(loader, key);
        ASSERT_TRUE(healthy);
        EXPECT_EQ(loader.get_stats().jpeg_cache_bytes, 0u);
        EXPECT_EQ(loader.get_stats().spill_cache_entries, 1u);
        expect_close(loader.load_image_immediate(file, params), cold, 0.f);
        const auto integers = cold.mul(65535.0f).add(0.5f).to(DataType::Int32);
        CacheAccess::encode(loader, key, integers, 1, true);
        EXPECT_EQ(*CacheAccess::blob(loader, key), *healthy);
        for (const auto invalid : {-1.f, 65536.f}) {
            const auto outside_grid = host(std::vector<float>(WIDTH * HEIGHT * 3, invalid), {3, HEIGHT, WIDTH}).to(DataType::Int32);
            CacheAccess::encode(loader, key, outside_grid, 64u << 20, true);
            EXPECT_EQ(*CacheAccess::blob(loader, key), *healthy);
        }
        auto truncated = *healthy;
        truncated.resize(truncated.size() / 2);
        std::vector<uint16_t> wider_samples(WIDTH * 2 * HEIGHT * 3, 12345);
        image_codecs::PngEncodeOptions options;
        options.compression_level = 1;
        std::vector<uint8_t> wider;
        ASSERT_TRUE(image_codecs::encode_png_memory(
            {reinterpret_cast<const uint8_t*>(wider_samples.data()), wider_samples.size() * sizeof(uint16_t)},
            WIDTH * 2, HEIGHT, 3, 16, WIDTH * 2 * 3 * sizeof(uint16_t), wider, options));
        for (const auto& damaged : {std::vector<uint8_t>{0xff}, truncated, wider}) {
            const auto hits = loader.get_stats().hot_path_hits;
            CacheAccess::put(loader, key, damaged);
            expect_close(loader.load_image_immediate(file, params), cold, 0.f);
            EXPECT_EQ(loader.get_stats().hot_path_hits, hits);
            expect_close(loader.load_image_immediate(file, params), cold, 0.f);
            EXPECT_EQ(loader.get_stats().hot_path_hits, hits + 1);
        }
    }

    TEST_P(PipelinedLoaderBackends, PreparedPngCacheAllowsExpansionNearStagingBound) {
        if (GetParam() == GpuBackend::CUDA)
            GTEST_SKIP();
        const GpuBackendScope scope(GetParam());
        // RGB UInt16 staging is just below 64 MiB. The warp may change its
        // compressibility, so test unwarped integer samples separately too.
        constexpr int width = 4096, height = 2730;
        const auto file = path("cache-incompressible.png");
        {
            std::vector<uint16_t> samples(size_t(width) * height * 3);
            uint32_t random = 0x12345678;
            for (auto& sample : samples) {
                random ^= random << 13;
                random ^= random >> 17;
                random ^= random << 5;
                sample = static_cast<uint16_t>(random);
            }
            ASSERT_TRUE(save_png(file, samples.data(), width, height, 3, 16, 0));
        }
        PipelinedLoaderConfig config;
        config.min_spill_free_bytes = 0; // spill regardless of free space in the temp dir
        config.backend = GetParam();
        config.max_cache_bytes = 1; // Exercise actual-size disk spill too.
        PipelinedImageLoader loader(config);
        auto camera = distorted_camera();
        camera.src_width = camera.dst_width = width;
        camera.src_height = camera.dst_height = height;
        camera.src_fx = camera.dst_fx = float(width);
        camera.src_fy = camera.dst_fy = float(height);
        camera.src_cx = camera.dst_cx = width * 0.5f;
        camera.src_cy = camera.dst_cy = height * 0.5f;
        for (auto& coefficient : camera.distortion)
            coefficient = 0.f;
        lfs::io::LoadParams params;
        params.undistort = &camera;
        params.decode_16bit = true;
        ASSERT_TRUE(CacheAccess::eligible(loader, params));
        {
            const auto cold = loader.load_image_immediate(file, params).cpu();
            ASSERT_EQ(loader.get_stats().spill_cache_entries, 1u);
            EXPECT_LE(loader.get_stats().spill_cache_bytes, CacheAccess::png_encoded_limit());
            const auto warm = loader.load_image_immediate(file, params).cpu();
            ASSERT_EQ(warm.shape(), cold.shape());
            EXPECT_EQ(std::memcmp(warm.ptr<float>(), cold.ptr<float>(), cold.bytes()), 0);
        }
        EXPECT_EQ(loader.get_stats().hot_path_hits, 1u);
        auto integers = Tensor::empty_pageable_host({3, size_t(height), size_t(width)}, DataType::Int32);
        uint32_t random = 0x12345678;
        for (size_t i = 0; i < integers.numel(); ++i) {
            random ^= random << 13;
            random ^= random >> 17;
            random ^= random << 5;
            integers.ptr<int32_t>()[i] = static_cast<uint16_t>(random);
        }
        const auto expected = integers.to(GetParam()).to(DataType::Float32).div(65535.0f).cpu();
        CacheAccess::encode(loader, CacheAccess::key(loader, file, params), integers, CacheAccess::png_encoded_limit(), true);
        ASSERT_GT(loader.get_stats().spill_cache_bytes, 64u << 20);
        EXPECT_LE(loader.get_stats().spill_cache_bytes, CacheAccess::png_encoded_limit());
        const auto restored = loader.load_image_immediate(file, params).cpu();
        ASSERT_EQ(restored.shape(), expected.shape());
        EXPECT_EQ(std::memcmp(restored.ptr<float>(), expected.ptr<float>(), expected.bytes()), 0);
        EXPECT_EQ(loader.get_stats().hot_path_hits, 2u);
    }

    TEST_P(PipelinedLoaderBackends, PreparedExrCacheSpillsCorruptionFallsBackAndRecovers) {
        if (GetParam() == GpuBackend::CUDA)
            GTEST_SKIP();
        const GpuBackendScope scope(GetParam());
        PipelinedLoaderConfig config;
        config.min_spill_free_bytes = 0; // spill regardless of free space in the temp dir
        config.backend = GetParam();
        config.max_cache_bytes = 1;
        PipelinedImageLoader loader(config);
        auto camera = distorted_camera();
        lfs::io::LoadParams params;
        params.undistort = &camera;
        params.decode_16bit = true;
        params.decode_float = true; // Exercise the EXR path explicitly, outside training.
        const auto file = path("rgb16.png");
        const auto cold = loader.load_image_immediate(file, params).cpu();
        EXPECT_EQ(loader.get_stats().jpeg_cache_bytes, 0u);
        EXPECT_EQ(loader.get_stats().spill_cache_entries, 1u);
        expect_close(loader.load_image_immediate(file, params), cold, 0.f);
        const auto key = CacheAccess::key(loader, file, params);
        CacheAccess::put(loader, key, {0x76, 0x2f, 0x31, 0x01, 0xff});
        const auto hits = loader.get_stats().hot_path_hits;
        expect_close(loader.load_image_immediate(file, params), cold, 0.f);
        EXPECT_EQ(loader.get_stats().hot_path_hits, hits);
        expect_close(loader.load_image_immediate(file, params), cold, 0.f);
        EXPECT_EQ(loader.get_stats().hot_path_hits, hits + 1);
        std::vector<float> pixels(WIDTH * HEIGHT * 3, 0.5f);
        image_codecs::ExrEncodeOptions options;
        options.precision = image_codecs::ExrSampleType::Float;
        options.compression = image_codecs::ExrCompression::None;
        std::vector<uint8_t> damaged;
        ASSERT_TRUE(image_codecs::encode_exr_memory(
            {WIDTH, HEIGHT, 3, WIDTH * 3 * sizeof(float), {reinterpret_cast<const uint8_t*>(pixels.data()), pixels.size() * sizeof(float)}},
            damaged, options));
        // The last uncompressed RGB sample is replaced by a quiet NaN.
        ASSERT_GE(damaged.size(), 4u);
        const std::array<uint8_t, 4> nan{0, 0, 0xc0, 0x7f};
        std::copy(nan.begin(), nan.end(), damaged.end() - 4);
        CacheAccess::put(loader, key, std::move(damaged));
        expect_close(loader.load_image_immediate(file, params), cold, 0.f);
        EXPECT_EQ(loader.get_stats().hot_path_hits, hits + 1);
        std::vector<float> wider_pixels(WIDTH * 2 * HEIGHT * 3, 0.5f);
        std::vector<uint8_t> wider;
        ASSERT_TRUE(image_codecs::encode_exr_memory(
            {WIDTH * 2, HEIGHT, 3, WIDTH * 2 * 3 * sizeof(float), {reinterpret_cast<const uint8_t*>(wider_pixels.data()), wider_pixels.size() * sizeof(float)}},
            wider, options));
        CacheAccess::put(loader, key, std::move(wider));
        expect_close(loader.load_image_immediate(file, params), cold, 0.f);
        EXPECT_EQ(loader.get_stats().hot_path_hits, hits + 1);
    }

    TEST_P(PipelinedLoaderBackends, PreparedExrCacheFailurePreservesSourceAndSuccessfulBlob) {
        if (GetParam() == GpuBackend::CUDA)
            GTEST_SKIP();
        const GpuBackendScope scope(GetParam());
        PipelinedLoaderConfig config;
        config.backend = GetParam();
        PipelinedImageLoader loader(config);
        auto camera = distorted_camera();
        lfs::io::LoadParams params;
        params.undistort = &camera;
        params.decode_float = true;
        const auto file = path("rgbfloat.exr");
        const auto image = loader.load_image_immediate(file, params);
        const auto expected = image.cpu();
        const auto key = CacheAccess::key(loader, file, params);
        const auto before = CacheAccess::blob(loader, key);
        ASSERT_TRUE(before);
        CacheAccess::encode(loader, key, image, 1, true);
        expect_close(image, expected, 0.f);
        ASSERT_TRUE(CacheAccess::blob(loader, key));
        EXPECT_EQ(*CacheAccess::blob(loader, key), *before);
        expect_close(loader.load_image_immediate(file, params), expected, 0.f);
        camera.distortion[0] += 0.02f;
        EXPECT_NE(CacheAccess::key(loader, file, params), key);
        (void)loader.load_image_immediate(file, params);
        EXPECT_EQ(loader.get_stats().jpeg_cache_entries, 2u);
    }

    TEST_P(PipelinedLoaderBackends, PreparedCacheShutdownDoesNotStartAnotherEncodeOrWarn) {
        if (GetParam() == GpuBackend::CUDA)
            GTEST_SKIP();
        const GpuBackendScope scope(GetParam());
        PipelinedLoaderConfig config;
        config.backend = GetParam();
        PipelinedImageLoader loader(config);
        const auto image = host(std::vector<float>(WIDTH * HEIGHT * 3, 0.25f), {3, HEIGHT, WIDTH});
        const auto before = image.to_vector();
        loader.shutdown();
        std::atomic<unsigned> warnings{0};
        const DebugLogHandler logs([&](LogLevel level, const SourceSite&, std::string_view message) {
            if (level == LogLevel::Warn && message.find("Optional prepared image cache") != std::string_view::npos)
                ++warnings;
        });
        CacheAccess::encode(loader, "stopped-exr", image, 1, true);
        CacheAccess::encode(loader, "stopped-jpeg", image, 1);
        CacheAccess::encode(loader, "stopped-png", image.to(DataType::Int32), 1, true);
        EXPECT_EQ(loader.get_stats().jpeg_cache_entries, 0u);
        EXPECT_EQ(loader.get_stats().spill_cache_entries, 0u);
        EXPECT_EQ(image.to_vector(), before);
        EXPECT_EQ(warnings.load(), 0u);
    }

    Tensor decoded_prepared_blob(PipelinedImageLoader& loader, const std::filesystem::path& file,
                                 const lfs::io::LoadParams& params) {
        const auto bytes = CacheAccess::blob(loader, CacheAccess::key(loader, file, params));
        EXPECT_TRUE(bytes);
        if (!bytes)
            return {};
        lfs::core::image_codecs::Image decoded;
        std::string error;
        EXPECT_TRUE(lfs::core::image_codecs::decode_memory(bytes->data(), bytes->size(), decoded, error)) << error;
        if (decoded.sample_type == image_codecs::SampleType::UInt16) {
            auto planar = Tensor::empty_pageable_host({3, size_t(decoded.height), size_t(decoded.width)}, DataType::Int32);
            const size_t plane = size_t(decoded.width) * decoded.height;
            for (size_t i = 0; i < plane; ++i)
                for (size_t ch = 0; ch < 3; ++ch) {
                    uint16_t value;
                    std::memcpy(&value, decoded.data.data() + (i * 3 + ch) * sizeof(uint16_t), sizeof(value));
                    planar.ptr<int32_t>()[ch * plane + i] = value;
                }
            return planar.to(CacheAccess::backend(loader)).to(DataType::Float32).div(65535.0f).cpu();
        }
        const bool floating = decoded.sample_type == lfs::core::image_codecs::SampleType::Float32;
        EXPECT_EQ(decoded.channels, floating ? 4 : 3);
        const size_t plane = size_t(decoded.width) * decoded.height;
        std::vector<float> values(plane * 3);
        for (size_t i = 0; i < plane; ++i)
            for (size_t ch = 0; ch < 3; ++ch)
                if (floating)
                    std::memcpy(&values[ch * plane + i], decoded.data.data() + (i * 4 + ch) * sizeof(float), sizeof(float));
                else
                    values[ch * plane + i] = float(decoded.data[i * 3 + ch]) * UINT8_SCALE;
        return host(values, {3, size_t(decoded.height), size_t(decoded.width)});
    }

    TEST_P(PipelinedLoaderBackends, PreparedJpegCacheDecodesWarmPixelsAndInvalidatesParameters) {
        if (GetParam() == GpuBackend::CUDA)
            GTEST_SKIP();
        const GpuBackendScope scope(GetParam());
        PipelinedLoaderConfig config;
        config.backend = GetParam();
        PipelinedImageLoader loader(config);
        auto camera = distorted_camera();
        lfs::io::LoadParams params;
        params.undistort = &camera;
        auto cold = loader.load_image_immediate(path("rgb8.png"), params);
        ASSERT_EQ(loader.get_stats().jpeg_cache_entries, 1u);
        auto expected = decoded_prepared_blob(loader, path("rgb8.png"), params);
        cold.fill_(0.f);
        expect_close(loader.load_image_immediate(path("rgb8.png"), params), expected, 0.f);
        EXPECT_EQ(loader.get_stats().hot_path_hits, 1u);
        EXPECT_EQ(loader.get_stats().cpu_decode_calls, 2u); // Decode JPEG on every request.
        const auto old_key = CacheAccess::key(loader, path("rgb8.png"), params);
        camera.distortion[0] += 0.02f;
        EXPECT_NE(CacheAccess::key(loader, path("rgb8.png"), params), old_key);
        (void)loader.load_image_immediate(path("rgb8.png"), params);
        EXPECT_EQ(loader.get_stats().jpeg_cache_entries, 2u);
        params.max_width = 7;
        EXPECT_NE(loader.load_image_immediate(path("rgb8.png"), params).shape(), cold.shape());
        params.output_uint8 = true;
        EXPECT_EQ(loader.load_image_immediate(path("rgb8.png"), params).dtype(), DataType::UInt8);
        EXPECT_EQ(loader.load_image_immediate(path("rgb8.png"), params).dtype(), DataType::UInt8);
    }

    TEST_P(PipelinedLoaderBackends, PreparedJpegCacheMatchesFullChromaAndByteConversionPolicy) {
        if (GetParam() == GpuBackend::CUDA)
            GTEST_SKIP();
        const GpuBackendScope scope(GetParam());
        PipelinedLoaderConfig config;
        config.backend = GetParam();
        config.cache_jpeg_quality = 83;
        PipelinedImageLoader loader(config);
        auto camera = distorted_camera();
        lfs::io::LoadParams params;
        params.undistort = &camera;
        const auto image = loader.load_image_immediate(path("rgb8.png"), params).cpu();
        const auto values = image.to_vector();
        const size_t width = image.shape()[2], height = image.shape()[1], plane = width * height;
        std::vector<uint8_t> rgb(values.size());
        for (size_t i = 0; i < plane; ++i)
            for (size_t ch = 0; ch < 3; ++ch)
                rgb[i * 3 + ch] = static_cast<uint8_t>(std::clamp(values[ch * plane + i] * 255.0f, 0.f, 255.f));
        lfs::core::image_codecs::JpegEncodeOptions options;
        options.quality = config.cache_jpeg_quality;
        options.full_chroma = true;
        std::vector<uint8_t> expected;
        ASSERT_TRUE(lfs::core::image_codecs::encode_jpeg_memory(rgb, int(width), int(height), 3, width * 3, expected, options));
        const auto actual = CacheAccess::blob(loader, CacheAccess::key(loader, path("rgb8.png"), params));
        ASSERT_TRUE(actual);
        EXPECT_EQ(*actual, expected);
        EXPECT_EQ(actual->capacity(), actual->size());
    }

    TEST_P(PipelinedLoaderBackends, PreparedJpegCacheUsesSharedRunScopedKeyAndHonorsBudget) {
        if (GetParam() == GpuBackend::CUDA)
            GTEST_SKIP();
        const GpuBackendScope scope(GetParam());
        auto camera = distorted_camera();
        lfs::io::LoadParams params;
        params.undistort = &camera;
        PipelinedLoaderConfig config;
        config.min_spill_free_bytes = 0; // spill regardless of free space in the temp dir
        config.backend = GetParam();
        config.max_cache_bytes = 1024;
        PipelinedImageLoader loader(config);
        const auto file = path("rgb8.png");
        (void)loader.load_image_immediate(file, params);
        const auto key = CacheAccess::key(loader, file, params);
        EXPECT_TRUE(key.starts_with(lfs::core::path_to_utf8(file) + ":udr4_"));
        EXPECT_EQ(key.find(":jpeg:"), std::string::npos);
        const auto missing = path("missing-cache-key-source.png");
        EXPECT_EQ(CacheAccess::key(loader, missing, params),
                  lfs::core::path_to_utf8(missing) + key.substr(lfs::core::path_to_utf8(file).size()));
        {
            // Quality is fixed per loader; separate runs own separate blob caches.
            auto other_config = config;
            other_config.cache_jpeg_quality = 83;
            PipelinedImageLoader other(other_config);
            EXPECT_EQ(CacheAccess::key(other, file, params), key);
            EXPECT_EQ(other.get_stats().jpeg_cache_entries, 0u);
        }
        expect_close(loader.load_image_immediate(file, params), decoded_prepared_blob(loader, file, params), 0.f);
        EXPECT_LE(loader.get_stats().jpeg_cache_bytes, config.max_cache_bytes);
        EXPECT_GT(loader.release_host_cache(std::numeric_limits<size_t>::max()), 0u);
        EXPECT_EQ(loader.get_stats().jpeg_cache_bytes, 0u);
        const auto hits = loader.get_stats().hot_path_hits;
        (void)loader.load_image_immediate(file, params);
        EXPECT_EQ(loader.get_stats().hot_path_hits, hits + 1); // Existing spill restores the same JPEG bytes.
    }

    TEST_P(PipelinedLoaderBackends, PreparedJpegCacheSpillsOversizedEntriesAndCleansItsRun) {
        if (GetParam() == GpuBackend::CUDA)
            GTEST_SKIP();
        const GpuBackendScope scope(GetParam());
        auto camera = distorted_camera();
        lfs::io::LoadParams params;
        params.undistort = &camera;
        std::filesystem::path spill;
        {
            PipelinedLoaderConfig config;
            config.min_spill_free_bytes = 0; // spill regardless of free space in the temp dir
            config.backend = GetParam();
            config.max_cache_bytes = 1;
            PipelinedImageLoader loader(config);
            spill = loader.run_spill_directory();
            ASSERT_FALSE(spill.empty());
            (void)loader.load_image_immediate(path("rgb8.png"), params);
            const auto expected = decoded_prepared_blob(loader, path("rgb8.png"), params);
            EXPECT_EQ(loader.get_stats().jpeg_cache_bytes, 0u);
            EXPECT_EQ(loader.get_stats().spill_cache_entries, 1u);
            expect_close(loader.load_image_immediate(path("rgb8.png"), params), expected, 0.f);
            EXPECT_EQ(loader.get_stats().hot_path_hits, 1u);
        }
        EXPECT_FALSE(std::filesystem::exists(spill));
    }

    TEST_P(PipelinedLoaderBackends, PreparedCachesBypassEvaluationAndMaskedFill) {
        if (GetParam() == GpuBackend::CUDA)
            GTEST_SKIP();
        const GpuBackendScope scope(GetParam());
        PipelinedLoaderConfig config;
        config.backend = GetParam();
        PipelinedImageLoader loader(config);
        auto camera = distorted_camera();
        lfs::io::LoadParams params;
        params.undistort = &camera;
        params.skip_blob_cache = true;
        const auto expected = loader.load_image_immediate(path("rgb8.png"), params).cpu();
        expect_close(loader.load_image_immediate(path("rgb8.png"), params), expected, 0.f);
        params.skip_blob_cache = false;
        const lfs::io::MaskedImageFill fill{};
        params.masked_fill = &fill;
        (void)loader.load_image_immediate(path("rgb8.png"), params);
        EXPECT_EQ(loader.get_stats().jpeg_cache_entries, 0u);
        EXPECT_EQ(loader.get_stats().hot_path_hits, 0u);
    }

    TEST_P(PipelinedLoaderBackends, PreparedCachesConcurrentWarmConsumersOwnIndependentImages) {
        if (GetParam() == GpuBackend::CUDA)
            GTEST_SKIP();
        const GpuBackendScope scope(GetParam());
        for (bool high_precision : {false, true}) {
            SCOPED_TRACE(high_precision);
            const char* source_name = high_precision ? "rgb16.png" : "rgb8.png";
            PipelinedLoaderConfig config;
            config.backend = GetParam();
            PipelinedImageLoader loader(config);
            auto camera = distorted_camera();
            lfs::io::LoadParams params;
            params.undistort = &camera;
            params.decode_16bit = high_precision;
            (void)loader.load_image_immediate(path(source_name), params);
            const auto expected = decoded_prepared_blob(loader, path(source_name), params);
            const auto backend = GetParam();
            std::vector<std::future<Tensor>> consumers;
            for (int i = 0; i < 8; ++i)
                consumers.push_back(std::async(std::launch::async, [&, backend] {
                    const GpuBackendScope worker(backend);
                    auto image = loader.load_image_immediate(path(source_name), params);
                    auto copy = image.cpu();
                    image.fill_(0.f);
                    return copy;
                }));
            for (auto& consumer : consumers)
                expect_close(consumer.get(), expected, 0.f);
            EXPECT_EQ(loader.get_stats().hot_path_hits, 8u);
            EXPECT_EQ(loader.get_stats().jpeg_cache_entries, 1u);
            expect_close(loader.load_image_immediate(path(source_name), params), expected, 0.f);
        }
    }

    TEST_P(PipelinedLoaderBackends, PreparedCachesConcurrentMissesPublishOnlyCompleteBlobs) {
        if (GetParam() == GpuBackend::CUDA)
            GTEST_SKIP();
        const GpuBackendScope scope(GetParam());
        for (bool high_precision : {false, true}) {
            SCOPED_TRACE(high_precision);
            const char* source_name = high_precision ? "rgb16.png" : "rgb8.png";
            PipelinedLoaderConfig config;
            config.backend = GetParam();
            PipelinedImageLoader loader(config);
            auto camera = distorted_camera();
            lfs::io::LoadParams params;
            params.undistort = &camera;
            params.decode_16bit = high_precision;
            params.skip_blob_cache = true;
            const auto cold = loader.load_image_immediate(path(source_name), params).cpu().to_vector();
            params.skip_blob_cache = false;
            std::vector<std::future<Tensor>> consumers;
            const auto backend = GetParam();
            for (int i = 0; i < 8; ++i)
                consumers.push_back(std::async(std::launch::async, [&, backend] {
                    const GpuBackendScope worker(backend);
                    return loader.load_image_immediate(path(source_name), params).cpu();
                }));
            std::vector<Tensor> loaded;
            for (auto& consumer : consumers)
                loaded.push_back(consumer.get());
            const auto warm = decoded_prepared_blob(loader, path(source_name), params).to_vector();
            for (const auto& image : loaded) {
                const auto values = image.to_vector();
                EXPECT_TRUE(values == cold || values == warm);
            }
            EXPECT_EQ(loader.get_stats().jpeg_cache_entries, 1u);
            EXPECT_EQ(loader.get_stats().cpu_decode_calls, 9u);
        }
    }

    TEST_P(PipelinedLoaderBackends, PreparedCachesPrefetchSharesImmediateBlobWithoutAnotherWarp) {
        if (GetParam() == GpuBackend::CUDA)
            GTEST_SKIP();
        const GpuBackendScope scope(GetParam());
        for (bool high_precision : {false, true}) {
            SCOPED_TRACE(high_precision);
            const char* source_name = high_precision ? "rgb16.png" : "rgb8.png";
            PipelinedLoaderConfig config;
            config.backend = GetParam();
            PipelinedImageLoader loader(config);
            auto camera = distorted_camera();
            lfs::io::LoadParams params;
            params.undistort = &camera;
            params.decode_16bit = high_precision;
            (void)loader.load_image_immediate(path(source_name), params);
            const auto expected = decoded_prepared_blob(loader, path(source_name), params);
            std::vector<ImageRequest> requests;
            for (size_t i = 0; i < 8; ++i) {
                auto r = request(source_name);
                r.sequence_id = i;
                r.params = params;
                r.undistort = &camera;
                requests.push_back(r);
            }
            loader.prefetch(requests);
            for (size_t i = 0; i < requests.size(); ++i) {
                auto completion = loader.try_get_completion_for(std::chrono::seconds(20));
                ASSERT_TRUE(completion && completion->outcome);
                expect_close(completion->outcome->tensor, expected, 0.f);
            }
            EXPECT_EQ(loader.get_stats().hot_path_hits, 8u);
            EXPECT_EQ(loader.get_stats().jpeg_cache_entries, 1u);
        }
    }

    TEST_P(PipelinedLoaderBackends, PreparedJpegCacheFailurePreservesImageAndRecovers) {
        if (GetParam() == GpuBackend::CUDA)
            GTEST_SKIP();
        const GpuBackendScope scope(GetParam());
        PipelinedLoaderConfig config;
        config.backend = GetParam();
        PipelinedImageLoader loader(config);
        auto camera = distorted_camera();
        lfs::io::LoadParams params;
        params.undistort = &camera;
        params.skip_blob_cache = true;
        auto image = loader.load_image_immediate(path("rgb8.png"), params);
        const auto expected = image.cpu();
        params.skip_blob_cache = false;
        const auto key = CacheAccess::key(loader, path("rgb8.png"), params);
        std::atomic<size_t> warnings{0}, debug_failures{0};
        {
            const DebugLogHandler handler([&](LogLevel level, const SourceSite&, std::string_view message) {
                if (message.find("Optional prepared image cache") == std::string_view::npos)
                    return;
                if (level == LogLevel::Warn)
                    ++warnings;
                if (level == LogLevel::Debug)
                    ++debug_failures;
            });
            CacheAccess::encode(loader, key, image, 1); // Actual encoder ResourceExhausted.
            CacheAccess::encode(loader, key, image, 1);
        }
        EXPECT_EQ(warnings.load(), 1u);
        EXPECT_EQ(debug_failures.load(), 2u);
        EXPECT_EQ(loader.get_stats().jpeg_cache_entries, 0u);
        expect_close(image, expected, 0.f);
        (void)loader.load_image_immediate(path("rgb8.png"), params);
        EXPECT_EQ(loader.get_stats().jpeg_cache_entries, 1u);
        expect_close(loader.load_image_immediate(path("rgb8.png"), params), decoded_prepared_blob(loader, path("rgb8.png"), params), 0.f);
    }

    TEST_P(PipelinedLoaderBackends, PreparedJpegCacheCorruptBlobFallsBackToOriginal) {
        if (GetParam() == GpuBackend::CUDA)
            GTEST_SKIP();
        const GpuBackendScope scope(GetParam());
        PipelinedLoaderConfig config;
        config.backend = GetParam();
        PipelinedImageLoader loader(config);
        auto camera = distorted_camera();
        lfs::io::LoadParams params;
        params.undistort = &camera;
        params.skip_blob_cache = true;
        const auto expected = loader.load_image_immediate(path("rgb8.png"), params).cpu();
        params.skip_blob_cache = false;
        const auto key = CacheAccess::key(loader, path("rgb8.png"), params);
        std::atomic<size_t> warnings{0}, debug_failures{0};
        {
            const DebugLogHandler handler([&](LogLevel level, const SourceSite&, std::string_view message) {
                if (message.find("Optional prepared image cache") == std::string_view::npos)
                    return;
                if (level == LogLevel::Warn)
                    ++warnings;
                if (level == LogLevel::Debug)
                    ++debug_failures;
            });
            for (int attempt = 0; attempt < 2; ++attempt) {
                CacheAccess::put(loader, key, {0xff, 0xd8, 0xff, 0xd9});
                expect_close(loader.load_image_immediate(path("rgb8.png"), params), expected, 0.f);
            }
        }
        EXPECT_EQ(warnings.load(), 1u);
        EXPECT_EQ(debug_failures.load(), 2u);
        EXPECT_EQ(loader.get_stats().hot_path_hits, 0u);
        expect_close(loader.load_image_immediate(path("rgb8.png"), params), decoded_prepared_blob(loader, path("rgb8.png"), params), 0.f);
    }

    INSTANTIATE_TEST_SUITE_P(Backends, PipelinedLoaderBackends,
                             ::testing::Values(GpuBackend::CUDA, GpuBackend::Vulkan, GpuBackend::Metal),
                             [](const auto& info) { return std::string(gpu_backend_name(info.param)); });

    TEST(PipelinedLoaderVulkan, PortableLoaderDecodesEveryRequestAndCachesOnlyEncodedBytes) {
        if (!gpu_backend_available(GpuBackend::Vulkan))
            GTEST_SKIP();
        const GpuBackendScope backend(GpuBackend::Vulkan);
        const auto directory = fixture_directory() / "rgb_cache";
        std::filesystem::create_directories(directory);
        const auto first = directory / "first.jpg", second = directory / "second.jpg";
        const auto pixels = samples<uint8_t>(3, 255, 17);
        auto encoded = pixels;
        const auto host_pixels = lfs::core::Tensor::from_blob(encoded.data(), {size_t(HEIGHT), size_t(WIDTH), size_t(3)}, lfs::core::Device::CPU,
                                                              DataType::UInt8);
        lfs::core::save_image_u8(first, host_pixels, 95);
        lfs::core::save_image_u8(second, host_pixels, 95);
        {
            PipelinedLoaderConfig config;
            config.backend = GpuBackend::Vulkan;
            PipelinedImageLoader loader(config);
            lfs::io::LoadParams params;
            auto initial = loader.load_image_immediate(first, params);
            const auto expected = initial.cpu();
            initial.fill_(0.f);
            // No decoded image is kept: the second request decodes again and
            // does not see the first output's writes.
            expect_close(loader.load_image_immediate(first, params), expected, 0.f);
            EXPECT_EQ(loader.get_stats().cpu_decode_calls, 2u);
            params.output_uint8 = true;
            EXPECT_EQ(loader.load_image_immediate(first, params).dtype(), DataType::UInt8);
            params.output_uint8 = false;
            params.resize_factor = 2;
            EXPECT_NE(loader.load_image_immediate(first, params).shape(), expected.shape());
            params.resize_factor = -1;

            // Prefetched JPEGs decode from the encoded-bytes cache, like CUDA.
            std::vector<ImageRequest> requests;
            for (size_t i = 0; i < 6; ++i) {
                ImageRequest request;
                request.sequence_id = i;
                request.path = i % 2 ? second : first;
                requests.push_back(request);
            }
            loader.prefetch(requests);
            for (size_t i = 0; i < requests.size(); ++i) {
                auto completion = loader.try_get_completion_for(std::chrono::seconds(20));
                ASSERT_TRUE(completion);
                ASSERT_TRUE(completion->outcome);
                expect_close(completion->outcome->tensor, expected, 0.f);
            }
            EXPECT_EQ(loader.get_stats().cpu_decode_calls, 10u);
            EXPECT_GT(loader.release_host_cache(std::numeric_limits<size_t>::max()), 0u);
        }
        std::filesystem::remove_all(directory);
    }

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
                bool failed_and_recovered = false;
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
                    auto completion = loader.try_get_completion_for(std::chrono::seconds(20));
                    delivered = completion && completion->outcome && completion->outcome->mask &&
                                completion->outcome->depth && completion->outcome->normal;
                    (void)loader.load_image_immediate(request.path, {});

                    // A JPEG 2000 signature is unsupported by the host codecs.
                    const auto unsupported = directory / "unsupported.jp2";
                    std::ofstream file(unsupported, std::ios::binary);
                    file.write("\0\0\0\x0c"
                               "jP  \r\n\x87\n",
                               12);
                    file.close();
                    ImageRequest bad;
                    bad.sequence_id = 1;
                    bad.path = unsupported;
                    loader.prefetch({bad});
                    auto failure = loader.try_get_completion_for(std::chrono::seconds(20));
                    const bool typed_failure = failure && !failure->outcome &&
                                               failure->outcome.error().code() == lfs::ErrorCode::DataLoss &&
                                               std::string(failure->outcome.error().user_message()).find("Unsupported image format") != std::string::npos;
                    request.sequence_id = 2;
                    loader.prefetch({request});
                    auto recovered = loader.try_get_completion_for(std::chrono::seconds(20));
                    failed_and_recovered = typed_failure && recovered && recovered->outcome;
                }
                Logger::get().remove_log_handler(handler);
                std::filesystem::remove_all(directory);
                // Retire the device before the immediate process exit. Windows
                // drivers may fail during _Exit with live Vulkan worker threads.
                const bool shutdown_ok = bool(shutdown_gpu_backend(GpuBackend::Vulkan));
                std::_Exit(delivered && failed_and_recovered && loader_logged && !codec_touched && shutdown_ok ? 0 : 1);
            },
            ::testing::ExitedWithCode(0), "");
        GTEST_FLAG_SET(death_test_style, style);
    }

} // namespace
