/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "core/camera_views.hpp"
#include "core/image_io.hpp"
#include "io/cache_image_loader.hpp"
#include "io/nvcodec_image_loader.hpp"
#include "io/pipelined_image_loader.hpp"
#include "licht_test_support.hpp"
#include "training/dataset.hpp"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <string>

namespace {
    using namespace lfs::core;
    constexpr size_t width = 129, height = 97;
    class MaskRgbCache : public ::testing::Test {
    protected:
        lfs::test::licht::TemporaryDirectory temp{"mask-rgb-cache"};
        void SetUp() override {
            int count = 0;
            if (cudaGetDeviceCount(&count) != cudaSuccess || !count)
                GTEST_SKIP() << "CUDA unavailable";
        }
        void check_memory_precedence();
        void check_memory_revision_cache();
        auto config() {
            lfs::io::PipelinedLoaderConfig c;
            c.prefetch_count = 1;
            c.output_queue_size = 2;
            c.jpeg_batch_size = 1;
            c.decoder_pool_size = 1;
            c.io_threads = 1;
            c.cold_process_threads = 1;
            c.max_cache_bytes = 8 * 1024 * 1024;
            return c;
        }
    };
} // namespace

TEST_F(MaskRgbCache, ImmediateLoaderUndistortsDecodedPixelsOnTheDestinationGrid) {
    constexpr int w = 128, h = 64;
    std::vector<uint8_t> pixels(w * h * 3);
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x)
            for (int c = 0; c < 3; ++c)
                pixels[(y * w + x) * 3 + c] = (3 * x + 5 * y + c * 71) % 256;

    const auto bytes = Tensor::from_blob(pixels.data(), {h, w, 3}, Device::CPU, DataType::UInt8);
    const auto lens = compute_undistort_params(84, 81, w / 2, h / 2, w, h,
                                               Tensor::from_vector({-.18f, .04f}, {2}, Device::CPU), {},
                                               CameraModelType::PINHOLE);
    const Camera panorama(Tensor::eye(3, Device::CPU), Tensor::zeros({3}, Device::CPU), 0, 0, 0, 0,
                          {}, {}, CameraModelType::EQUIRECTANGULAR, "source.png", temp.path / "source.png", {}, w, h, 0);
    const auto faces = plan_camera_views(panorama, 0);
    ASSERT_EQ(faces.size(), kPanoramaViewCount);
    auto& cache = lfs::io::CacheLoader::getInstance(true);
    cache.reset_cache();
    cache.update_cache_params(true, 1, lfs::io::DEFAULT_MIN_FREE_GB, lfs::io::DEFAULT_MIN_FREE_MEMORY_RATIO,
                              false, lfs::io::DEFAULT_PRINT_STATUS_FREQ);

    // The loader runs its kernels on the stream the caller passes; the result must be complete wherever it is read.
    cudaStream_t side_stream = nullptr;
    ASSERT_EQ(cudaStreamCreateWithFlags(&side_stream, cudaStreamNonBlocking), cudaSuccess);
    const auto check = [&](const std::filesystem::path& path, const Tensor& decoded) {
        SCOPED_TRACE(path.extension().string());
        // Two faces share all calibration and size fields except their rotation.
        for (const auto* mapping : {&lens, &faces[0].params, &faces[1].params}) {
            for (const auto& [resize, cap] : {std::pair{1, 0}, std::pair{2, 0}, std::pair{2, 13}}) {
                const auto grid = prepare_undistort_params(*mapping, w, h, resize, cap);
                const auto expected = undistort_image(decoded, grid, nullptr).clamp(0.f, 1.f);
                for (const auto [u8, stream] : {std::pair{false, cudaStream_t{}}, std::pair{true, cudaStream_t{}},
                                                std::pair{false, side_stream}, std::pair{true, side_stream}}) {
                    SCOPED_TRACE(::testing::Message() << "view=" << is_view(*mapping) << " resize=" << resize
                                                      << " max_width=" << cap << " uint8=" << u8
                                                      << " side stream=" << (stream != nullptr));
                    cache.reset_cache();
                    cache.update_cache_params(true, 1, lfs::io::DEFAULT_MIN_FREE_GB, lfs::io::DEFAULT_MIN_FREE_MEMORY_RATIO,
                                              false, lfs::io::DEFAULT_PRINT_STATUS_FREQ);
                    const lfs::io::LoadParams params{
                        .resize_factor = resize,
                        .max_width = cap,
                        .cuda_stream = stream,
                        .undistort = mapping,
                        .output_uint8 = u8};
                    for (int visit = 0; visit < 2; ++visit) {
                        const auto image = cache.load_cached_image(path, params);
                        EXPECT_EQ(image.dtype(), u8 ? DataType::UInt8 : DataType::Float32);
                        EXPECT_EQ(image.shape(), expected.shape());
                        if (image.shape() != expected.shape())
                            continue;
                        auto actual = image.to(Device::CUDA).to(DataType::Float32);
                        if (u8)
                            actual = actual / 255.f;
                        const float tolerance = u8 ? 1.f / 255.f : 1.f / 65535.f;
                        EXPECT_LE((actual - expected).abs().max().item(), tolerance) << "visit=" << visit;
                    }
                }
            }
        }
    };
    const auto png = temp.path / "source.png";
    ASSERT_TRUE(save_png(png, pixels.data(), w, h, 3, 8, 0));
    check(png, bytes.to(Device::CUDA).to(DataType::Float32).permute({2, 0, 1}).contiguous() / 255.f);
    if (lfs::io::NvCodecImageLoader::is_available()) {
        const auto jpeg = temp.path / "source.jpg";
        save_image_u8(jpeg, bytes, 100);
        lfs::io::NvCodecImageLoader decoder({.enable_fallback = false, .decoder_pool_size = 1});
        check(jpeg, decoder.load_image_gpu(jpeg));
    }
    cache.reset_cache();
    ASSERT_EQ(cudaStreamDestroy(side_stream), cudaSuccess);
}

// An undistorted image decodes at its file's own depth; the result stays bit for bit the one of a 16-bit decode.
TEST_F(MaskRgbCache, UndistortedDecodeMatchesTheSixteenBitDecode) {
    constexpr int w = 128, h = 64;
    std::vector<uint8_t> bytes(w * h * 3);
    std::vector<uint16_t> words(w * h * 3);
    for (size_t i = 0; i < bytes.size(); ++i) {
        bytes[i] = static_cast<uint8_t>((i * 37 + i / 97) % 256);
        words[i] = static_cast<uint16_t>((i * 7919 + i / 13) % 65536);
    }
    const auto eight = temp.path / "eight.png", sixteen = temp.path / "sixteen.png";
    ASSERT_TRUE(save_png(eight, bytes.data(), w, h, 3, 8, 0));
    ASSERT_TRUE(save_png(sixteen, words.data(), w, h, 3, 16, 0));
    const auto lens = compute_undistort_params(84, 81, w / 2, h / 2, w, h,
                                               Tensor::from_vector({-.18f, .04f}, {2}, Device::CPU), {},
                                               CameraModelType::PINHOLE);
    for (const auto& path : {eight, sixteen}) {
        for (const bool u8 : {false, true}) {
            SCOPED_TRACE(::testing::Message() << path.filename().string() << " uint8=" << u8);
            const lfs::io::LoadParams params{.resize_factor = 2, .undistort = &lens, .output_uint8 = u8};
            const auto own_depth = lfs::io::load_rgb_image_cpu_decoded(path, params, false).cpu();
            const auto widened = lfs::io::load_rgb_image_cpu_decoded(path, params, true).cpu();
            ASSERT_EQ(own_depth.shape(), widened.shape());
            ASSERT_EQ(own_depth.dtype(), widened.dtype());
            EXPECT_EQ(std::memcmp(own_depth.data_ptr(), widened.data_ptr(), own_depth.bytes()), 0);
        }
    }
}

TEST_F(MaskRgbCache, IgnoredSourceColorsCannotChangeColdHotOrSpilledTargets) {
    check_memory_precedence();
    check_memory_revision_cache();
    for (const bool alpha : {false, true}) {
        for (const bool invert : {false, true}) {
            const size_t channels = alpha ? 4 : 3;
            std::vector<uint8_t> mask(width * height), original(width * height * channels), changed(original.size());
            for (size_t y = 0; y < height; ++y)
                for (size_t x = 0; x < width; ++x) {
                    const size_t i = y * width + x;
                    const bool keep = !(x >= 43 && x < 89 && y >= 21 && y < 76);
                    mask[i] = keep != invert ? 255 : 0;
                    for (size_t c = 0; c < channels; ++c) {
                        original[i * channels + c] = c == 3 ? mask[i] : static_cast<uint8_t>((x * 7 + y * 3 + c * 61) % 256);
                        changed[i * channels + c] = c == 3 || keep ? original[i * channels + c] : static_cast<uint8_t>((x * 41 + y * 79 + c * 113) % 256);
                    }
                }
            ASSERT_TRUE(save_png(temp.path / "mask.png", mask.data(), width, height, 1, 8, 0));
            ASSERT_TRUE(save_png(temp.path / "original.png", original.data(), width, height, channels, 8, 0));
            ASSERT_TRUE(save_png(temp.path / "changed.png", changed.data(), width, height, channels, 8, 0));
            for (const int resize : {1, 2, 4})
                for (const bool warped : {false, true}) {
                    SCOPED_TRACE(alpha);
                    SCOPED_TRACE(invert);
                    SCOPED_TRACE(resize);
                    SCOPED_TRACE(warped);
                    Camera cam(Tensor::eye(3, Device::CPU), Tensor::zeros({3}, Device::CPU), 110.f, 110.f,
                               width / 2.f, height / 2.f, Tensor::from_vector({-.1f, .02f}, {2}, Device::CPU), Tensor{},
                               CameraModelType::PINHOLE, "original.png", temp.path / "original.png", temp.path / "mask.png", width, height, 0);
                    if (warped)
                        cam.prepare_undistortion();
                    std::vector<std::vector<std::vector<float>>> samples;
                    std::vector<std::vector<float>> masks;
                    for (const char* image : {"original.png", "original.png", "changed.png"}) {
                        lfs::io::PipelinedImageLoader loader(config());
                        lfs::io::ImageRequest req;
                        req.path = temp.path / image;
                        req.params.resize_factor = resize;
                        if (warped)
                            req.params.undistort = req.undistort = &cam.undistort_params();
                        req.extract_alpha_as_mask = alpha;
                        if (!alpha)
                            req.mask_path = temp.path / "mask.png";
                        req.mask_params = req.alpha_mask_params = {.invert = invert, .threshold = .5f};
                        std::vector<std::vector<float>> visits;
                        for (size_t visit = 0; visit < 3; ++visit) {
                            if (visit == 2)
                                ASSERT_GT(loader.release_host_cache(loader.get_stats().jpeg_cache_bytes), 0u);
                            req.sequence_id = visit;
                            loader.prefetch({req});
                            auto ready = loader.get();
                            ASSERT_TRUE(ready.error.empty()) << ready.error;
                            ASSERT_TRUE(ready.mask);
                            visits.push_back(ready.tensor.cpu().to_vector());
                            if (samples.empty())
                                masks.push_back(ready.mask->cpu().to_vector());
                        }
                        samples.push_back(std::move(visits));
                    }
                    for (size_t visit = 0; visit < 3; ++visit) {
                        EXPECT_EQ(samples[0][visit], samples[1][visit]) << "same-input repeat; tolerance = 0";
                        size_t wrong = 0;
                        float maximum = 0;
                        for (size_t i = 0; i < samples[0][visit].size(); ++i)
                            if (masks[visit][i % masks[visit].size()] > 0) {
                                const float delta = std::abs(samples[0][visit][i] - samples[2][visit][i]);
                                wrong += delta != 0;
                                maximum = std::max(maximum, delta);
                            }
                        EXPECT_EQ(wrong, 0u) << "visit=" << visit << "; max delta=" << maximum;
                    }
                }
        }
    }
}

namespace {
    void MaskRgbCache::check_memory_precedence() {
        const auto dir = temp.path / "memory-precedence";
        std::filesystem::create_directories(dir);
        constexpr int factor = 2;
        std::vector<uint8_t> mask(width * height, 255), sidecar(mask.size(), 255);
        std::vector<uint8_t> original(width * height * 3), changed(original.size());
        for (size_t y = 21; y < 76; ++y)
            for (size_t x = 43; x < 89; ++x)
                mask[y * width + x] = 0;
        for (size_t y = 0; y < height; ++y)
            for (size_t x = 0; x < width; ++x) {
                const size_t pixel = y * width + x;
                const bool keep = mask[pixel] != 0;
                for (size_t c = 0; c < 3; ++c) {
                    const size_t i = pixel * 3 + c;
                    original[i] = static_cast<uint8_t>((x * 11 + y * 7 + c * 53) % 256);
                    changed[i] = keep ? original[i] : static_cast<uint8_t>((x * 67 + y * 37 + c * 97) % 256);
                }
            }
        ASSERT_TRUE(save_png(dir / "sidecar.png", sidecar.data(), width, height, 1, 8, 0));
        ASSERT_TRUE(save_png(dir / "original.png", original.data(), width, height, 3, 8, 0));
        ASSERT_TRUE(save_png(dir / "changed.png", changed.data(), width, height, 3, 8, 0));
        std::vector<std::vector<float>> images, masks;
        for (const char* name : {"original.png", "changed.png"}) {
            const auto image = dir / name;
            auto camera = std::make_shared<Camera>(Tensor::eye(3, Device::CPU), Tensor::zeros({3}, Device::CPU),
                                                   110.f, 110.f, width / 2.f, height / 2.f, Tensor{}, Tensor{},
                                                   CameraModelType::PINHOLE, name, image, dir / "sidecar.png", width, height, 0);
            auto host_mask = Tensor::empty({height, width}, Device::CPU, DataType::UInt8);
            std::copy(mask.begin(), mask.end(), host_mask.ptr<uint8_t>());
            camera->set_mask_tensor(std::move(host_mask));
            lfs::training::DatasetConfig settings;
            settings.resize_factor = factor;
            auto dataset = std::make_shared<lfs::training::CameraDataset>(
                std::vector<std::shared_ptr<Camera>>{camera}, settings);
            lfs::training::PipelinedAuxiliaryImageConfig auxiliary;
            auxiliary.load_masks = auxiliary.fill_masked_rgb = true;
            auxiliary.mask_threshold = .5f;
            lfs::training::PipelinedDataLoader<lfs::training::InfiniteRandomSampler> loader(
                dataset, lfs::training::InfiniteRandomSampler(1, 7), config(), auxiliary);
            const auto ready = loader.next();
            ASSERT_TRUE(ready);
            ASSERT_TRUE(ready->mask);
            images.push_back(ready->data.image.to(DataType::Float32).cpu().to_vector());
            masks.push_back(ready->mask->to(DataType::Float32).cpu().to_vector());
        }
        ASSERT_EQ(masks[0], masks[1]);
        const size_t plane = masks[0].size();
        ASSERT_EQ(images[0].size(), 3 * plane);
        EXPECT_GT(std::count(masks[0].begin(), masks[0].end(), 0.f), 0);
        for (size_t c = 0; c < 3; ++c)
            for (size_t i = 0; i < plane; ++i)
                if (masks[0][i] > 0.f)
                    EXPECT_EQ(images[0][c * plane + i], images[1][c * plane + i]);
    }

    void MaskRgbCache::check_memory_revision_cache() {
        std::vector<uint8_t> pixels(width * height * 3);
        for (size_t i = 0; i < pixels.size(); ++i)
            pixels[i] = static_cast<uint8_t>((i * 37 + 19) % 256);
        const auto image_path = temp.path / "memory-revision.png";
        ASSERT_TRUE(save_png(image_path, pixels.data(), width, height, 3, 8, 0));

        auto cam = std::make_shared<Camera>(
            Tensor::eye(3, Device::CPU), Tensor::zeros({3}, Device::CPU), 110.f, 110.f,
            width / 2.f, height / 2.f, Tensor{}, Tensor{}, CameraModelType::PINHOLE,
            image_path.filename().string(), image_path, std::filesystem::path{}, width, height, 0);
        auto host_mask = Tensor::empty({height, width}, Device::CPU, DataType::UInt8);
        std::fill_n(host_mask.ptr<uint8_t>(), width * height, uint8_t{255});
        std::fill_n(host_mask.ptr<uint8_t>() + width * height / 3, width * height / 3, uint8_t{0});
        cam->set_mask_tensor(host_mask);

        lfs::io::PipelinedImageLoader loader(config());
        lfs::io::ImageRequest request;
        request.path = image_path;
        request.params.resize_factor = 2;
        request.fill_masked_rgb = true;
        request.mask_params.threshold = .5f;
        request.rgb_mask = cam->raw_mask_tensor();
        request.rgb_mask_revision = cam->mask_revision();
        const auto load = [&](size_t sequence) {
            request.sequence_id = sequence;
            loader.prefetch({request});
            auto ready = loader.get();
            EXPECT_TRUE(ready.error.empty()) << ready.error;
            return ready.tensor.to(DataType::Float32).cpu().to_vector();
        };

        const auto first = load(0);
        ASSERT_FALSE(first.empty());
        const auto published = loader.get_stats();
        ASSERT_GT(published.jpeg_cache_entries, 0u) << "First filled RGB must be published before visit 2";
        ASSERT_EQ(published.cold_path_misses, 1u);

        const auto second = load(1);
        const auto hit = loader.get_stats();
        ASSERT_EQ(hit.hot_path_hits, published.hot_path_hits + 1) << "Visit 2 must use the filled RGB cache";
        ASSERT_EQ(hit.cold_path_misses, published.cold_path_misses);
        ASSERT_EQ(first.size(), second.size());
        ASSERT_TRUE(std::any_of(second.begin(), second.end(), [](float value) { return value > 0.f; }));

        const auto storage = cam->raw_mask_tensor().data_ptr();
        const auto revision = cam->mask_revision();
        std::fill_n(host_mask.ptr<uint8_t>(), width * height, uint8_t{0});
        cam->set_mask_tensor(host_mask);
        ASSERT_EQ(cam->raw_mask_tensor().data_ptr(), storage) << "Only the revision may change the cache key";
        ASSERT_GT(cam->mask_revision(), revision);
        request.rgb_mask = cam->raw_mask_tensor();
        request.rgb_mask_revision = cam->mask_revision();
        const auto third = load(2);
        const auto invalidated = loader.get_stats();
        ASSERT_EQ(invalidated.cold_path_misses, hit.cold_path_misses + 1);
        ASSERT_EQ(invalidated.hot_path_hits, hit.hot_path_hits);
        ASSERT_EQ(third.size(), first.size());
        EXPECT_TRUE(std::all_of(third.begin(), third.end(), [](float value) { return value == 0.f; }));
        EXPECT_NE(first, third);
    }
} // namespace
