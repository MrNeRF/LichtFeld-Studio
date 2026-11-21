/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

// Depth and normal priors no larger than their target travel at their own size
// and are fitted where the loss reads them. The fit must give the numbers the
// loader's resize and the loss's Lanczos give a prior resized in the loader.

#include <gtest/gtest.h>

#include "core/camera.hpp"
#include "core/cuda/lanczos_resize/lanczos_resize.hpp"
#include "core/tensor/internal/cuda_stream_context.hpp"
#include "core/tensor/internal/memory_pool.hpp"
#include "io/nvcodec_image_loader.hpp"
#include "training/dataset.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <random>
#include <string>
#include <thread>
#include <vector>

namespace {
    using lfs::core::DataType;
    using lfs::core::Device;
    using lfs::core::Tensor;
    using lfs::core::TensorShape;

    bool has_cuda_device() {
        int count = 0;
        return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
    }

    // Depth with every kind of value the prior resize treats as invalid.
    Tensor depth_prior(const size_t height, const size_t width) {
        std::mt19937 rng(4092);
        std::uniform_real_distribution<float> value(0.5f, 40.0f);
        std::vector<float> depth(height * width);
        for (size_t i = 0; i < depth.size(); ++i) {
            switch (i % 11) {
            case 3: depth[i] = 0.0f; break;
            case 5: depth[i] = -1.0f; break;
            case 7: depth[i] = std::numeric_limits<float>::quiet_NaN(); break;
            case 9: depth[i] = std::numeric_limits<float>::infinity(); break;
            default: depth[i] = value(rng);
            }
        }
        return Tensor::from_vector(depth, {height, width}, Device::CUDA);
    }

    // Unnormalized normals facing the camera, as the prior transform gives them,
    // with short and zero vectors that the prior resize treats as invalid.
    Tensor normal_prior(const size_t height, const size_t width) {
        std::mt19937 rng(92);
        std::uniform_real_distribution<float> lateral(-0.8f, 0.8f);
        std::uniform_real_distribution<float> facing(0.5f, 1.0f);
        const size_t plane = height * width;
        std::vector<float> normal(3 * plane);
        for (size_t i = 0; i < plane; ++i) {
            const float scale = i % 13 == 4 ? 0.0f : i % 13 == 8 ? 0.3f
                                                                 : 1.0f;
            normal[i] = scale * lateral(rng);
            normal[plane + i] = scale * lateral(rng);
            normal[2 * plane + i] = scale * facing(rng);
        }
        return Tensor::from_vector(normal, {3, height, width}, Device::CUDA);
    }

    std::vector<float> host(const Tensor& tensor) {
        return tensor.cpu().to_vector();
    }

    // The prior kernel the loader enlarges with, written out on the CPU: validity
    // from the nearest pixel, bilinear weights over the valid neighbours,
    // renormalized; normals scaled to unit length.
    std::vector<float> cpu_prior_resize(const std::vector<float>& src, const int channels,
                                        const int sw, const int sh, const int dw, const int dh) {
        const size_t sp = static_cast<size_t>(sw) * sh;
        const size_t dp = static_cast<size_t>(dw) * dh;
        const auto valid = [&](const size_t i) {
            if (channels == 1)
                return std::isfinite(src[i]) && src[i] > 0.0f;
            const float x = src[i], y = src[sp + i], z = src[2 * sp + i];
            return std::isfinite(x) && std::isfinite(y) && std::isfinite(z) && x * x + y * y + z * z >= 0.25f;
        };
        std::vector<float> dst(channels * dp, 0.0f);
        for (int y = 0; y < dh; ++y) {
            for (int x = 0; x < dw; ++x) {
                const float sx = std::clamp((x + 0.5f) * sw / dw - 0.5f, 0.0f, sw - 1.0f);
                const float sy = std::clamp((y + 0.5f) * sh / dh - 0.5f, 0.0f, sh - 1.0f);
                const int x0 = static_cast<int>(sx);
                const int y0 = static_cast<int>(sy);
                const size_t nearest =
                    static_cast<size_t>(static_cast<int>(sy + 0.5f)) * sw + static_cast<int>(sx + 0.5f);
                double value[3] = {};
                double weight = 0.0;
                if (valid(nearest)) {
                    for (int j = 0; j < 2; ++j) {
                        for (int i = 0; i < 2; ++i) {
                            const size_t index =
                                static_cast<size_t>(std::min(y0 + j, sh - 1)) * sw + std::min(x0 + i, sw - 1);
                            if (!valid(index))
                                continue;
                            const double w = (i ? sx - x0 : 1.0 - (sx - x0)) * (j ? sy - y0 : 1.0 - (sy - y0));
                            weight += w;
                            for (int c = 0; c < channels; ++c)
                                value[c] += w * src[c * sp + index];
                        }
                    }
                }
                if (channels == 3)
                    weight = std::sqrt(value[0] * value[0] + value[1] * value[1] + value[2] * value[2]);
                const size_t out = static_cast<size_t>(y) * dw + x;
                for (int c = 0; c < channels; ++c)
                    dst[c * dp + out] = weight > 1e-8 ? static_cast<float>(value[c] / weight) : 0.0f;
            }
        }
        return dst;
    }

    // Bit for bit; NaN only where both have NaN.
    void expect_same(const std::vector<float>& actual, const std::vector<float>& expected) {
        ASSERT_EQ(actual.size(), expected.size());
        for (size_t i = 0; i < actual.size(); ++i) {
            if (std::isnan(expected[i])) {
                ASSERT_TRUE(std::isnan(actual[i])) << "element " << i;
                continue;
            }
            ASSERT_EQ(actual[i], expected[i]) << "element " << i;
        }
    }

    std::unique_ptr<lfs::io::NvCodecImageLoader> make_codec() {
        lfs::io::NvCodecImageLoader::Options options;
        options.decoder_pool_size = 1;
        return std::make_unique<lfs::io::NvCodecImageLoader>(options);
    }
} // namespace

TEST(PriorOwnSize, FitDepthGivesTheLoadersEnlargement) {
    if (!has_cuda_device())
        GTEST_SKIP() << "CUDA device required";
    const auto prior = depth_prior(23, 37);
    const auto loader_output = host(lfs::core::resize_depth_prior(prior, 46, 74));
    // The loader enlarged to the request's size, which the render normally has.
    expect_same(host(lfs::core::fit_depth_prior(prior, 46, 74, 46, 74)), loader_output);
    // A request of 0 x 0 means the render's size.
    expect_same(host(lfs::core::fit_depth_prior(prior, 0, 0, 46, 74)), loader_output);
    // A prior read at the request's size still passes the prior resize, as in the loader.
    expect_same(host(lfs::core::fit_depth_prior(prior, 23, 37, 23, 37)),
                host(lfs::core::resize_depth_prior(prior, 23, 37)));
}

TEST(PriorOwnSize, EnlargementMatchesTheKernelWrittenOutOnTheCpu) {
    if (!has_cuda_device())
        GTEST_SKIP() << "CUDA device required";
    // Release builds the kernel with -use_fast_math, whose division is approximate; with
    // power-of-two output sizes and whole-number ratios every sample point is exact on
    // both sides and clear of a pixel boundary, so only the accumulation differs.
    const auto depth = depth_prior(16, 32);
    const auto normal = normal_prior(16, 32);
    for (const int scale : {2, 4}) {
        SCOPED_TRACE(scale);
        const int dh = 16 * scale;
        const int dw = 32 * scale;
        const auto gpu_depth = host(lfs::core::fit_depth_prior(depth, dh, dw, dh, dw));
        const auto cpu_depth = cpu_prior_resize(host(depth), 1, 32, 16, dw, dh);
        ASSERT_EQ(gpu_depth.size(), cpu_depth.size());
        for (size_t i = 0; i < cpu_depth.size(); ++i) {
            if (cpu_depth[i] == 0.0f)
                ASSERT_EQ(gpu_depth[i], 0.0f) << "depth " << i;
            else
                ASSERT_NEAR(gpu_depth[i], cpu_depth[i], 1e-5f * std::abs(cpu_depth[i])) << "depth " << i;
        }
        const auto gpu_normal = host(lfs::core::fit_normal_prior(normal, dh, dw, dh, dw));
        const auto cpu_normal = cpu_prior_resize(host(normal), 3, 32, 16, dw, dh);
        ASSERT_EQ(gpu_normal.size(), cpu_normal.size());
        for (size_t i = 0; i < cpu_normal.size(); ++i) {
            if (cpu_normal[i] == 0.0f)
                ASSERT_EQ(gpu_normal[i], 0.0f) << "normal " << i;
            else
                ASSERT_NEAR(gpu_normal[i], cpu_normal[i], 1e-5f) << "normal " << i;
        }
    }
}

TEST(PriorOwnSize, FitNormalGivesTheLoadersEnlargement) {
    if (!has_cuda_device())
        GTEST_SKIP() << "CUDA device required";
    const auto prior = normal_prior(23, 37);
    expect_same(host(lfs::core::fit_normal_prior(prior, 46, 74, 46, 74)),
                host(lfs::core::resize_normal_prior(prior, 46, 74)));
    expect_same(host(lfs::core::fit_normal_prior(prior, 23, 37, 23, 37)),
                host(lfs::core::resize_normal_prior(prior, 23, 37)));
}

TEST(PriorOwnSize, FitKeepsTheOldNumbersWhenTheCameraSizeChangedAfterTheRequest) {
    if (!has_cuda_device())
        GTEST_SKIP() << "CUDA device required";
    // The loader enlarged to the requested 46 x 74, then the loss resized that to
    // the 31 x 50 render with Lanczos.
    const auto depth = depth_prior(23, 37);
    expect_same(host(lfs::core::fit_depth_prior(depth, 46, 74, 31, 50)),
                host(lfs::core::lanczos_resize_grayscale(lfs::core::resize_depth_prior(depth, 46, 74), 31, 50, 2)));
    const auto normal = normal_prior(23, 37);
    expect_same(host(lfs::core::fit_normal_prior(normal, 46, 74, 31, 50)),
                host(lfs::core::lanczos_resize_float_chw(lfs::core::resize_normal_prior(normal, 46, 74), 31, 50, 2)));
}

// The fit only enqueues the prior resize on the stream it is given. An input made on another
// stream and dropped by the caller right after the call must survive until that resize has read
// it: the resize waits behind a host gate while the input's block goes back to the producer
// stream, which overwrites it before the gate opens.
TEST(PriorOwnSize, FitKeepsItsInputUntilItsStreamHasReadIt) {
    if (!has_cuda_device())
        GTEST_SKIP() << "CUDA device required";
    using namespace lfs::core;
    // 768 KiB, above the slab allocator's threshold, so the block comes from the stream-ordered pool.
    constexpr size_t H = 384;
    constexpr size_t W = 512;
    cudaStream_t producer{};
    cudaStream_t consumer{};
    cudaStream_t gate_holder{};
    cudaEvent_t gate{};
    ASSERT_EQ(cudaStreamCreateWithFlags(&producer, cudaStreamNonBlocking), cudaSuccess);
    ASSERT_EQ(cudaStreamCreateWithFlags(&consumer, cudaStreamNonBlocking), cudaSuccess);
    ASSERT_EQ(cudaStreamCreateWithFlags(&gate_holder, cudaStreamNonBlocking), cudaSuccess);
    ASSERT_EQ(cudaEventCreateWithFlags(&gate, cudaEventDisableTiming), cudaSuccess);
    std::atomic<bool> released{false};
    std::atomic<bool> returned{false};
    struct OpenGate {
        std::atomic<bool>& flag;
        ~OpenGate() { flag.store(true, std::memory_order_release); }
    } open_on_exit{released};
    Tensor expected;
    Tensor fitted;
    bool reused = false;
    bool gate_held = false;
    {
        const CUDAStreamGuard guard(producer);
        Tensor input = depth_prior(H, W);
        ASSERT_EQ(input.stream(), producer);
        expected = resize_depth_prior(input, H, W, producer);
        // The fit's output is allocated on the consumer stream, so the producer stream never waits for
        // the gated one to reuse a block of it.
        const auto fit_on_consumer = [&] {
            const CUDAStreamGuard on_consumer(consumer);
            return fit_depth_prior(input, H, W, H, W, consumer);
        };
        // Load the kernel and fill the pool first, so the gated call has nothing that waits on the device.
        (void)fit_on_consumer();
        ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
        void* const block = input.data_ptr();

        ASSERT_EQ(cudaLaunchHostFunc(
                      gate_holder,
                      [](void* flag) {
                          while (!static_cast<std::atomic<bool>*>(flag)->load(std::memory_order_acquire)) {
                          }
                      },
                      &released),
                  cudaSuccess);
        ASSERT_EQ(cudaEventRecord(gate, gate_holder), cudaSuccess);
        ASSERT_EQ(cudaStreamWaitEvent(consumer, gate, 0), cudaSuccess);
        // Opens the gate even if something below blocks on the gated stream.
        std::thread opener([&] {
            const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
            while (!returned.load(std::memory_order_acquire) && std::chrono::steady_clock::now() < deadline)
                std::this_thread::yield();
            released.store(true, std::memory_order_release);
        });
        fitted = fit_on_consumer();
        input = Tensor();
        // NaN bytes, which the prior resize treats as invalid depth.
        Tensor sentinel = Tensor::empty({H, W}, Device::CUDA, DataType::Float32);
        ASSERT_EQ(cudaMemsetAsync(sentinel.data_ptr(), 0xFF, sentinel.bytes(), producer), cudaSuccess);
        reused = sentinel.data_ptr() == block;
        // Unless the producer stream waits for the resize, the sentinel lands while the gate holds.
        const auto settle = std::chrono::steady_clock::now() + std::chrono::milliseconds(500);
        while (cudaStreamQuery(producer) == cudaErrorNotReady && std::chrono::steady_clock::now() < settle)
            std::this_thread::yield();
        gate_held = !released.load(std::memory_order_acquire);
        returned.store(true, std::memory_order_release);
        opener.join();
        ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    }
    const auto result = host(fitted);
    fitted = Tensor();
    CudaMemoryPool::instance().release_stream(producer);
    CudaMemoryPool::instance().release_stream(consumer);
    EXPECT_EQ(cudaStreamDestroy(producer), cudaSuccess);
    EXPECT_EQ(cudaStreamDestroy(consumer), cudaSuccess);
    EXPECT_EQ(cudaStreamDestroy(gate_holder), cudaSuccess);
    EXPECT_EQ(cudaEventDestroy(gate), cudaSuccess);
    ASSERT_TRUE(gate_held) << "the fit waited for the gated stream, so the test cannot observe the order";
    if (!reused)
        GTEST_SKIP() << "the pool did not hand the input's block back, so the test cannot observe the order";
    expect_same(result, host(expected));
}

// The loader's slot estimate charges a prior at its own size only when the prior travels so;
// a prior larger than the image is resized to the image's size and charged at it.
TEST(PriorOwnSize, SlotChargesPriorsLargerThanTheImageAtTheImagesSize) {
    lfs::core::Camera camera;
    camera.set_image_dimensions(1237, 822);
    const size_t image = size_t{1237} * 822;
    camera.set_prior_sizes({4946, 3286}, {4946, 3286});
    EXPECT_EQ(lfs::training::pipelined_prior_pixels(camera, camera.depth_prior_size()), image);
    EXPECT_EQ(lfs::training::pipelined_prior_pixels(camera, camera.normal_prior_size()), image);
    EXPECT_EQ(lfs::training::pipelined_prior_pixels(camera, {1238, 822}), image);
    // No larger than the image: its own size. Unknown: the image's.
    EXPECT_EQ(lfs::training::pipelined_prior_pixels(camera, {1237, 822}), image);
    EXPECT_EQ(lfs::training::pipelined_prior_pixels(camera, {618, 411}), size_t{618} * 411);
    EXPECT_EQ(lfs::training::pipelined_prior_pixels(camera, {}), image);
}

namespace {
    // One lossless J2K round trip of a prior file's own samples, in the two J2K forms the prior
    // cache uses for them: 16-bit normals as RGB, 8-bit depth as grayscale. (8-bit normals are
    // kept raw: nvJPEG2000 can hang decoding 8-bit RGB codestreams, as it does for 37x53 noise.)
    struct SampleCase {
        bool rgb;
        size_t height;
        size_t width;
    };

    class PriorOwnSizeJpeg2k : public ::testing::TestWithParam<SampleCase> {};
} // namespace

TEST_P(PriorOwnSizeJpeg2k, RoundTripKeepsTheSamples) {
    if (!has_cuda_device())
        GTEST_SKIP() << "CUDA device required";
    std::unique_ptr<lfs::io::NvCodecImageLoader> codec;
    try {
        codec = make_codec();
    } catch (const std::exception& e) {
        GTEST_SKIP() << "nvImageCodec unavailable: " << e.what();
    }
    const auto [rgb, height, width] = GetParam();
    const size_t count = height * width * (rgb ? 3 : 1);
    std::mt19937 rng(4092);
    std::uniform_int_distribution<int> sample(0, rgb ? 65535 : 255);
    std::vector<uint16_t> u16(count);
    std::vector<uint8_t> u8(count);
    for (size_t i = 0; i < count; ++i) {
        u16[i] = static_cast<uint16_t>(sample(rng));
        u8[i] = static_cast<uint8_t>(u16[i]);
    }
    std::vector<uint8_t> bytes;
    if (rgb) {
        // uint16 samples travel in a Float16 container, as the loader holds them.
        const auto input = Tensor::from_blob(u16.data(), TensorShape({height, width, 3}), Device::CPU, DataType::Float16)
                               .to(Device::CUDA);
        bytes = codec->encode_to_jpeg2k(input, lfs::io::RgbLayout::HWC);
    } else {
        const auto input = Tensor::from_blob(u8.data(), TensorShape({height, width}), Device::CPU, DataType::UInt8)
                               .to(Device::CUDA);
        bytes = codec->encode_grayscale_to_jpeg2k(input, nullptr, true, true);
    }
    const auto decoded = codec->decode_jpeg2k_16bit_from_memory_gpu(bytes, nullptr, true, true);
    ASSERT_EQ(decoded.shape(), rgb ? TensorShape({height, width, 3}) : TensorShape({height, width}));
    ASSERT_EQ(decoded.dtype(), rgb ? DataType::Float32 : DataType::UInt8);
    const auto values = host(decoded);
    for (size_t i = 0; i < count; ++i) {
        // 16-bit samples come back as the first visit scales them, 8-bit ones as they are.
        const float expected = rgb ? static_cast<float>(u16[i]) * (1.0f / 65535.f) : static_cast<float>(u8[i]);
        ASSERT_EQ(values[i], expected) << "sample " << i;
    }
}

INSTANTIATE_TEST_SUITE_P(
    Sizes, PriorOwnSizeJpeg2k,
    ::testing::Values(SampleCase{true, 37, 53}, SampleCase{true, 1951, 2610}, SampleCase{false, 37, 53},
                      SampleCase{false, 1951, 2610}),
    [](const ::testing::TestParamInfo<SampleCase>& info) {
        return std::string(info.param.rgb ? "rgb16_" : "gray8_") + std::to_string(info.param.height) + "x" +
               std::to_string(info.param.width);
    });
