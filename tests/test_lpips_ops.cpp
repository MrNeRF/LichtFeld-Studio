/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/failure_report.hpp"
#include "core/logger.hpp"
#include "core/nn/models/lpips.hpp"
#include "core/nn/nn_kernels.hpp"
#include "core/parameters.hpp"
#include "core/tensor_cuda_interop.hpp"
#include "cuda_backend_test.hpp"
#include "lfs/training/ops/registry.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <gtest/gtest.h>

namespace {
    using namespace lfs::core;
    namespace kernels = lfs::core::nn::kernels;
    using LpipsOpsTest = lfs::test::CudaBackendTest;

    Tensor pattern(TensorShape shape, int seed, DataType dtype = DataType::Float16) {
        std::vector<float> values(shape.elements());
        for (size_t i = 0; i < values.size(); ++i)
            values[i] = static_cast<float>(static_cast<int>((i * 17 + seed * 13) % 101) - 50) / 256.f;
        return Tensor::from_vector(values, shape, Device::GPU).to(dtype);
    }
    void same(const Tensor& a, const Tensor& b) {
        ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
        const auto x = a.cpu().contiguous(), y = b.cpu().contiguous();
        ASSERT_EQ(x.bytes(), y.bytes());
        EXPECT_EQ(std::memcmp(x.data_ptr(), y.data_ptr(), x.bytes()), 0);
    }
    const lfs::gpu_ops::LpipsOps& ops() {
        return *lfs::training::training_ops(GpuBackend::CUDA).lpips;
    }

    TEST_F(LpipsOpsTest, RejectsOperandDtypesBeforeLaunching) {
        auto weight = pattern({64, 8, 3, 3}, 1, DataType::Float32);
        auto taps = Tensor::zeros({9, 64, 8}, Device::GPU, DataType::Float16);
        EXPECT_THROW(ops().weight_taps(weight, taps), std::runtime_error);

        auto rgb = pattern({1, 3, 4, 4}, 2, DataType::Float32);
        auto rgb_weight = pattern({64, 3, 3, 3}, 3), bias = pattern({64}, 4);
        auto rgb_output = Tensor::zeros({1, 64, 4, 4}, Device::GPU, DataType::Float32);
        EXPECT_THROW(ops().rgb_conv(rgb, rgb_weight, bias, rgb_output, {}), std::runtime_error);

        auto input = pattern({1, 8, 4, 4}, 5, DataType::Float32);
        // Retain enough backing storage to make the old float32 write safe while
        // testing rejection of a float16 output view.
        auto backing = Tensor::zeros({2, 64, 4, 4}, Device::GPU, DataType::Float16);
        auto output = backing.slice(0, 0, 1);
        Tensor absent;
        lfs::gpu_ops::ConvParams params;
        params.pad_h = params.pad_w = 1;
        reset_failure_report_dedup_for_testing();
        std::string report;
        const auto handler = Logger::get().add_log_handler(
            [&](LogLevel level, const SourceSite&, std::string_view message) {
                if (level == LogLevel::Error)
                    report += message;
            });
        EXPECT_THROW(ops().convolution(input, weight, absent, absent, output, absent, params), std::runtime_error);
        Logger::get().remove_log_handler(handler);
        EXPECT_NE(report.find("LFS FAILURE REPORT"), std::string::npos);
        EXPECT_NE(report.find("Family: tensor contract violation"), std::string::npos);
        EXPECT_NE(report.find("Detection site:"), std::string::npos);
        EXPECT_NE(report.find("lpips_cuda.cpp"), std::string::npos);
        EXPECT_NE(report.find("output"), std::string::npos);
        EXPECT_NE(report.find("dtype=float16"), std::string::npos);
        EXPECT_NE(report.find("shape=[1, 64, 4, 4]"), std::string::npos);
        EXPECT_NE(report.find("Host stack trace:"), std::string::npos);
        EXPECT_NE(report.find("convolution"), std::string::npos);
        EXPECT_NE(report.find("RejectsOperandDtypesBeforeLaunching"), std::string::npos);

        auto features = Tensor::zeros({1, 8, 4, 4}, Device::GPU, DataType::Float32);
        auto linear = pattern({1, 8, 1, 1}, 6);
        auto score = Tensor::zeros({1}, Device::GPU);
        EXPECT_THROW(ops().pool_reduce(features, features, linear, score, absent, absent,
                                       {0, 4, 0, 4, 1.f / 16.f}),
                     std::runtime_error);
        ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    }

    TEST_F(LpipsOpsTest, WeightTapsRejectsInvalidOperands) {
        auto weight = pattern({64, 8, 3, 3}, 1);
        auto taps = Tensor::zeros({9, 64, 8}, Device::GPU, DataType::Float16);
        for (const auto& invalid : {Tensor{}, weight.cpu(), weight.transpose(2, 3),
                                    weight.reshape({64, 8, 9}), pattern({64, 8, 1, 1}, 1)}) {
            EXPECT_THROW(ops().weight_taps(invalid, taps), std::runtime_error);
        }
        for (auto invalid : {taps.to(DataType::Float32), taps.cpu(), taps.transpose(0, 1),
                             taps.slice(0, 0, 8)}) {
            EXPECT_THROW(ops().weight_taps(weight, invalid), std::runtime_error);
        }
        if (gpu_backend_available(GpuBackend::Vulkan)) {
            GpuBackendScope scope(GpuBackend::Vulkan);
            auto foreign = Tensor::zeros(weight.shape(), Device::GPU, DataType::Float16);
            EXPECT_THROW(ops().weight_taps(foreign, taps), std::runtime_error);
        }
    }

    TEST_F(LpipsOpsTest, RgbConvolutionRejectsInvalidOperands) {
        auto input = pattern({1, 3, 4, 4}, 1, DataType::Float32);
        auto weight = pattern({64, 3, 3, 3}, 2), bias = pattern({64}, 3);
        auto output = Tensor::zeros({1, 64, 4, 4}, Device::GPU, DataType::Float16);
        for (const auto& invalid : {input.to(DataType::Float16), input.cpu(), input.transpose(2, 3),
                                    input.reshape({1, 4, 3, 4})}) {
            EXPECT_THROW(ops().rgb_conv(invalid, weight, bias, output, {}), std::runtime_error);
        }
        for (const auto& invalid : {weight.to(DataType::Float32), weight.cpu(), weight.transpose(2, 3),
                                    weight.reshape({64, 3, 1, 9})}) {
            EXPECT_THROW(ops().rgb_conv(input, invalid, bias, output, {}), std::runtime_error);
        }
        for (const auto& invalid : {bias.to(DataType::Float32), bias.cpu(), bias.slice(0, 0, 63)}) {
            EXPECT_THROW(ops().rgb_conv(input, weight, invalid, output, {}), std::runtime_error);
        }
        for (auto invalid : {output.cpu(), output.transpose(2, 3), output.reshape({1, 32, 8, 4})}) {
            EXPECT_THROW(ops().rgb_conv(input, weight, bias, invalid, {}), std::runtime_error);
        }
    }

    TEST_F(LpipsOpsTest, ConvolutionRejectsInvalidOperands) {
        auto input = pattern({1, 8, 4, 4}, 1), weight = pattern({64, 8, 3, 3}, 2), bias = pattern({64}, 3);
        auto output = Tensor::zeros({1, 64, 4, 4}, Device::GPU, DataType::Float16);
        auto taps = Tensor::zeros({9, 64, 8}, Device::GPU, DataType::Float16);
        Tensor absent;
        lfs::gpu_ops::ConvParams params;
        params.pad_h = params.pad_w = 1;
        for (const auto& invalid : {input.to(DataType::Int32), input.cpu(), input.transpose(2, 3),
                                    input.reshape({8, 4, 4})}) {
            EXPECT_THROW(ops().convolution(invalid, weight, absent, bias, output, absent, params), std::runtime_error);
        }
        for (const auto& invalid : {weight.to(DataType::Float32), weight.cpu(), weight.transpose(2, 3),
                                    weight.reshape({64, 4, 3, 6})}) {
            EXPECT_THROW(ops().convolution(input, invalid, absent, bias, output, absent, params), std::runtime_error);
        }
        for (const auto& invalid : {bias.to(DataType::Float32), bias.cpu(), bias.slice(0, 0, 63)}) {
            EXPECT_THROW(ops().convolution(input, weight, absent, invalid, output, absent, params), std::runtime_error);
        }
        for (auto invalid : {output.to(DataType::Float32), output.cpu(), output.transpose(2, 3),
                             output.reshape({1, 64, 2, 8})}) {
            EXPECT_THROW(ops().convolution(input, weight, absent, bias, invalid, absent, params), std::runtime_error);
        }
        for (auto invalid : {taps.to(DataType::Float32), taps.cpu(), taps.transpose(0, 1), taps.slice(0, 0, 8)}) {
            EXPECT_THROW(ops().convolution(input, weight, invalid, bias, output, absent, params), std::runtime_error);
            EXPECT_THROW(ops().convolution(input, weight, absent, bias, output, invalid, params), std::runtime_error);
        }
    }

    TEST_F(LpipsOpsTest, PoolReduceRejectsInvalidOperands) {
        auto x = pattern({1, 8, 4, 4}, 1), y = pattern({1, 8, 4, 4}, 2), weight = pattern({1, 8, 1, 1}, 3);
        auto score = Tensor::zeros({1}, Device::GPU);
        auto px = Tensor::zeros({1, 8, 2, 2}, Device::GPU, DataType::Float16), py = px.clone();
        const lfs::gpu_ops::PoolReduceParams params{0, 4, 0, 4, 1.f / 16.f};
        for (const auto& invalid : {x.to(DataType::Float32), x.cpu(), x.transpose(2, 3), x.reshape({8, 4, 4})}) {
            EXPECT_THROW(ops().pool_reduce(invalid, y, weight, score, px, py, params), std::runtime_error);
            EXPECT_THROW(ops().pool_reduce(x, invalid, weight, score, px, py, params), std::runtime_error);
        }
        for (const auto& invalid : {weight.to(DataType::Float32), weight.cpu(), weight.slice(1, 0, 7)}) {
            EXPECT_THROW(ops().pool_reduce(x, y, invalid, score, px, py, params), std::runtime_error);
        }
        for (auto invalid : {Tensor{}, score.to(DataType::Float16), score.cpu()}) {
            EXPECT_THROW(ops().pool_reduce(x, y, weight, invalid, px, py, params), std::runtime_error);
        }
        for (auto invalid : {px.to(DataType::Float32), px.cpu(), px.transpose(2, 3), px.slice(1, 0, 7)}) {
            EXPECT_THROW(ops().pool_reduce(x, y, weight, score, invalid, py, params), std::runtime_error);
            EXPECT_THROW(ops().pool_reduce(x, y, weight, score, px, invalid, params), std::runtime_error);
        }
    }

    TEST_F(LpipsOpsTest, Float32ConvolutionMatchesConstantReference) {
        auto input = Tensor::ones({1, 8, 4, 4}, Device::GPU);
        auto weight = Tensor::ones({2, 8, 3, 3}, Device::GPU);
        auto output = Tensor::zeros({1, 2, 2, 2}, Device::GPU);
        Tensor absent;
        ops().convolution(input, weight, absent, absent, output, absent, {});
        for (const auto value : output.cpu().to_vector())
            EXPECT_EQ(value, 72.f);
    }

    TEST_F(LpipsOpsTest, WeightTapsMatchLauncher) {
        TensorCudaStream stream;
        CUDAStreamGuard guard(stream.get());
        auto w = pattern({128, 64, 3, 3}, 7);
        auto actual = Tensor::empty({9, 128, 64}, Device::GPU, DataType::Float16);
        auto expected = Tensor::empty(actual.shape(), Device::GPU, actual.dtype());
        ops().weight_taps(w, actual);
        kernels::conv3x3_weight_taps(w.data_ptr(), expected.data_ptr(), 128, 64, getCurrentCUDAStream());
        same(actual, expected);
    }

    TEST_F(LpipsOpsTest, RgbConvolutionMatchesLauncher) {
        TensorCudaStream stream;
        CUDAStreamGuard guard(stream.get());
        auto x = pattern({1, 3, 7, 9}, 1, DataType::Float32);
        auto w = pattern({64, 3, 3, 3}, 2), b = pattern({64}, 3);
        auto actual = Tensor::empty({1, 64, 7, 9}, Device::GPU, DataType::Float16);
        auto expected = Tensor::empty(actual.shape(), Device::GPU, actual.dtype());
        for (bool normalize : {false, true}) {
            lfs::gpu_ops::RGBConvParams p{{-.030f, -.088f, -.188f}, {.458f, .448f, .450f}, normalize};
            ops().rgb_conv(x, w, b, actual, p);
            kernels::lpips_rgb_conv3x3(x.ptr<float>(), w.data_ptr(), b.data_ptr(), expected.data_ptr(),
                                       p.shift.data(), p.scale.data(), normalize, 1, 7, 9, getCurrentCUDAStream());
            same(actual, expected);
        }
    }

    TEST_F(LpipsOpsTest, ConvolutionMatchesLauncher) {
        TensorCudaStream stream;
        CUDAStreamGuard guard(stream.get());
        auto x = pattern({1, 64, 7, 9}, 1), w = pattern({64, 64, 3, 3}, 2), b = pattern({64}, 3);
        auto taps = Tensor::empty({9, 64, 64}, Device::GPU, DataType::Float16);
        kernels::conv3x3_weight_taps(w.data_ptr(), taps.data_ptr(), 64, 64, getCurrentCUDAStream());
        auto actual = Tensor::empty({1, 64, 7, 9}, Device::GPU, DataType::Float16);
        auto expected = Tensor::empty(actual.shape(), Device::GPU, actual.dtype());
        lfs::gpu_ops::ConvParams p;
        p.pad_h = p.pad_w = 1;
        p.activation = nn::Activation::Relu;
        Tensor absent;
        for (bool cached : {true, false}) {
            auto scratch = Tensor::empty(taps.shape(), Device::GPU, DataType::Float16);
            auto reference_scratch = Tensor::empty(taps.shape(), Device::GPU, DataType::Float16);
            ops().convolution(x, w, cached ? taps : absent, b, actual, cached ? absent : scratch, p);
            kernels::conv2d_implicit(x.data_ptr(), w.data_ptr(), cached ? taps.data_ptr() : nullptr,
                                     b.data_ptr(), expected.data_ptr(), cached ? nullptr : reference_scratch.data_ptr(),
                                     1, 64, 7, 9, 64, 3, 3, 7, 9, 1, 1, 1, 1, 1, 1, 0,
                                     static_cast<int>(nn::Activation::Relu), DataType::Float16, getCurrentCUDAStream());
            same(actual, expected);
        }
    }

    TEST_F(LpipsOpsTest, PoolReduceMatchesLauncher) {
        TensorCudaStream stream;
        CUDAStreamGuard guard(stream.get());
        // One reduction block makes byte equality independent of atomic scheduling.
        auto x = pattern({1, 64, 2, 8}, 1), y = pattern({1, 64, 2, 8}, 2), w = pattern({1, 64, 1, 1}, 3);
        for (bool pool : {true, false}) {
            auto actual = Tensor::zeros({1}, Device::GPU), expected = Tensor::zeros({1}, Device::GPU);
            auto px = Tensor::empty({1, 64, 1, 4}, Device::GPU, DataType::Float16);
            auto py = Tensor::empty(px.shape(), Device::GPU, px.dtype());
            auto ex = Tensor::empty(px.shape(), Device::GPU, px.dtype());
            auto ey = Tensor::empty(px.shape(), Device::GPU, px.dtype());
            Tensor absent;
            lfs::gpu_ops::PoolReduceParams p{0, 2, 1, 7, 1.f / 12.f};
            ops().pool_reduce(x, y, w, actual, pool ? px : absent, pool ? py : absent, p);
            kernels::lpips_pool_reduce(x.data_ptr(), y.data_ptr(), w.data_ptr(), expected.ptr<float>(),
                                       pool ? ex.data_ptr() : nullptr, pool ? ey.data_ptr() : nullptr,
                                       1, 64, 2, 8, p.y0, p.y1, p.x0, p.x1, p.inverse_count, getCurrentCUDAStream());
            same(actual, expected);
            if (pool) {
                same(px, ex);
                same(py, ey);
            }
        }
    }

    TEST_F(LpipsOpsTest, ModelUsesEveryEntryAndReusesWeightTaps) {
        TensorCudaStream stream;
        CUDAStreamGuard guard(stream.get());
        const char* home = std::getenv("HOME");
        if (!home)
            GTEST_SKIP() << "No model cache";
        const auto path = std::filesystem::path(home) / ".lichtfeld/onnx/lpips-vgg16-v0.1.lfw";
        if (!std::filesystem::exists(path))
            GTEST_SKIP() << "LPIPS weights unavailable";
        auto model = nn::models::Lpips::load(path, Device::GPU, DataType::Float16);
        ASSERT_TRUE(model);
        static std::array<int, 4> calls;
        calls = {};
        static const nn::LpipsDispatch counted{
            .weight_taps = [](const Tensor& w, Tensor& t) { ++calls[0]; ops().weight_taps(w, t); },
            .rgb_conv = [](const Tensor& x, const Tensor& w, const Tensor& b, Tensor& y, const nn::RGBConvParams& p) {
                ++calls[1]; ops().rgb_conv(x, w, b, y, p); },
            .convolution = [](const Tensor& x, const Tensor& w, const Tensor& t, const Tensor& b, Tensor& y, Tensor& scratch, const nn::Conv2dParams& p) {
                ++calls[2]; ops().convolution(x, w, t, b, y, scratch, p); },
            .pool_reduce = [](const Tensor& x, const Tensor& y, const Tensor& w, Tensor& score, Tensor& px, Tensor& py, const nn::PoolReduceParams& p) {
                ++calls[3]; ops().pool_reduce(x, y, w, score, px, py, p); },
        };
        model->set_dispatch(counted);
        auto x = pattern({1, 3, 16, 16}, 9, DataType::Float32);
        for (int i = 0; i < 2; ++i) {
            auto value = model->forward(x, x);
            ASSERT_TRUE(value);
            // Float normalization leaves the same tiny self-distance on the direct path.
            EXPECT_NEAR(*value, 0.f, 1e-12f);
        }
        auto direct = nn::models::Lpips::load(path, Device::GPU, DataType::Float16);
        ASSERT_TRUE(direct);
        auto y = pattern({1, 3, 16, 16}, 12, DataType::Float32);
        auto reference = direct->forward(x, y);
        auto actual = model->forward(x, y);
        ASSERT_TRUE(reference);
        ASSERT_TRUE(actual);
        EXPECT_GT(*actual, 0.f);
        // Full-image score accumulation uses float atomics; individual ops above
        // compare bytes with a single reduction block.
        EXPECT_NEAR(*actual, *reference, 1e-6f);
        EXPECT_EQ(calls, (std::array<int, 4>{kernels::conv3x3_mma_available() ? 12 : 0, 6, 72, 15}));
        auto tiled = nn::models::Lpips::load(path, Device::GPU, DataType::Float16,
                                             nn::models::InputScaling::Identity, 1);
        ASSERT_TRUE(tiled);
        tiled->set_dispatch(ops());
        ASSERT_EQ(tiled->tile_size_for(33, 35), 16u);
        x = pattern({1, 3, 33, 35}, 9, DataType::Float32);
        y = pattern({1, 3, 33, 35}, 12, DataType::Float32);
        reference = direct->forward(x, y);
        actual = tiled->forward(x, y);
        ASSERT_TRUE(reference);
        ASSERT_TRUE(actual);
        EXPECT_NEAR(*actual, *reference, 1e-6f);
    }

    TEST(LpipsCapability, MissingOnUnimplementedBackends) {
        using namespace lfs::training;
        param::TrainingParameters params;
        params.optimization.enable_eval = true;
        EXPECT_TRUE(required_training_families(params, {}).test(static_cast<size_t>(Family::Lpips)));
        FamilySet required;
        required.set(static_cast<size_t>(Family::Lpips));
        EXPECT_EQ(missing_training_families(TrainingOps{}, required), std::vector<std::string_view>{"Lpips"});
        EXPECT_FALSE(unavailable_training_family(GpuBackend::CUDA, Family::Lpips));
        EXPECT_NE(training_ops(GpuBackend::Vulkan).lpips, nullptr);
        EXPECT_FALSE(unavailable_training_family(GpuBackend::Vulkan, Family::Lpips));
        for (const auto backend : {GpuBackend::Metal}) {
            EXPECT_EQ(training_ops(backend).lpips, nullptr);
            const auto reason = unavailable_training_family(backend, Family::Lpips);
            ASSERT_TRUE(reason);
            EXPECT_NE(reason->find("Lpips"), std::string::npos);
        }
    }

    TEST(LpipsVulkanRuntime, HalfConvolutionChunksSpatialTilesAcrossBatches) {
        if (!gpu_backend_available(GpuBackend::Vulkan))
            GTEST_SKIP();
        const lfs::test::DefaultGpuBackendForTesting backend(GpuBackend::Vulkan);
        ASSERT_TRUE(backend.switched());
        constexpr size_t batch = 32769, width = 17;
        const auto& table = *lfs::training::training_ops(GpuBackend::Vulkan).lpips;
        auto input = Tensor::ones({batch, 32, 1, width}, Device::GPU, DataType::Float16);
        auto weight = Tensor::ones({1, 32, 3, 3}, Device::GPU, DataType::Float16);
        auto actual = Tensor::empty({batch, 1, 1, width}, Device::GPU, DataType::Float16);
        Tensor absent, scratch;
        lfs::gpu_ops::ConvParams params;
        params.pad_h = params.pad_w = 1;
        table.convolution(input, weight, absent, absent, actual, scratch, params);
        const auto values = actual.to(DataType::Float32).cpu().to_vector();
        for (size_t i = 0; i < values.size(); ++i) {
            const size_t x = i % width;
            ASSERT_EQ(values[i], x == 0 || x == width - 1 ? 64.f : 96.f) << i;
        }
    }

    TEST(LpipsVulkanRuntime, HalfConvolutionMatchesExactBorderReference) {
        if (!gpu_backend_available(GpuBackend::Vulkan))
            GTEST_SKIP();
        const lfs::test::DefaultGpuBackendForTesting backend(GpuBackend::Vulkan);
        ASSERT_TRUE(backend.switched());
        constexpr size_t batch = 2, height = 5, width = 19;
        const auto& table = *lfs::training::training_ops(GpuBackend::Vulkan).lpips;
        for (const size_t channels : {32u, 64u, 96u, 128u}) {
            constexpr size_t outputs = 96;
            auto x = pattern({batch, channels, height, width}, 1);
            auto w = pattern({outputs, channels, 3, 3}, 2);
            auto bias = pattern({outputs}, 3);
            const auto hx = x.to(DataType::Float32).cpu().to_vector();
            const auto hw = w.to(DataType::Float32).cpu().to_vector();
            const auto hb = bias.to(DataType::Float32).cpu().to_vector();
            auto taps = Tensor::empty({9, outputs, channels}, Device::GPU, DataType::Float16);
            table.weight_taps(w, taps);
            auto actual = Tensor::empty({batch, outputs, height, width}, Device::GPU, DataType::Float16);
            Tensor absent, scratch;
            for (const bool replicate : {false, true}) {
                lfs::gpu_ops::ConvParams params;
                params.pad_h = params.pad_w = 1;
                params.pad_mode = replicate ? nn::ConvPadMode::Replicate : nn::ConvPadMode::Zeros;
                params.activation = nn::Activation::Relu;
                std::vector<float> reference(actual.numel());
                // Dyadic inputs keep these sums exact in float, independently
                // of the matrix reduction order; compare the final half bits.
                for (size_t n = 0; n < batch; ++n)
                    for (size_t oc = 0; oc < outputs; ++oc)
                        for (size_t y = 0; y < height; ++y)
                            for (size_t xout = 0; xout < width; ++xout) {
                                float sum = hb[oc];
                                for (size_t ic = 0; ic < channels; ++ic)
                                    for (int ky = 0; ky < 3; ++ky)
                                        for (int kx = 0; kx < 3; ++kx) {
                                            int iy = int(y) + ky - 1, ix = int(xout) + kx - 1;
                                            if (replicate) {
                                                iy = std::clamp(iy, 0, int(height) - 1);
                                                ix = std::clamp(ix, 0, int(width) - 1);
                                            }
                                            if (iy >= 0 && iy < int(height) && ix >= 0 && ix < int(width))
                                                sum += hx[((n * channels + ic) * height + size_t(iy)) * width + size_t(ix)] *
                                                       hw[((oc * channels + ic) * 3 + size_t(ky)) * 3 + size_t(kx)];
                                        }
                                reference[((n * outputs + oc) * height + y) * width + xout] = std::max(0.f, sum);
                            }
                const auto expected = Tensor::from_vector(reference, actual.shape(), Device::CPU).to(DataType::Float16);
                for (const bool cached : {false, true}) {
                    SCOPED_TRACE(::testing::Message() << channels << " channels, replicate=" << replicate << ", cached=" << cached);
                    table.convolution(x, w, cached ? taps : absent, bias, actual, scratch, params);
                    const auto result = actual.cpu();
                    EXPECT_EQ(std::memcmp(result.data_ptr(), expected.data_ptr(), expected.bytes()), 0);
                }
            }
        }
    }

    TEST(LpipsVulkanRuntime, TiledForwardUsesVulkanTensorOperations) {
        const char* home = std::getenv("HOME");
        if (!home)
            GTEST_SKIP() << "No model cache";
        const auto path = std::filesystem::path(home) / ".lichtfeld/onnx/lpips-vgg16-v0.1.lfw";
        if (!std::filesystem::exists(path))
            GTEST_SKIP() << "LPIPS weights unavailable";

        GpuBackendScope backend(GpuBackend::Vulkan);
        auto model = nn::models::Lpips::load(path, Device::GPU, DataType::Float16,
                                             nn::models::InputScaling::Identity, 1);
        ASSERT_TRUE(model);
        model->set_dispatch(*lfs::training::training_ops(GpuBackend::Vulkan).lpips);
        ASSERT_EQ(model->tile_size_for(33, 35), 16u);
        const auto x = pattern({1, 3, 33, 35}, 9, DataType::Float32);
        const auto y = pattern({1, 3, 33, 35}, 12, DataType::Float32);
        const auto value = model->forward(x, y);
        ASSERT_TRUE(value);
        EXPECT_GT(*value, 0.f);
    }

    TEST(LpipsVulkanRuntime, LargeWeightTapsDispatchCoversAllOutputs) {
        GpuBackendScope backend(GpuBackend::Vulkan);
        constexpr size_t output_channels = 1024;
        constexpr size_t input_channels = 2048;
        const TensorShape shape{output_channels, input_channels, 3, 3};
        std::vector<float> values(shape.elements());
        for (size_t i = 0; i < values.size(); ++i)
            values[i] = static_cast<float>(i % 251) / 251.f;
        const auto source = Tensor::from_vector(values, shape, Device::CPU).to(DataType::Float16);
        const auto weight = source.gpu();
        auto actual = Tensor::empty({9, output_channels, input_channels}, Device::GPU, DataType::Float16);
        auto expected = Tensor::empty(actual.shape(), Device::CPU, DataType::Float16);
        const auto* source_values = static_cast<const uint16_t*>(source.data_ptr());
        auto* expected_values = static_cast<uint16_t*>(expected.data_ptr());
        for (size_t tap = 0; tap < 9; ++tap)
            for (size_t oc = 0; oc < output_channels; ++oc)
                for (size_t ic = 0; ic < input_channels; ++ic)
                    expected_values[tap * output_channels * input_channels + oc * input_channels + ic] =
                        source_values[(oc * input_channels + ic) * 9 + tap];

        lfs::training::training_ops(GpuBackend::Vulkan).lpips->weight_taps(weight, actual);
        const auto actual_cpu = actual.cpu();
        EXPECT_EQ(std::memcmp(actual_cpu.data_ptr(), expected.data_ptr(), actual.bytes()), 0);
    }

    TEST(LpipsVulkanRuntime, FullResolutionMetricUsesBoundedDispatches) {
        const char* home = std::getenv("HOME");
        if (!home)
            GTEST_SKIP() << "No model cache";
        const auto path = std::filesystem::path(home) / ".lichtfeld/onnx/lpips-vgg16-v0.1.lfw";
        if (!std::filesystem::exists(path))
            GTEST_SKIP() << "LPIPS weights unavailable";

        GpuBackendScope backend(GpuBackend::Vulkan);
        auto model = nn::models::Lpips::load(path, Device::GPU, DataType::Float16,
                                             nn::models::InputScaling::Identity,
                                             512ULL * 1024ULL * 1024ULL);
        ASSERT_TRUE(model);
        model->set_dispatch(*lfs::training::training_ops(GpuBackend::Vulkan).lpips);
        ASSERT_LT(model->tile_size_for(840, 1297), 1297u);
        const auto x = pattern({1, 3, 840, 1297}, 9, DataType::Float32);
        const auto y = pattern({1, 3, 840, 1297}, 12, DataType::Float32);
        const auto value = model->forward(x, y);
        ASSERT_TRUE(value);
        EXPECT_TRUE(std::isfinite(*value));
    }
} // namespace
