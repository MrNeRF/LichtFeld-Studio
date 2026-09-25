/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "core/detail/gpu_backend_ops.hpp"
#include "cuda_backend_test.hpp"

#include "core/cuda/lanczos_resize/lanczos_resize.hpp"
#include "core/cuda/undistort/undistort.hpp"
#include "core/tensor/backend/cuda/runtime/cuda_stream_context.hpp"
#include "core/tensor/backend/vulkan/vk_context.hpp"
#include "core/tensor_backend.hpp"
#include "core/tensor_environment.hpp"
#include "core/tensor_image.hpp"
#include "core/tensor_ppisp.hpp"
#include <cmath>
#include <gtest/gtest.h>
#include <limits>
#include <optional>

namespace {
    using namespace lfs::core;
    enum class Storage { CPU,
                         CUDA,
                         Vulkan };
    class TensorImageOps : public ::testing::TestWithParam<Storage> {
    protected:
        Device device = Device::CPU;
        std::optional<GpuBackendScope> scope;
        void SetUp() override {
            const auto backend = GetParam() == Storage::Vulkan ? GpuBackend::Vulkan : GpuBackend::CUDA;
            device = GetParam() == Storage::CPU ? Device::CPU : Device::GPU;
            if (device == Device::GPU && !gpu_backend_available(backend))
                GTEST_SKIP();
            scope.emplace(backend);
        }
        void TearDown() override {
            if (GetParam() == Storage::Vulkan) {
                EXPECT_TRUE(shutdown_gpu_backend(GpuBackend::Vulkan));
                for (const auto& message : internal::vulkan_validation_messages_for_testing())
                    ADD_FAILURE() << message;
            }
        }
        Tensor onDevice(const Tensor& t) { return device == Device::CPU ? t : t.gpu(); }
        Tensor picture(int c, int h, int w) {
            std::vector<float> values(size_t(c) * h * w);
            for (size_t i = 0; i < values.size(); ++i)
                values[i] = float(int(i % 43) - 5) / 37.f;
            return Tensor::from_vector(values, {size_t(c), size_t(h), size_t(w)}, Device::CPU);
        }
        void close(const Tensor& actual, const Tensor& expected, float tolerance = 3e-5f) {
            const auto a = actual.to_vector(), b = expected.to_vector();
            ASSERT_EQ(a.size(), b.size());
            float error = 0;
            for (size_t i = 0; i < a.size(); ++i) {
                ASSERT_TRUE(std::isfinite(a[i])) << i;
                error = std::max(error, std::abs(a[i] - b[i]));
            }
            EXPECT_LE(error, tolerance);
        }
    };
    UndistortParams camera(int w, int h) {
        UndistortParams p{};
        p.src_width = p.dst_width = w;
        p.src_height = p.dst_height = h;
        p.src_fx = p.dst_fx = float(w);
        p.src_fy = p.dst_fy = float(h);
        p.src_cx = p.dst_cx = w * 0.5f;
        p.src_cy = p.dst_cy = h * 0.5f;
        p.num_distortion = 10;
        const float d[] = {0.12f, -0.03f, 0.004f, 0.002f, -0.003f, 0.001f, 0.002f, -0.001f, 0.001f, -0.002f};
        std::copy_n(d, 10, p.distortion);
        return p;
    }
    TEST_P(TensorImageOps, UndistortRgbAndMaskModelsStridesAndOffsets) {
        for (const auto size : {std::pair{1, 1}, std::pair{19, 7}})
            for (const auto model : {CameraModelType::PINHOLE, CameraModelType::FISHEYE, CameraModelType::THIN_PRISM_FISHEYE}) {
                const int w = size.first, h = size.second;
                auto p = camera(w, h);
                p.model_type = model;
                auto storage = picture(4, h + 2, w + 3);
                const auto host = storage.slice(0, 1, 4).slice(1, 1, h + 1).slice(2, 1, w + 1);
                const auto input = onDevice(storage).slice(0, 1, 4).slice(1, 1, h + 1).slice(2, 1, w + 1);
                close(internal::undistort_image_tensor(input, p, false), undistort_image(host, p, nullptr), 1.5e-4f);
                close(internal::undistort_image_tensor(input.slice(0, 0, 1).squeeze(0), p, true),
                      undistort_mask(host.slice(0, 0, 1).squeeze(0), p, nullptr), 1.5e-4f);
            }
    }
    TEST_P(TensorImageOps, ResizeDepthAndNormalInvalidSamplesStridesAndOffsets) {
        auto storage = picture(4, 7, 23);
        storage.ptr<float>()[23 * 7 + 23 * 2 + 4] = std::numeric_limits<float>::quiet_NaN();
        storage.ptr<float>()[23 * 7 + 23 * 2 + 5] = std::numeric_limits<float>::infinity();
        const auto host = storage.slice(0, 1, 4).slice(1, 1, 6).slice(2, 1, 20);
        const auto input = onDevice(storage).slice(0, 1, 4).slice(1, 1, 6).slice(2, 1, 20);
        for (const auto size : {std::pair{1, 1}, std::pair{17, 13}, std::pair{3, 2}}) {
            close(internal::resize_image_prior_tensor(input, size.second, size.first, true), resize_normal_prior(host, size.second, size.first));
            close(internal::resize_image_prior_tensor(input.slice(0, 0, 1).squeeze(0), size.second, size.first, false),
                  resize_depth_prior(host.slice(0, 0, 1).squeeze(0), size.second, size.first));
        }
    }
    TEST_P(TensorImageOps, EnvironmentPerspectivePanoramaBandsStridesAndOffsets) {
        for (const auto size : {std::pair{1, 1}, std::pair{19, 7}})
            for (const bool panorama : {false, true}) {
                const int w = size.first, h = size.second;
                const auto storage = picture(4, h + 2, w + 3), env_storage = picture(5, 11, 3);
                const auto host = storage.slice(0, 1, 4).slice(1, 1, h + 1).slice(2, 1, w + 1);
                const auto alpha = host.slice(0, 0, 1).squeeze(0).clamp(0, 1);
                const auto env = env_storage.slice(0, 1, 4).slice(1, 1, 10);
                EnvironmentCompositeParams p{};
                p.rotation[0] = p.rotation[4] = p.rotation[8] = 1;
                p.full_width = p.band_width = w;
                p.full_height = h + 4;
                p.band_height = h;
                p.y_offset = 2;
                p.focal_x = 13;
                p.focal_y = 17;
                p.center_x = w * 0.5f;
                p.center_y = p.full_height * 0.5f;
                p.equirect_view = panorama;
                p.env_width = 9;
                p.env_height = 3;
                p.exposure_factor = 1.3f;
                p.env_rotation_radians = 0.9f;
                const auto actual = environment_composite(onDevice(storage).slice(0, 1, 4).slice(1, 1, h + 1).slice(2, 1, w + 1),
                                                          onDevice(alpha), onDevice(env_storage).slice(0, 1, 4).slice(1, 1, 10), p)
                                        .to_vector_uint8();
                const auto expected = environment_composite(host, alpha, env, p).to_vector_uint8();
                ASSERT_EQ(actual.size(), expected.size());
                for (size_t i = 0; i < actual.size(); ++i)
                    EXPECT_LE(std::abs(int(actual[i]) - int(expected[i])), 1) << i;
            }
    }
    TEST_P(TensorImageOps, PpispReferenceBandsNegativeRgbAndOffsets) {
        for (const auto size : {std::pair{1, 1}, std::pair{19, 7}}) {
            const int w = size.first, h = size.second;
            const auto storage = picture(4, h + 2, w + 3);
            const auto host = storage.slice(0, 1, 4).slice(1, 1, h + 1).slice(2, 1, w + 1);
            const auto input = onDevice(storage).slice(0, 1, 4).slice(1, 1, h + 1).slice(2, 1, w + 1);
            PpispParams p;
            p.full_height = h + 4;
            p.y_offset = 2;
            p.exposure_factor = 1.31f;
            for (int c = 0; c < 3; ++c) {
                p.vignetting[c * 5] = 0.01f;
                p.vignetting[c * 5 + 1] = -0.03f;
                p.vignetting[c * 5 + 2] = -0.3f;
                p.crf[c * 5] = 0.8f;
                p.crf[c * 5 + 1] = 1.2f;
                p.crf[c * 5 + 2] = 1.1f;
                p.crf[c * 5 + 3] = 0.4f;
                p.crf[c * 5 + 4] = 1.2f * 0.4f / std::fma(1.2f - 0.8f, 0.4f, 0.8f);
            }
            close(ppisp_apply(input, p), ppisp_apply(host, p));
        }
    }
    TEST_P(TensorImageOps, GridStrideCoversMoreThan65535Groups) {
        if (device == Device::CPU)
            GTEST_SKIP() << "GPU dispatch limit";
        constexpr int width = 65536 * 256 + 17;
        auto p = camera(1, 1);
        p.dst_width = width;
        p.dst_fx = float(width);
        p.src_fx = 0;
        p.src_fy = 0;
        auto undistorted = internal::undistort_image_tensor(Tensor::full({1, 1}, 0.37f, device), p, true);
        EXPECT_NEAR(undistorted.slice(1, width - 1, width).item(), 0.37f, 1e-6f);
        undistorted = {};
        auto undistorted_rgb = internal::undistort_image_tensor(Tensor::full({3, 1, 1}, 0.37f, device), p, false);
        close(undistorted_rgb.slice(2, width - 1, width), Tensor::full({3, 1, 1}, 0.37f, Device::CPU));
        undistorted_rgb = {};
        auto prior = resize_depth_prior(Tensor::full({1, 1}, 0.7f, device), 1, width);
        EXPECT_NEAR(prior.slice(1, width - 1, width).item(), 0.7f, 1e-6f);
        prior = {};
        auto normals = resize_normal_prior(Tensor::full({3, 1, 1}, 1.f, device), 1, width);
        close(normals.slice(2, width - 1, width), Tensor::full({3, 1, 1}, 1 / std::sqrt(3.f), Device::CPU));
        normals = {};
        auto rgb = Tensor::full({3, 1, width}, 0.2f, device);
        auto corrected = ppisp_apply(rgb, {});
        close(corrected.slice(2, width - 1, width), ppisp_apply(Tensor::full({3, 1, 1}, 0.2f, Device::CPU), {}));
        corrected = {};
        rgb = {};
        constexpr int env_width = 65536 * 256 * 4 + 1;
        EnvironmentCompositeParams e{};
        e.rotation[0] = e.rotation[4] = e.rotation[8] = 1;
        e.full_width = e.band_width = env_width;
        e.full_height = e.band_height = 1;
        e.env_width = e.env_height = 1;
        e.focal_x = float(env_width);
        e.focal_y = 1;
        e.center_x = env_width * 0.5f;
        e.center_y = 0.5f;
        for (const bool panorama : {false, true}) {
            e.equirect_view = panorama;
            auto composed = environment_composite(Tensor::full({3, 1, env_width}, 0.2f, device), Tensor::ones({1, env_width}, device), Tensor::ones({1, 1, 3}, device), e);
            EXPECT_EQ(composed.slice(1, env_width - 1, env_width).to_vector_uint8(), (std::vector<uint8_t>{51, 51, 51}));
        }
    }
    TEST_P(TensorImageOps, RejectsMalformedInputs) {
        EXPECT_THROW(ppisp_apply(Tensor{}, {}), std::exception);
        const auto rgb = onDevice(picture(3, 7, 19));
        PpispParams p;
        p.y_offset = 1;
        EXPECT_THROW(ppisp_apply(rgb, p), std::exception);
        EXPECT_THROW(internal::resize_image_prior_tensor(rgb, 0, 3, true), std::exception);
        auto u = camera(18, 7);
        EXPECT_THROW(internal::undistort_image_tensor(rgb, u, false), std::exception);
        EXPECT_THROW(environment_composite(rgb, Tensor{}, Tensor{}, {}), std::exception);
    }
    class TensorImageStreams : public lfs::test::CudaBackendTest {};

    TEST_F(TensorImageStreams, OrdersIndependentCudaStreamsAndTemporaryInputs) {
        cudaStream_t producer = nullptr, consumer = nullptr;
        ASSERT_EQ(cudaStreamCreateWithFlags(&producer, cudaStreamNonBlocking), cudaSuccess);
        ASSERT_EQ(cudaStreamCreateWithFlags(&consumer, cudaStreamNonBlocking), cudaSuccess);
        {
            Tensor corrected, composite, undistorted;
            {
                Tensor rgb, alpha, env;
                {
                    const CUDAStreamGuard guard(producer);
                    rgb = Tensor::full({3, 17, 19}, 0.2f, Device::GPU);
                    alpha = Tensor::ones({17, 19}, Device::GPU);
                    env = Tensor::ones({1, 1, 3}, Device::GPU);
                }
                {
                    const CUDAStreamGuard guard(consumer);
                    corrected = ppisp_apply(rgb, {});
                    EnvironmentCompositeParams p{};
                    p.rotation[0] = p.rotation[4] = p.rotation[8] = 1;
                    p.full_width = p.band_width = 19;
                    p.full_height = p.band_height = 17;
                    p.env_width = p.env_height = 1;
                    p.focal_x = p.focal_y = 20;
                    composite = environment_composite(rgb, alpha, env, p);
                    auto u = camera(19, 17);
                    u.num_distortion = 0;
                    undistorted = internal::undistort_image_tensor(rgb, u, false);
                }
            }
            const auto expected = ppisp_apply(Tensor::full({3, 17, 19}, 0.2f, Device::CPU), {}).to_vector();
            const auto actual = corrected.to_vector();
            ASSERT_EQ(actual.size(), expected.size());
            for (size_t i = 0; i < actual.size(); ++i)
                EXPECT_NEAR(actual[i], expected[i], 1e-6f);
            for (float value : undistorted.to_vector())
                EXPECT_NEAR(value, 0.2f, 1e-6f);
            for (uint8_t value : composite.to_vector_uint8())
                EXPECT_EQ(value, 51);
        }
        internal::backend_ops(GpuBackend::CUDA).release_stream(internal::ExecContext{producer});
        internal::backend_ops(GpuBackend::CUDA).release_stream(internal::ExecContext{consumer});
        EXPECT_EQ(cudaStreamDestroy(producer), cudaSuccess);
        EXPECT_EQ(cudaStreamDestroy(consumer), cudaSuccess);
    }

    INSTANTIATE_TEST_SUITE_P(Backends, TensorImageOps, ::testing::Values(Storage::CPU, Storage::CUDA, Storage::Vulkan),
                             [](const ::testing::TestParamInfo<Storage>& info) { return info.param == Storage::CPU ? "CPU" : info.param == Storage::CUDA ? "CUDA"
                                                                                                                                                         : "Vulkan"; });
} // namespace
