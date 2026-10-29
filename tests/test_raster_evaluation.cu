/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "blend_bookkeeping.cuh"
#include "gaussian_evaluation.cuh"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <gtest/gtest.h>
#include <vector>

namespace {
    namespace raster = fast_lfs::rasterization;
    struct Input {
        float3 conic;
        float2 delta;
        float opacity;
    };

    __global__ void evaluate(const Input* inputs, float* outputs, int count) {
        const int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= count)
            return;
        const auto in = inputs[i];
        const auto conic = raster::kernels::base2_conic(in.conic);
        outputs[i] = raster::kernels::AlphaColumn::at(conic, in.opacity, in.delta.x).alpha(conic.z, in.delta.y);
    }

    __global__ void recover(const float2* inputs, float* outputs, int count) {
        const int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= count)
            return;
        outputs[i] = inputs[i].x * raster::kernels::rcp_approx(1.0f - inputs[i].y);
    }

    __global__ void bookkeeping(unsigned* outputs) {
        const unsigned i = threadIdx.x;
        const float value = i == 0 ? 0.0f : i * 1e-7f;
        const float stopped = raster::kernels::signed_transmittance(value, true);
        outputs[2 * i] = __float_as_uint(raster::kernels::transmittance_magnitude(stopped));
        outputs[2 * i + 1] = raster::kernels::transmittance_complete(stopped);
    }

    // Raw CUDA launches: skip without a CUDA device, like the other CUDA-only tests.
    class RasterEvaluation : public ::testing::Test {
    protected:
        void SetUp() override {
            int devices = 0;
            if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0)
                GTEST_SKIP() << "CUDA device unavailable";
        }
    };

    TEST_F(RasterEvaluation, SignedTransmittanceKeepsMagnitudeAndCompletion) {
        std::vector<unsigned> output(32 * 2);
        unsigned* device = nullptr;
        ASSERT_EQ(cudaMalloc(&device, output.size() * sizeof(unsigned)), cudaSuccess);
        bookkeeping<<<1, 32>>>(device);
        ASSERT_EQ(cudaGetLastError(), cudaSuccess);
        ASSERT_EQ(cudaMemcpy(output.data(), device, output.size() * sizeof(unsigned), cudaMemcpyDeviceToHost), cudaSuccess);
        EXPECT_EQ(cudaFree(device), cudaSuccess);
        for (unsigned i = 0; i < 32; ++i) {
            const float value = i == 0 ? 0.0f : i * 1e-7f;
            unsigned bits;
            std::memcpy(&bits, &value, sizeof(bits));
            EXPECT_EQ(output[2 * i], bits);
            EXPECT_EQ(output[2 * i + 1], 1u);
        }
    }

    TEST_F(RasterEvaluation, ColumnAlphaMatchesOpacityTimesGaussian) {
        std::vector<Input> inputs;
        for (float opacity : {0.0f, 0.002f, 1.0f / 255.0f, 0.02f, 0.5f, 0.999f, 1.0f}) {
            for (float cross : {-0.75f, 0.0f, 0.75f}) {
                for (int direction = 0; direction < 16; ++direction) {
                    const double angle = direction * 0.392699081698724;
                    const double x = std::cos(angle), y = std::sin(angle);
                    const double q = 0.5 * (x * x + y * y) + cross * x * y;
                    const double radius = std::sqrt(std::max(0.0, std::log(std::max(opacity, 1e-30f) * 255.0)) / q);
                    for (int offset = -16; offset <= 16; ++offset) {
                        const double r = radius * (1.0 + offset * 1e-6);
                        inputs.push_back({make_float3(1, cross, 1),
                                          make_float2(static_cast<float>(x * r), static_cast<float>(y * r)), opacity});
                    }
                }
            }
        }
        std::vector<float> outputs(inputs.size());
        Input* device_inputs = nullptr;
        float* device_outputs = nullptr;
        ASSERT_EQ(cudaMalloc(&device_inputs, inputs.size() * sizeof(Input)), cudaSuccess);
        ASSERT_EQ(cudaMalloc(&device_outputs, outputs.size() * sizeof(float)), cudaSuccess);
        ASSERT_EQ(cudaMemcpy(device_inputs, inputs.data(), inputs.size() * sizeof(Input), cudaMemcpyHostToDevice), cudaSuccess);
        evaluate<<<(inputs.size() + 127) / 128, 128>>>(device_inputs, device_outputs, inputs.size());
        ASSERT_EQ(cudaGetLastError(), cudaSuccess);
        ASSERT_EQ(cudaMemcpy(outputs.data(), device_outputs, outputs.size() * sizeof(float), cudaMemcpyDeviceToHost), cudaSuccess);
        EXPECT_EQ(cudaFree(device_inputs), cudaSuccess);
        EXPECT_EQ(cudaFree(device_outputs), cudaSuccess);
        for (size_t i = 0; i < inputs.size(); ++i) {
            SCOPED_TRACE(i);
            const auto in = inputs[i];
            const double x = in.delta.x, y = in.delta.y;
            const double power = -0.5 * (in.conic.x * x * x + in.conic.z * y * y) - in.conic.y * x * y;
            const double alpha = in.opacity * std::exp(power);
            EXPECT_NEAR(outputs[i], alpha, std::max(1e-9, alpha * 2e-6));
            // Acceptance is exact away from the rounding interval of the threshold.
            if (std::abs(alpha - raster::config::min_alpha_threshold) > 2e-6 * raster::config::min_alpha_threshold)
                EXPECT_EQ(outputs[i] >= raster::config::min_alpha_threshold, alpha >= raster::config::min_alpha_threshold);
        }
    }

    TEST_F(RasterEvaluation, ApproximateReciprocalRecoversTransmittance) {
        std::vector<float2> inputs;
        for (int i = 0; i <= 2048; ++i) {
            const float alpha = raster::config::max_fragment_alpha * static_cast<float>(i) / 2048.0f;
            for (float after : {1.0f, 0.1f, 1e-4f, 1e-7f})
                inputs.push_back(make_float2(after, alpha));
        }
        std::vector<float> outputs(inputs.size());
        float2* device_inputs = nullptr;
        float* device_outputs = nullptr;
        ASSERT_EQ(cudaMalloc(&device_inputs, inputs.size() * sizeof(float2)), cudaSuccess);
        ASSERT_EQ(cudaMalloc(&device_outputs, outputs.size() * sizeof(float)), cudaSuccess);
        ASSERT_EQ(cudaMemcpy(device_inputs, inputs.data(), inputs.size() * sizeof(float2), cudaMemcpyHostToDevice), cudaSuccess);
        recover<<<(inputs.size() + 127) / 128, 128>>>(device_inputs, device_outputs, inputs.size());
        ASSERT_EQ(cudaGetLastError(), cudaSuccess);
        ASSERT_EQ(cudaMemcpy(outputs.data(), device_outputs, outputs.size() * sizeof(float), cudaMemcpyDeviceToHost), cudaSuccess);
        EXPECT_EQ(cudaFree(device_inputs), cudaSuccess);
        EXPECT_EQ(cudaFree(device_outputs), cudaSuccess);
        for (size_t i = 0; i < inputs.size(); ++i) {
            SCOPED_TRACE(i);
            const double reference = static_cast<double>(inputs[i].x) / (1.0 - static_cast<double>(inputs[i].y));
            EXPECT_NEAR(outputs[i], reference, reference * 4e-7);
        }
    }
} // namespace
