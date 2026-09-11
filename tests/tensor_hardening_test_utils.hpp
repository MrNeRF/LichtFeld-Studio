/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/tensor.hpp"
#include "core/tensor/internal/memory_pool.hpp"

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <torch/torch.h>

namespace tensor_hardening {

    using lfs::core::DataType;
    using lfs::core::Device;
    using lfs::core::Tensor;
    using lfs::core::TensorShape;

    cudaError_t launch_delay_kernel(cudaStream_t stream, uint64_t cycles);

    inline bool has_cuda_device() {
        int count = 0;
        return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
    }

    class CudaTest : public ::testing::Test {
    protected:
        void SetUp() override {
            if (!has_cuda_device()) {
                GTEST_SKIP() << "CUDA device required";
            }
            ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
        }
    };

    inline void destroy_stream_safely(cudaStream_t stream) {
        if (stream == nullptr) {
            return;
        }
        lfs::core::CudaMemoryPool::instance().release_stream(stream);
        EXPECT_EQ(cudaStreamDestroy(stream), cudaSuccess);
    }

    inline std::vector<int64_t> torch_shape(const Tensor& tensor) {
        std::vector<int64_t> result;
        result.reserve(tensor.shape().rank());
        for (const size_t dim : tensor.shape().dims()) {
            result.push_back(static_cast<int64_t>(dim));
        }
        return result;
    }

    inline void expect_shape_matches(const Tensor& actual, const torch::Tensor& expected,
                                     const std::string& context) {
        ASSERT_EQ(actual.shape().rank(), static_cast<size_t>(expected.dim()))
            << context << ": rank mismatch; ours=" << actual.shape().str()
            << ", torch=" << expected.sizes();
        for (size_t dim = 0; dim < actual.shape().rank(); ++dim) {
            EXPECT_EQ(actual.shape()[dim], static_cast<size_t>(expected.size(dim)))
                << context << ": dimension " << dim << " mismatch; ours="
                << actual.shape().str() << ", torch=" << expected.sizes();
        }
    }

    inline std::vector<float> lfs_values_as_float(const Tensor& input) {
        Tensor materialized = input.is_contiguous() ? input : input.contiguous();
        if (materialized.dtype() != DataType::Float32) {
            materialized = materialized.to(DataType::Float32);
        }
        return materialized.cpu().to_vector();
    }

    inline std::vector<float> torch_values_as_float(const torch::Tensor& input) {
        const auto materialized = input.detach().to(torch::kCPU).to(torch::kFloat32).contiguous().flatten();
        const auto* values = materialized.data_ptr<float>();
        return {values, values + materialized.numel()};
    }

    inline void expect_float_values_match(const Tensor& actual, const torch::Tensor& expected,
                                          const std::string& context,
                                          const float rtol = 1e-5f,
                                          const float atol = 1e-6f) {
        expect_shape_matches(actual, expected, context);
        const auto ours = lfs_values_as_float(actual);
        const auto torch_values = torch_values_as_float(expected);
        ASSERT_EQ(ours.size(), torch_values.size()) << context << ": element count mismatch";

        size_t mismatch_count = 0;
        std::ostringstream mismatch_sample;
        for (size_t i = 0; i < ours.size(); ++i) {
            const float lhs = ours[i];
            const float rhs = torch_values[i];
            bool matches = false;
            if (std::isnan(rhs)) {
                matches = std::isnan(lhs);
            } else if (std::isinf(rhs)) {
                matches = std::isinf(lhs) && std::signbit(lhs) == std::signbit(rhs);
            } else {
                matches = std::abs(lhs - rhs) <= atol + rtol * std::abs(rhs);
            }
            if (!matches) {
                if (mismatch_count < 8) {
                    mismatch_sample << "\n  index " << i << ": expected=" << rhs
                                    << ", actual=" << lhs;
                }
                ++mismatch_count;
            }
        }
        EXPECT_EQ(mismatch_count, 0u)
            << context << ": " << mismatch_count << " mismatched element(s); sample:"
            << mismatch_sample.str();
    }

    inline void expect_int_values_match(const Tensor& actual, const torch::Tensor& expected,
                                        const std::string& context) {
        expect_shape_matches(actual, expected, context);
        Tensor materialized = actual.is_contiguous() ? actual : actual.contiguous();
        const auto torch_values = expected.detach().to(torch::kCPU).to(torch::kInt64).contiguous().flatten();
        ASSERT_EQ(materialized.numel(), static_cast<size_t>(torch_values.numel()))
            << context << ": element count mismatch";

        std::vector<int64_t> ours;
        if (materialized.dtype() == DataType::Int64) {
            ours = materialized.cpu().to_vector_int64();
        } else {
            const auto ints = materialized.cpu().to_vector_int();
            ours.assign(ints.begin(), ints.end());
        }
        const auto* reference = torch_values.data_ptr<int64_t>();
        size_t mismatch_count = 0;
        std::ostringstream mismatch_sample;
        for (size_t i = 0; i < ours.size(); ++i) {
            if (ours[i] != reference[i]) {
                if (mismatch_count < 8) {
                    mismatch_sample << "\n  index " << i << ": expected=" << reference[i]
                                    << ", actual=" << ours[i];
                }
                ++mismatch_count;
            }
        }
        EXPECT_EQ(mismatch_count, 0u)
            << context << ": " << mismatch_count << " mismatched element(s); sample:"
            << mismatch_sample.str();
    }

    inline void expect_bool_values_match(const Tensor& actual, const torch::Tensor& expected,
                                         const std::string& context) {
        expect_shape_matches(actual, expected, context);
        Tensor materialized = actual.is_contiguous() ? actual : actual.contiguous();
        const auto ours = materialized.cpu().to_vector_bool();
        const auto torch_values = expected.detach().to(torch::kCPU).to(torch::kBool).contiguous().flatten();
        ASSERT_EQ(ours.size(), static_cast<size_t>(torch_values.numel()))
            << context << ": element count mismatch";
        const auto* reference = torch_values.data_ptr<bool>();
        size_t mismatch_count = 0;
        std::ostringstream mismatch_sample;
        for (size_t i = 0; i < ours.size(); ++i) {
            if (ours[i] != reference[i]) {
                if (mismatch_count < 8) {
                    mismatch_sample << "\n  index " << i << ": expected=" << reference[i]
                                    << ", actual=" << ours[i];
                }
                ++mismatch_count;
            }
        }
        EXPECT_EQ(mismatch_count, 0u)
            << context << ": " << mismatch_count << " mismatched element(s); sample:"
            << mismatch_sample.str();
    }

    inline Tensor lfs_float_tensor(const std::vector<float>& values,
                                   const std::initializer_list<size_t> shape,
                                   const Device device) {
        return Tensor::from_vector(values, TensorShape(shape), device);
    }

    inline Tensor lfs_int_tensor(const std::vector<int>& values,
                                 const std::initializer_list<size_t> shape,
                                 const Device device) {
        return Tensor::from_vector(values, TensorShape(shape), device);
    }

    inline Tensor lfs_bool_tensor(const std::vector<bool>& values,
                                  const std::initializer_list<size_t> shape,
                                  const Device device) {
        return Tensor::from_vector(values, TensorShape(shape), device);
    }

    // A stream blocked behind a short GPU delay. It widens ordering windows
    // without a blocking host callback, which can serialize CUDA API calls.
    class GateStream {
    public:
        GateStream() {
            EXPECT_EQ(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking), cudaSuccess);
            EXPECT_EQ(cudaStreamCreateWithFlags(&gate_holder_, cudaStreamNonBlocking), cudaSuccess);
            EXPECT_EQ(cudaEventCreateWithFlags(&gate_, cudaEventDisableTiming), cudaSuccess);
        }

        GateStream(const GateStream&) = delete;
        GateStream& operator=(const GateStream&) = delete;

        ~GateStream() {
            if (stream_ != nullptr) {
                lfs::core::CudaMemoryPool::instance().release_stream(stream_);
                cudaStreamDestroy(stream_);
            }
            if (gate_holder_ != nullptr) {
                cudaStreamDestroy(gate_holder_);
            }
            if (gate_ != nullptr) {
                cudaEventDestroy(gate_);
            }
        }

        void close() {
            EXPECT_EQ(launch_delay_kernel(gate_holder_, 100'000'000), cudaSuccess);
            EXPECT_EQ(cudaEventRecord(gate_, gate_holder_), cudaSuccess);
            EXPECT_EQ(cudaStreamWaitEvent(stream_, gate_, 0), cudaSuccess);
        }

        [[nodiscard]] cudaStream_t get() const { return stream_; }

    private:
        cudaStream_t stream_ = nullptr;
        cudaStream_t gate_holder_ = nullptr;
        cudaEvent_t gate_ = nullptr;
    };

} // namespace tensor_hardening
