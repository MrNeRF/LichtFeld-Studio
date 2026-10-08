/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

// Edge-case semantics the CPU and CUDA tensors share: C pow, NaN casts to
// integers, and UInt32 broadcasts, masked select and masked assignment.

#include "core/tensor.hpp"

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <string>
#include <thread>
#include <vector>

#include <cuda_runtime.h>

namespace {

    using lfs::core::DataType;
    using lfs::core::Device;
    using lfs::core::Tensor;
    using lfs::core::TensorShape;

    constexpr float kInf = std::numeric_limits<float>::infinity();
    constexpr float kNan = std::numeric_limits<float>::quiet_NaN();

    class TensorEdgeSemantics : public testing::TestWithParam<Device> {
    protected:
        // A tensor of `dtype` holding `values`, on the device under test.
        template <typename T>
        Tensor make(const DataType dtype, const std::vector<size_t>& shape, const std::vector<T>& values) const {
            Tensor cpu = Tensor::empty(TensorShape(shape), Device::CPU, dtype);
            EXPECT_EQ(cpu.bytes(), values.size() * sizeof(T));
            if (!values.empty())
                std::memcpy(cpu.data_ptr(), values.data(), cpu.bytes());
            return GetParam() == Device::CPU ? cpu : cpu.to(GetParam());
        }

        template <typename T>
        static std::vector<T> host(const Tensor& tensor) {
            const Tensor cpu = tensor.cpu().contiguous();
            std::vector<T> values(cpu.numel());
            if (!values.empty())
                std::memcpy(values.data(), cpu.data_ptr(), cpu.bytes());
            return values;
        }
    };

    void expect_floats(const std::vector<float>& got, const std::vector<float>& want) {
        ASSERT_EQ(got.size(), want.size());
        for (size_t i = 0; i < got.size(); ++i) {
            if (std::isnan(want[i]))
                EXPECT_TRUE(std::isnan(got[i])) << i << ": " << got[i];
            else
                EXPECT_FLOAT_EQ(got[i], want[i]) << i;
        }
    }

    TEST_P(TensorEdgeSemantics, FloatPowFollowsC) {
        const Tensor base = make<float>(DataType::Float32, {7}, {-2.f, -3.f, 0.f, -kInf, 0.f, kNan, -8.f});
        const Tensor exponent = make<float>(DataType::Float32, {7}, {3.f, -1.f, 0.f, 2.5f, -1.f, 0.f, 1.f / 3.f});
        expect_floats(host<float>(base.pow(exponent)), {-8.f, -1.f / 3.f, 1.f, kInf, kInf, 1.f, kNan});
        expect_floats(host<float>(base.pow(3.f)), {-8.f, -27.f, 0.f, -kInf, 0.f, kNan, -512.f});
        const Tensor cube = make<float>(DataType::Float32, {1}, {3.f});
        expect_floats(host<float>(base.pow(cube)), {-8.f, -27.f, 0.f, -kInf, 0.f, kNan, -512.f});
    }

    TEST_P(TensorEdgeSemantics, FloatToIntegerCastsSaturate) {
        const Tensor x = make<float>(DataType::Float32, {6}, {kNan, kInf, -kInf, 3.7f, -3.7f, 5e9f});
        EXPECT_EQ(host<int32_t>(x.to(DataType::Int32)),
                  (std::vector<int32_t>{0, INT32_MAX, INT32_MIN, 3, -3, INT32_MAX}));
        EXPECT_EQ(host<int64_t>(x.to(DataType::Int64)),
                  (std::vector<int64_t>{0, INT64_MAX, INT64_MIN, 3, -3, 5000000000}));
        EXPECT_EQ(host<uint32_t>(x.to(DataType::UInt32)),
                  (std::vector<uint32_t>{0, UINT32_MAX, 0, 3, 0, UINT32_MAX}));
    }

    TEST_P(TensorEdgeSemantics, InverseTrigPropagatesNan) {
        const Tensor x = make<float>(DataType::Float32, {2}, {kNan, 0.5f});
        expect_floats(host<float>(x.asin()), {kNan, std::asin(0.5f)});
        expect_floats(host<float>(x.acos()), {kNan, std::acos(0.5f)});
    }

    TEST_P(TensorEdgeSemantics, ScalarBoolCastKeepsRank) {
        const Tensor flag = make<int32_t>(DataType::Int32, {}, {5}).to(DataType::Bool);
        EXPECT_EQ(flag.ndim(), 0u);
        EXPECT_EQ(host<uint8_t>(flag), (std::vector<uint8_t>{1}));
    }

    TEST_P(TensorEdgeSemantics, EmptyScalarReductions) {
        const Tensor empty = make<float>(DataType::Float32, {0}, {});
        EXPECT_TRUE(std::isnan(empty.mean_scalar()));
        EXPECT_ANY_THROW((void)empty.max_scalar());
        EXPECT_ANY_THROW((void)empty.min_scalar());
    }

    TEST_P(TensorEdgeSemantics, CopyIntoExpandedViewThrows) {
        Tensor view = make<float>(DataType::Float32, {1}, {7.f}).expand(TensorShape({3}));
        EXPECT_ANY_THROW(view.copy_(make<float>(DataType::Float32, {3}, {1.f, 2.f, 3.f})));
    }

    TEST_P(TensorEdgeSemantics, EmptyReadbackWaitsForCudaStream) {
        if (GetParam() != Device::CUDA)
            return;

        Tensor empty = make<float>(DataType::Float32, {7, 0}, {});
        struct CallbackState {
            std::atomic<bool> started{false};
            std::atomic<bool> release{false};
            std::atomic<bool> done{false};
        } state;
        ASSERT_EQ(cudaLaunchHostFunc(
                      empty.stream(),
                      [](void* user_data) {
                          auto& callback = *static_cast<CallbackState*>(user_data);
                          callback.started.store(true, std::memory_order_release);
                          while (!callback.release.load(std::memory_order_acquire)) {
                          }
                          callback.done.store(true, std::memory_order_release);
                      },
                      &state),
                  cudaSuccess);
        while (!state.started.load(std::memory_order_acquire))
            std::this_thread::yield();

        std::thread release_callback([&state] {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            state.release.store(true, std::memory_order_release);
        });
        (void)empty.cpu();
        EXPECT_TRUE(state.done.load(std::memory_order_acquire));
        release_callback.join();
        EXPECT_EQ(cudaStreamSynchronize(empty.stream()), cudaSuccess);
    }

    TEST_P(TensorEdgeSemantics, TakeReadsStridedIndices) {
        const Tensor values = make<float>(DataType::Float32, {6}, {10, 11, 12, 13, 14, 15});
        const Tensor strided = make<int32_t>(DataType::Int32, {4}, {1, 2, 3, 4})
                                   .reshape(TensorShape({2, 2}))
                                   .slice(1, 0, 1)
                                   .squeeze(1);
        EXPECT_EQ(host<float>(values.take(strided)), (std::vector<float>{11.f, 13.f}));
    }

    TEST_P(TensorEdgeSemantics, HalfKeepsSubnormals) {
        const Tensor a = make<float>(DataType::Float32, {2}, {0.001f, -0.0078125f}).to(DataType::Float16);
        const Tensor b = make<float>(DataType::Float32, {2}, {0.01f, 0.00390625f}).to(DataType::Float16);
        const Tensor product = a.mul(b).to(DataType::Float32);
        const auto values = host<float>(product);
        EXPECT_NEAR(values[0], 1.0e-5f, 1e-7f);
        EXPECT_FLOAT_EQ(values[1], -3.0517578125e-5f);
    }

    TEST_P(TensorEdgeSemantics, CumsumStaysAccurateOnLongLines) {
        const Tensor sums = make<float>(DataType::Float32, {40000}, std::vector<float>(40000, 0.1f)).cumsum(0);
        EXPECT_NEAR(host<float>(sums).back(), 4000.f, 4000.f * 1e-6f);
    }

    TEST_P(TensorEdgeSemantics, UInt32ComparesSelectsAndAssigns) {
        const Tensor a = make<uint32_t>(DataType::UInt32, {4}, {1, 22, 3, 0x80000000u});
        EXPECT_EQ(host<uint8_t>(a.lt(make<uint32_t>(DataType::UInt32, {1}, {20}))), (std::vector<uint8_t>{1, 0, 1, 0}));
        const Tensor square = make<uint32_t>(DataType::UInt32, {2, 2}, {1, 20, 20, 3});
        EXPECT_EQ(host<uint8_t>(square.eq(make<uint32_t>(DataType::UInt32, {2, 1}, {20, 20}))),
                  (std::vector<uint8_t>{0, 1, 1, 0}));
        const Tensor mask = make<uint8_t>(DataType::Bool, {4}, {1, 0, 1, 1});
        EXPECT_EQ(host<uint32_t>(a.masked_select(mask)), (std::vector<uint32_t>{1, 3, 0x80000000u}));
        Tensor scattered = a.clone();
        scattered[mask] = make<uint32_t>(DataType::UInt32, {3}, {7, 8, 0xFFFFFFFFu});
        EXPECT_EQ(host<uint32_t>(scattered), (std::vector<uint32_t>{7, 22, 8, 0xFFFFFFFFu}));
    }

    INSTANTIATE_TEST_SUITE_P(Devices, TensorEdgeSemantics, testing::Values(Device::CPU, Device::CUDA),
                             [](const auto& info) { return std::string(info.param == Device::CPU ? "CPU" : "CUDA"); });

} // namespace
