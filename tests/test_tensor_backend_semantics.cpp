/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

// Edge-case semantics every tensor backend must share: scalars, empty
// operands, C pow, unsigned and half dtypes, saturating casts and fault
// reporting. Each case runs on the CPU and on every available GPU backend.

#include "core/nn/ops.hpp"
#include "core/tensor.hpp"
#include "core/tensor_backend.hpp"
#include "core/tensor_upload.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <format>
#include <iostream>
#include <limits>
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace {

    using lfs::core::DataType;
    using lfs::core::Device;
    using lfs::core::GpuBackend;
    using lfs::core::Tensor;
    using lfs::core::TensorShape;

    constexpr float kInf = std::numeric_limits<float>::infinity();
    constexpr float kNan = std::numeric_limits<float>::quiet_NaN();

    TEST(TensorProcessConfigurationDeathTest, BackendAndExecutionOptionsSurviveReexec) {
        const auto configuration = [] {
            const auto options = lfs::core::tensor_backend_options();
            return std::format("tensor-config: {} {} {} {}\n",
                               lfs::core::gpu_backend_name(lfs::core::configured_gpu_backend()),
                               options.vulkan_validation, options.force_fp32_half,
                               options.force_no_atomic_float);
        };
        // The matcher is evaluated by the parent. Recompute the configuration
        // inside the re-executed child so a lost selector cannot pass silently
        // on machines where the harness's default CUDA backend is also built.
        const auto expected = configuration();
        const auto style = GTEST_FLAG_GET(death_test_style);
        GTEST_FLAG_SET(death_test_style, "threadsafe");
        EXPECT_EXIT(
            {
                std::cerr << configuration() << std::flush;
                std::_Exit(0);
            },
            testing::ExitedWithCode(0), expected);
        GTEST_FLAG_SET(death_test_style, style);
    }

    struct Target {
        const char* name;
        std::optional<GpuBackend> backend;
    };

    class TensorBackendSemantics : public testing::TestWithParam<Target> {
    protected:
        void SetUp() override {
            if (const auto backend = GetParam().backend) {
                if (!lfs::core::gpu_backend_available(*backend))
                    GTEST_SKIP() << GetParam().name << " device unavailable";
                scope_.emplace(*backend);
            }
        }

        Device device() const { return GetParam().backend ? Device::GPU : Device::CPU; }

        // A tensor of `dtype` holding `values`, on the backend under test.
        template <typename T>
        Tensor make(const DataType dtype, const std::vector<size_t>& shape, const std::vector<T>& values) const {
            Tensor cpu = Tensor::empty(TensorShape(shape), Device::CPU, dtype);
            EXPECT_EQ(cpu.bytes(), values.size() * sizeof(T));
            if (!values.empty())
                std::memcpy(cpu.data_ptr(), values.data(), cpu.bytes());
            return device() == Device::CPU ? cpu : cpu.to(Device::GPU);
        }

        template <typename T>
        static std::vector<T> host(const Tensor& tensor) {
            const Tensor cpu = tensor.cpu().contiguous();
            std::vector<T> values(cpu.numel());
            if (!values.empty())
                std::memcpy(values.data(), cpu.data_ptr(), cpu.bytes());
            return values;
        }

        std::optional<lfs::core::GpuBackendScope> scope_;
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

    TEST_P(TensorBackendSemantics, SoftmaxNonfiniteRows) {
        const std::vector<float> rows{
            0.f, kInf, kInf, 0.f, kInf, kInf,
            0.f, -kInf, -kInf, 0.f, -kInf, -kInf,
            0.f, kNan, kNan, 0.f, -kInf, kNan, kNan, kNan};
        const std::vector<float> expected{
            kNan, kNan, kNan, kNan, kNan, kNan,
            1.f, 0.f, 0.f, 1.f, 0.f, 0.f,
            kNan, kNan, kNan, kNan, kNan, kNan, kNan, kNan};
        for (const auto dtype : {DataType::Float32, DataType::Float16}) {
            SCOPED_TRACE(static_cast<int>(dtype));
            const auto values = make<float>(DataType::Float32, {10, 2}, rows).to(dtype);
            expect_floats(host<float>(lfs::core::nn::softmax(values).to(DataType::Float32)), expected);
            const auto zeros = make<float>(DataType::Float32, {10, 2}, std::vector<float>(20, 0.f)).to(dtype);
            expect_floats(host<float>(lfs::core::nn::softmax(zeros, &values).to(DataType::Float32)), expected);
            const auto mask = make<float>(DataType::Float32, {1, 2}, {-kInf, -kInf}).to(dtype);
            expect_floats(host<float>(lfs::core::nn::softmax(zeros, &mask).to(DataType::Float32)),
                          std::vector<float>(20, 0.f));
        }
    }

    TEST_P(TensorBackendSemantics, ScalarNonzeroHasOneEmptyRow) {
        const Tensor found = make<float>(DataType::Float32, {}, {2.5f}).nonzero();
        EXPECT_EQ(found.shape(), TensorShape({1, 0}));
        EXPECT_EQ(make<float>(DataType::Float32, {}, {0.f}).nonzero().shape(), TensorShape({0, 0}));
    }

    TEST_P(TensorBackendSemantics, EmptyContractionIsZero) {
        const Tensor a = make<float>(DataType::Float32, {2, 0}, {}), b = make<float>(DataType::Float32, {0, 3}, {});
        EXPECT_EQ(host<float>(a.mm(b)), std::vector<float>(6, 0.f));
        const Tensor batched = make<float>(DataType::Float32, {1, 2, 0}, {}).bmm(make<float>(DataType::Float32, {1, 0, 3}, {}));
        EXPECT_EQ(batched.shape(), TensorShape({1, 2, 3}));
        EXPECT_EQ(host<float>(batched), std::vector<float>(6, 0.f));
    }

    // Fails if a batch beyond one launch's z dimension (65535 on CUDA and typical Vulkan devices) is rejected or
    // its tail batches are computed with the wrong operands.
    TEST_P(TensorBackendSemantics, BatchedMatmulHandlesBatchesBeyondOneLaunch) {
        constexpr size_t batch = 70001, m = 3, k = 3, n = 2;
        std::vector<float> a(batch * m * k), b(batch * k * n);
        for (size_t i = 0; i < a.size(); ++i)
            a[i] = static_cast<float>(i % 13) - 6.0f;
        for (size_t i = 0; i < b.size(); ++i)
            b[i] = static_cast<float>(i % 7) * 0.5f;
        std::vector<float> want(batch * m * n, 0.0f);
        for (size_t p = 0; p < batch; ++p)
            for (size_t r = 0; r < m; ++r)
                for (size_t c = 0; c < n; ++c)
                    for (size_t j = 0; j < k; ++j)
                        want[(p * m + r) * n + c] += a[(p * m + r) * k + j] * b[(p * k + j) * n + c];
        const Tensor product = make<float>(DataType::Float32, {batch, m, k}, a).bmm(make<float>(DataType::Float32, {batch, k, n}, b));
        EXPECT_EQ(product.shape(), TensorShape({batch, m, n}));
        expect_floats(host<float>(product), want);
    }

    TEST_P(TensorBackendSemantics, FloatPowFollowsC) {
        const Tensor base = make<float>(DataType::Float32, {7}, {-2.f, -3.f, 0.f, -kInf, 0.f, kNan, -8.f});
        const Tensor exponent = make<float>(DataType::Float32, {7}, {3.f, -1.f, 0.f, 2.5f, -1.f, 0.f, 1.f / 3.f});
        expect_floats(host<float>(base.pow(exponent)), {-8.f, -1.f / 3.f, 1.f, kInf, kInf, 1.f, kNan});
        expect_floats(host<float>(base.pow(3.f)), {-8.f, -27.f, 0.f, -kInf, 0.f, kNan, -512.f});
        // Broadcast operands take a separate kernel.
        const Tensor cube = make<float>(DataType::Float32, {1}, {3.f});
        expect_floats(host<float>(base.pow(cube)), {-8.f, -27.f, 0.f, -kInf, 0.f, kNan, -512.f});
    }

    TEST_P(TensorBackendSemantics, FloatScalarMinimumAndMaximum) {
        const Tensor x = make<float>(DataType::Float32, {4}, {-2.f, 0.25f, 3.f, 0.5f});
        expect_floats(host<float>(x.minimum(0.5f)), {-2.f, 0.25f, 0.5f, 0.5f});
        expect_floats(host<float>(x.maximum(0.5f)), {0.5f, 0.5f, 3.f, 0.5f});
    }

    TEST_P(TensorBackendSemantics, IntegerPowBroadcasts) {
        const Tensor i64 = make<int64_t>(DataType::Int64, {2}, {3, 7}).pow(make<int64_t>(DataType::Int64, {1}, {2}));
        EXPECT_EQ(host<int64_t>(i64), (std::vector<int64_t>{9, 49}));
        const Tensor i32 = make<int32_t>(DataType::Int32, {3}, {-2, -3, 3}).pow(make<int32_t>(DataType::Int32, {1}, {19}));
        EXPECT_EQ(host<int32_t>(i32), (std::vector<int32_t>{-524288, -1162261467, 1162261467}));
    }

    TEST_P(TensorBackendSemantics, UInt32ArithmeticComparesAndSelects) {
        const Tensor a = make<uint32_t>(DataType::UInt32, {4}, {1, 22, 3, 0x80000000u});
        EXPECT_EQ(host<uint32_t>(a.add(make<uint32_t>(DataType::UInt32, {4}, {1, 22, 3, 1}))),
                  (std::vector<uint32_t>{2, 44, 6, 0x80000001u}));
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

    TEST_P(TensorBackendSemantics, TakeReadsStridedIndices) {
        const Tensor values = make<float>(DataType::Float32, {6}, {10, 11, 12, 13, 14, 15});
        const Tensor strided = make<int32_t>(DataType::Int32, {4}, {1, 2, 3, 4}).reshape(TensorShape({2, 2})).slice(1, 0, 1).squeeze(1);
        EXPECT_EQ(host<float>(values.take(strided)), (std::vector<float>{11.f, 13.f}));
    }

    TEST_P(TensorBackendSemantics, HalfKeepsSubnormals) {
        const Tensor a = make<float>(DataType::Float32, {2}, {0.001f, -0.0078125f}).to(DataType::Float16);
        const Tensor b = make<float>(DataType::Float32, {2}, {0.01f, 0.00390625f}).to(DataType::Float16);
        const auto product = host<float>(a.mul(b).to(DataType::Float32));
        EXPECT_NEAR(product[0], 1.0e-5f, 1e-7f);
        EXPECT_FLOAT_EQ(product[1], -3.0517578125e-5f);
    }

    TEST_P(TensorBackendSemantics, CumsumStaysAccurateOnLongLines) {
        const Tensor sums = make<float>(DataType::Float32, {40000}, std::vector<float>(40000, 0.1f)).cumsum(0);
        EXPECT_NEAR(host<float>(sums).back(), 4000.f, 4000.f * 1e-6f);
    }

    TEST_P(TensorBackendSemantics, ScalarBoolCastKeepsRank) {
        const Tensor flag = make<int32_t>(DataType::Int32, {}, {5}).to(DataType::Bool);
        EXPECT_EQ(flag.ndim(), 0u);
        EXPECT_EQ(host<uint8_t>(flag), (std::vector<uint8_t>{1}));
    }

    TEST_P(TensorBackendSemantics, InverseTrigPropagatesNan) {
        const Tensor x = make<float>(DataType::Float32, {2}, {kNan, 0.5f});
        expect_floats(host<float>(x.asin()), {kNan, std::asin(0.5f)});
        expect_floats(host<float>(x.acos()), {kNan, std::acos(0.5f)});
    }

    TEST_P(TensorBackendSemantics, EmptyScalarReductions) {
        const Tensor empty = make<float>(DataType::Float32, {0}, {});
        EXPECT_TRUE(std::isnan(empty.mean_scalar()));
        EXPECT_ANY_THROW((void)empty.max_scalar());
        EXPECT_ANY_THROW((void)empty.min_scalar());
    }

    TEST_P(TensorBackendSemantics, FloatToIntegerCastsSaturate) {
        const Tensor x = make<float>(DataType::Float32, {6}, {kNan, kInf, -kInf, 3.7f, -3.7f, 5e9f});
        EXPECT_EQ(host<int32_t>(x.to(DataType::Int32)),
                  (std::vector<int32_t>{0, INT32_MAX, INT32_MIN, 3, -3, INT32_MAX}));
        EXPECT_EQ(host<int64_t>(x.to(DataType::Int64)),
                  (std::vector<int64_t>{0, INT64_MAX, INT64_MIN, 3, -3, 5000000000}));
        EXPECT_EQ(host<uint32_t>(x.to(DataType::UInt32)),
                  (std::vector<uint32_t>{0, UINT32_MAX, 0, 3, 0, UINT32_MAX}));
    }

    TEST_P(TensorBackendSemantics, CopyIntoExpandedViewThrows) {
        Tensor view = make<float>(DataType::Float32, {1}, {7.f}).expand(TensorShape({3}));
        EXPECT_ANY_THROW(view.copy_(make<float>(DataType::Float32, {3}, {1.f, 2.f, 3.f})));
    }

    TEST_P(TensorBackendSemantics, IndexFaultOnEmptyTensorSurfacesAtItsReadback) {
        Tensor empty = make<float>(DataType::Float32, {7, 0}, {});
        EXPECT_ANY_THROW({
            empty.index_fill_(0, make<int64_t>(DataType::Int64, {1}, {9}), 1.f);
            (void)empty.cpu();
        });
        // The fault is consumed: unrelated work runs clean.
        EXPECT_FLOAT_EQ(make<float>(DataType::Float32, {3}, {1.f, 1.f, 1.f}).sum_scalar(), 3.f);
    }

    TEST_P(TensorBackendSemantics, InBatchUploadOrdersAfterPendingReaders) {
        if (!GetParam().backend)
            GTEST_SKIP() << "Uploads target a GPU backend";
        Tensor destination = Tensor::full({4099}, 3.0f, Device::GPU, DataType::Float32);
        // A pending reader keeps the destination busy, so the bytes go through
        // staging and a queued copy that must run after this read.
        Tensor previous = Tensor::zeros({4099}, Device::GPU, DataType::Float32);
        previous.copy_from(destination);
        std::vector<float> values(4099, 7.0f);
        lfs::core::TensorUpload upload;
        upload.enqueue_in_batch(destination, std::as_bytes(std::span(values)));
        values.assign(values.size(), 0.0f);
        upload.wait();
        EXPECT_TRUE(upload.poll());
        EXPECT_EQ(destination.cpu().to_vector(), std::vector<float>(4099, 7.0f));
        EXPECT_EQ(previous.cpu().to_vector(), std::vector<float>(4099, 3.0f));
    }

    INSTANTIATE_TEST_SUITE_P(Backends, TensorBackendSemantics,
                             testing::Values(Target{"CPU", std::nullopt}, Target{"Metal", GpuBackend::Metal},
                                             Target{"Vulkan", GpuBackend::Vulkan}, Target{"CUDA", GpuBackend::CUDA}),
                             [](const auto& info) { return std::string(info.param.name); });

} // namespace
