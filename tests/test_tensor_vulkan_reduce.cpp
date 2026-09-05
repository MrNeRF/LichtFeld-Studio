/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/tensor.hpp"
#include "core/tensor/backend/vulkan/vk_context.hpp"
#include "core/tensor_backend.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

namespace {
    using namespace lfs::core;

    Tensor upload_vulkan(const Tensor& cpu) {
        GpuBackendScope scope(GpuBackend::Vulkan);
        return cpu.to(Device::CUDA);
    }

    // Asymmetric pattern so sums do not cancel to zero and mean stays finite.
    std::vector<float> pattern(const size_t count, const size_t period = 31) {
        std::vector<float> values(count);
        for (size_t i = 0; i < count; ++i) {
            values[i] = static_cast<float>(static_cast<int>(i % period) - 10) / 16.0f;
        }
        return values;
    }

    double tolerance(const double expected, const size_t count) {
        const double rtol = 1.0e-5 * std::max(1.0, std::log2(static_cast<double>(std::max<size_t>(2, count))));
        return 1.0e-6 + rtol * std::abs(expected);
    }

    enum class Kind { Sum,
                      Mean,
                      Max,
                      Min,
                      Prod };

    // Double-precision reference over the middle extent of an (outer, reduce, inner) view.
    std::vector<double> reference(const std::vector<float>& values, const size_t outer,
                                  const size_t reduce, const size_t inner, const Kind kind) {
        std::vector<double> result(outer * inner);
        for (size_t o = 0; o < outer; ++o) {
            for (size_t i = 0; i < inner; ++i) {
                double accumulator = kind == Kind::Prod  ? 1.0
                                     : kind == Kind::Max ? -std::numeric_limits<double>::infinity()
                                     : kind == Kind::Min ? std::numeric_limits<double>::infinity()
                                                         : 0.0;
                for (size_t r = 0; r < reduce; ++r) {
                    const double value = values[(o * reduce + r) * inner + i];
                    switch (kind) {
                    case Kind::Sum:
                    case Kind::Mean: accumulator += value; break;
                    case Kind::Max: accumulator = std::max(accumulator, value); break;
                    case Kind::Min: accumulator = std::min(accumulator, value); break;
                    case Kind::Prod: accumulator *= value; break;
                    }
                }
                result[o * inner + i] = kind == Kind::Mean ? accumulator / static_cast<double>(reduce)
                                                           : accumulator;
            }
        }
        return result;
    }

    Tensor apply(const Tensor& tensor, const Kind kind, const std::vector<int>& axes) {
        switch (kind) {
        case Kind::Sum: return axes.empty() ? tensor.sum() : tensor.sum(std::span<const int>(axes));
        case Kind::Mean: return axes.empty() ? tensor.mean() : tensor.mean(std::span<const int>(axes));
        case Kind::Max: return axes.empty() ? tensor.max() : tensor.max(std::span<const int>(axes));
        case Kind::Min: return axes.empty() ? tensor.min() : tensor.min(std::span<const int>(axes));
        case Kind::Prod: return axes.empty() ? tensor.prod() : tensor.prod(std::span<const int>(axes));
        }
        return tensor;
    }

    const char* name(const Kind kind) {
        switch (kind) {
        case Kind::Sum: return "sum";
        case Kind::Mean: return "mean";
        case Kind::Max: return "max";
        case Kind::Min: return "min";
        case Kind::Prod: return "prod";
        }
        return "?";
    }

    void expect_reduction(const Tensor& actual, const std::vector<double>& expected,
                          const size_t reduce, const std::string& label) {
        ASSERT_EQ(gpu_backend_of(actual), GpuBackend::Vulkan) << label;
        const std::vector<float> values = actual.cpu().to(DataType::Float32).to_vector();
        ASSERT_EQ(values.size(), expected.size()) << label;
        for (size_t i = 0; i < values.size(); ++i) {
            EXPECT_NEAR(values[i], expected[i], tolerance(expected[i], reduce))
                << label << " index=" << i;
        }
    }

    class TensorVulkanReduce : public testing::Test {
    protected:
        void SetUp() override {
            ASSERT_TRUE(gpu_backend_available(GpuBackend::Vulkan));
        }

        void TearDown() override {
            const auto status = shutdown_gpu_backend(GpuBackend::Vulkan);
            EXPECT_TRUE(status.has_value());
            for (const std::string& message :
                 internal::vulkan_validation_messages_for_testing()) {
                ADD_FAILURE() << message;
            }
        }
    };

    TEST_F(TensorVulkanReduce, FullReductionsCoverSingleGroupPartialAndTailPaths) {
        // Catches a wrong tail in the single-workgroup path (count <= 4096), a
        // partial stage that drops its last workgroup, and a stage 2 that reads
        // partials with the wrong width.
        constexpr std::array sizes{size_t{1}, size_t{7}, size_t{4096}, size_t{4097},
                                   size_t{8193}, size_t{1048583}};
        for (const size_t count : sizes) {
            const std::vector<float> values = pattern(count);
            const Tensor vulkan = upload_vulkan(Tensor::from_vector(values, {count}, Device::CPU));
            for (const Kind kind : {Kind::Sum, Kind::Mean, Kind::Max, Kind::Min}) {
                const Tensor result = apply(vulkan, kind, {});
                expect_reduction(result, reference(values, 1, count, 1, kind), count,
                                 std::string(name(kind)) + " n=" + std::to_string(count));
            }
        }
        std::vector<float> factors(1000);
        for (size_t i = 0; i < factors.size(); ++i) {
            factors[i] = 1.0f + static_cast<float>(static_cast<int>(i % 7) - 3) * 1.0e-4f;
        }
        const Tensor vulkan = upload_vulkan(Tensor::from_vector(factors, {factors.size()}, Device::CPU));
        const double expected = reference(factors, 1, factors.size(), 1, Kind::Prod)[0];
        EXPECT_NEAR(vulkan.prod().item(), expected, 1.0e-4 * std::abs(expected));
    }

    TEST_F(TensorVulkanReduce, ScalarFastPathsReadBackTheDeviceResult) {
        // Catches a scalar entry that returns before the readback completes or
        // scales the mean by the wrong count.
        constexpr size_t count = 100003;
        const std::vector<float> values = pattern(count);
        const Tensor vulkan = upload_vulkan(Tensor::from_vector(values, {count}, Device::CPU));
        const double sum = reference(values, 1, count, 1, Kind::Sum)[0];
        EXPECT_NEAR(vulkan.sum_scalar(), sum, tolerance(sum, count));
        EXPECT_NEAR(vulkan.mean_scalar(), sum / count, tolerance(sum / count, count));
        EXPECT_FLOAT_EQ(vulkan.max_scalar(), 20.0f / 16.0f);
        EXPECT_FLOAT_EQ(vulkan.min_scalar(), -10.0f / 16.0f);
    }

    TEST_F(TensorVulkanReduce, AxisReductionsSelectSegmentedStridedAndSplitPaths) {
        // Shapes chosen so each host path runs: a segment per workgroup, a
        // segment per thread, a plain strided loop, and the split-partials pass.
        struct Case {
            std::vector<size_t> shape;
            int axis;
        };
        const std::array cases{
            Case{{64, 300}, 1},
            Case{{64, 17}, 1},
            Case{{300, 64}, 0},
            Case{{100000, 3}, 0},
            Case{{40, 2000, 3}, 1},
        };
        for (const Case& test : cases) {
            size_t count = 1;
            for (const size_t dim : test.shape) {
                count *= dim;
            }
            const std::vector<float> values = pattern(count);
            const Tensor vulkan = upload_vulkan(
                Tensor::from_vector(values, TensorShape(test.shape), Device::CPU));
            size_t outer = 1;
            size_t inner = 1;
            for (size_t axis = 0; axis < test.shape.size(); ++axis) {
                (static_cast<int>(axis) < test.axis ? outer : static_cast<int>(axis) > test.axis ? inner
                                                                                                 : count) =
                    (static_cast<int>(axis) == test.axis) ? count : (static_cast<int>(axis) < test.axis ? outer : inner) * test.shape[axis];
            }
            const size_t reduce = test.shape[static_cast<size_t>(test.axis)];
            for (const Kind kind : {Kind::Sum, Kind::Mean, Kind::Max, Kind::Min}) {
                const Tensor result = apply(vulkan, kind, {test.axis});
                expect_reduction(result, reference(values, outer, reduce, inner, kind), reduce,
                                 std::string(name(kind)) + " axis=" + std::to_string(test.axis) +
                                     " rank=" + std::to_string(test.shape.size()));
            }
            const Tensor kept = vulkan.sum(test.axis, true);
            EXPECT_EQ(kept.shape().rank(), test.shape.size());
            EXPECT_EQ(kept.shape()[static_cast<size_t>(test.axis)], 1u);
        }
    }

    TEST_F(TensorVulkanReduce, MultiAxisReductionsUseTheRunAndGeneralPaths) {
        // Catches a contiguous-run decomposition that mislabels outer and inner
        // and a general path that decodes reduced coordinates in the wrong order.
        const std::vector<float> values = pattern(4 * 5 * 6);
        const Tensor vulkan = upload_vulkan(Tensor::from_vector(values, {4, 5, 6}, Device::CPU));
        expect_reduction(vulkan.sum({0, 1}), reference(values, 1, 20, 6, Kind::Sum), 20, "sum{0,1}");
        expect_reduction(vulkan.sum({1, 2}), reference(values, 4, 30, 1, Kind::Sum), 30, "sum{1,2}");
        std::vector<double> general(5, 0.0);
        std::vector<double> general_max(5, -std::numeric_limits<double>::infinity());
        for (size_t a = 0; a < 4; ++a) {
            for (size_t b = 0; b < 5; ++b) {
                for (size_t c = 0; c < 6; ++c) {
                    const double value = values[(a * 5 + b) * 6 + c];
                    general[b] += value;
                    general_max[b] = std::max(general_max[b], value);
                }
            }
        }
        expect_reduction(vulkan.sum({0, 2}), general, 24, "sum{0,2}");
        expect_reduction(vulkan.max({0, 2}), general_max, 24, "max{0,2}");
        expect_reduction(vulkan.sum({0, 2}, true).reshape({5}), general, 24, "sum{0,2} keepdim");
    }

    TEST_F(TensorVulkanReduce, IntegerAndBoolReductionsWidenLikeCuda) {
        // Catches an integer path that accumulates in 32 bits or writes the
        // wrong output width, and logical reductions that ignore the identity.
        std::vector<int> integers(50 * 20);
        for (size_t i = 0; i < integers.size(); ++i) {
            integers[i] = static_cast<int>(i % 97) * 1000000 - 40000000;
        }
        const Tensor vulkan = upload_vulkan(Tensor::from_vector(integers, {50, 20}, Device::CPU));
        int64_t total = 0;
        std::vector<double> rows(50, 0.0);
        int minimum = integers[0];
        int maximum = integers[0];
        for (size_t i = 0; i < integers.size(); ++i) {
            total += integers[i];
            rows[i / 20] += integers[i];
            minimum = std::min(minimum, integers[i]);
            maximum = std::max(maximum, integers[i]);
        }
        EXPECT_EQ(vulkan.sum().item<int64_t>(), total);
        EXPECT_EQ(vulkan.max().item<int>(), maximum);
        EXPECT_EQ(vulkan.min().item<int>(), minimum);
        expect_reduction(vulkan.sum(1), rows, 20, "int32 sum(1)");

        std::vector<float> flags(8 * 16, 0.0f);
        for (size_t i = 0; i < flags.size(); ++i) {
            flags[i] = (i / 16 == 3) ? 0.0f : (i % 3 == 0 ? 1.0f : 0.0f);
        }
        for (size_t i = 5 * 16; i < 6 * 16; ++i) {
            flags[i] = 1.0f;
        }
        const Tensor mask = upload_vulkan(Tensor::from_vector(flags, {8, 16}, Device::CPU)).gt(0.5f);
        ASSERT_EQ(mask.dtype(), DataType::Bool);
        EXPECT_EQ(mask.sum().item<int64_t>(), static_cast<int64_t>(std::count(flags.begin(), flags.end(), 1.0f)));
        EXPECT_EQ(mask.any().cpu().to(DataType::Float32).item(), 1.0f);
        EXPECT_EQ(mask.all().cpu().to(DataType::Float32).item(), 0.0f);
        const std::vector<float> any_rows = mask.any(1).cpu().to(DataType::Float32).to_vector();
        const std::vector<float> all_rows = mask.all(1).cpu().to(DataType::Float32).to_vector();
        ASSERT_EQ(any_rows.size(), 8u);
        for (size_t row = 0; row < 8; ++row) {
            EXPECT_EQ(any_rows[row], row == 3 ? 0.0f : 1.0f) << "row " << row;
            EXPECT_EQ(all_rows[row], row == 5 ? 1.0f : 0.0f) << "row " << row;
        }
    }

    TEST_F(TensorVulkanReduce, MaxAndMinFollowTheCudaNaNAndSignedZeroPolicy) {
        // Catches a max that uses the plain comparison, which loses NaN and
        // returns the wrong zero sign.
        const Tensor zeros = upload_vulkan(Tensor::from_vector({-0.0f, 0.0f, -0.0f}, {3}, Device::CPU));
        EXPECT_FALSE(std::signbit(zeros.max().item()));
        EXPECT_TRUE(std::signbit(zeros.min().item()));
        const float nan = std::numeric_limits<float>::quiet_NaN();
        std::vector<float> values = pattern(5000);
        values[2600] = nan;
        const Tensor poisoned = upload_vulkan(Tensor::from_vector(values, {5000}, Device::CPU));
        EXPECT_TRUE(std::isnan(poisoned.max().item()));
        EXPECT_TRUE(std::isnan(poisoned.min().item()));
        EXPECT_TRUE(std::isnan(poisoned.max_scalar()));
        EXPECT_TRUE(std::isnan(poisoned.sum_scalar()));
    }

    TEST_F(TensorVulkanReduce, FusedTransformReductionsApplyTheChainFirst) {
        // Catches a fused entry that reduces the raw input, indexes a tensor rhs
        // by the wrong element, or scales the segmented mean by the full count.
        constexpr size_t rows = 128;
        constexpr size_t columns = 1000;
        const std::vector<float> values = pattern(rows * columns);
        std::vector<float> weights(rows * columns);
        for (size_t i = 0; i < weights.size(); ++i) {
            weights[i] = 0.5f + static_cast<float>(i % 5) * 0.25f;
        }
        const Tensor x = upload_vulkan(Tensor::from_vector(values, {rows, columns}, Device::CPU));
        const Tensor y = upload_vulkan(Tensor::from_vector(weights, {rows, columns}, Device::CPU));
        std::vector<float> shifted(values.size());
        std::vector<float> weighted(values.size());
        for (size_t i = 0; i < values.size(); ++i) {
            shifted[i] = values[i] + 0.25f;
            weighted[i] = values[i] * weights[i];
        }
        expect_reduction(x.add(0.25f).sum(), reference(shifted, 1, shifted.size(), 1, Kind::Sum),
                         shifted.size(), "add.sum");
        expect_reduction(x.add(0.25f).sum(-1), reference(shifted, rows, columns, 1, Kind::Sum),
                         columns, "add.sum(-1)");
        expect_reduction(x.add(0.25f).mean(-1), reference(shifted, rows, columns, 1, Kind::Mean),
                         columns, "add.mean(-1)");
        expect_reduction(x.mul(y).sum(), reference(weighted, 1, weighted.size(), 1, Kind::Sum),
                         weighted.size(), "mul(tensor).sum");
        expect_reduction(x.mul(y).sum(-1), reference(weighted, rows, columns, 1, Kind::Sum),
                         columns, "mul(tensor).sum(-1)");
    }

    TEST_F(TensorVulkanReduce, CumsumCoversThreadAndWorkgroupScans) {
        // Catches a chunked scan that drops the carry between chunks, a strided
        // line that steps by the wrong inner size, and an int path that rounds.
        struct Case {
            std::vector<size_t> shape;
            int axis;
        };
        const std::array cases{
            Case{{3, 100000}, 1},
            Case{{100000, 5}, 1},
            Case{{2000, 3}, 0},
            Case{{70001}, 0},
        };
        for (const Case& test : cases) {
            size_t count = 1;
            for (const size_t dim : test.shape) {
                count *= dim;
            }
            size_t outer = 1;
            size_t inner = 1;
            for (size_t axis = 0; axis < test.shape.size(); ++axis) {
                if (static_cast<int>(axis) < test.axis) {
                    outer *= test.shape[axis];
                } else if (static_cast<int>(axis) > test.axis) {
                    inner *= test.shape[axis];
                }
            }
            const size_t dim = test.shape[static_cast<size_t>(test.axis)];
            const std::vector<float> values = pattern(count);
            const Tensor vulkan = upload_vulkan(
                Tensor::from_vector(values, TensorShape(test.shape), Device::CPU));
            const std::vector<float> result = vulkan.cumsum(test.axis).cpu().to_vector();
            ASSERT_EQ(result.size(), count);
            for (size_t o = 0; o < outer; ++o) {
                for (size_t i = 0; i < inner; ++i) {
                    double running = 0.0;
                    for (size_t k = 0; k < dim; ++k) {
                        const size_t index = (o * dim + k) * inner + i;
                        running += values[index];
                        ASSERT_NEAR(result[index], running, tolerance(running, k + 1))
                            << "shape rank " << test.shape.size() << " axis " << test.axis
                            << " index " << index;
                    }
                }
            }
        }
        std::vector<int> integers(5 * 3000);
        for (size_t i = 0; i < integers.size(); ++i) {
            integers[i] = static_cast<int>(i % 13) - 6;
        }
        const Tensor vulkan = upload_vulkan(Tensor::from_vector(integers, {5, 3000}, Device::CPU));
        const std::vector<float> scanned = vulkan.cumsum(1).cpu().to(DataType::Float32).to_vector();
        for (size_t row = 0; row < 5; ++row) {
            int running = 0;
            for (size_t k = 0; k < 3000; ++k) {
                running += integers[row * 3000 + k];
                ASSERT_EQ(scanned[row * 3000 + k], static_cast<float>(running)) << row << "," << k;
            }
        }
    }

    TEST_F(TensorVulkanReduce, CountsAndSpecialValueChecksReadBackExactTallies) {
        // Catches a counter that is not zeroed between calls, a tally that
        // misses the tail beyond the last full workgroup, and an infinity test
        // that also matches NaN.
        constexpr size_t count = 1048583;
        std::vector<float> values = pattern(count);
        size_t zeros = 0;
        for (size_t i = 0; i < count; ++i) {
            if (i % 31 == 10) {
                values[i] = 0.0f;
                ++zeros;
            }
        }
        const Tensor cpu = Tensor::from_vector(values, {count}, Device::CPU);
        const Tensor vulkan = upload_vulkan(cpu);
        EXPECT_EQ(vulkan.count_nonzero(), count - zeros);
        const Tensor mask = vulkan.gt(0.0f);
        size_t positives = 0;
        for (const float value : values) {
            positives += value > 0.0f ? 1 : 0;
        }
        EXPECT_EQ(mask.count_nonzero(), positives);
        EXPECT_EQ(mask.count_nonzero(), positives);
        EXPECT_FALSE(vulkan.has_nan());
        EXPECT_FALSE(vulkan.has_inf());
        values[count - 1] = std::numeric_limits<float>::quiet_NaN();
        const Tensor with_nan = upload_vulkan(Tensor::from_vector(values, {count}, Device::CPU));
        EXPECT_TRUE(with_nan.has_nan());
        EXPECT_FALSE(with_nan.has_inf());
        values[count - 1] = 1.0f;
        values[4099] = -std::numeric_limits<float>::infinity();
        const Tensor with_inf = upload_vulkan(Tensor::from_vector(values, {count}, Device::CPU));
        EXPECT_FALSE(with_inf.has_nan());
        EXPECT_TRUE(with_inf.has_inf());
    }

} // namespace
