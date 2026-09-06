/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/tensor.hpp"
#include "core/tensor/backend/vulkan/vk_context.hpp"
#include "core/tensor_backend.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <numeric>
#include <string>
#include <vector>

namespace {
    using namespace lfs::core;

    Tensor upload_vulkan(const Tensor& cpu) {
        GpuBackendScope scope(GpuBackend::Vulkan);
        return cpu.to(Device::GPU);
    }

    std::vector<float> pattern(const size_t count, const size_t period = 97) {
        std::vector<float> values(count);
        for (size_t i = 0; i < count; ++i) {
            values[i] = static_cast<float>(static_cast<int>(i % period) - 40) / 8.0f;
        }
        return values;
    }

    // Deterministic pseudo-random indices in [0, extent), with the same
    // generator the host test can replay.
    std::vector<int> pseudo_indices(const size_t count, const size_t extent, uint32_t seed) {
        std::vector<int> result(count);
        for (size_t i = 0; i < count; ++i) {
            seed = seed * 1664525u + 1013904223u;
            result[i] = static_cast<int>((seed >> 8) % static_cast<uint32_t>(extent));
        }
        return result;
    }

    std::vector<float> download(const Tensor& tensor) {
        return tensor.cpu().to(DataType::Float32).to_vector();
    }

    void expect_exact(const Tensor& actual, const std::vector<float>& expected, const std::string& label) {
        ASSERT_EQ(gpu_backend_of(actual), GpuBackend::Vulkan) << label;
        const std::vector<float> values = download(actual);
        ASSERT_EQ(values.size(), expected.size()) << label;
        for (size_t i = 0; i < values.size(); ++i) {
            if (std::isnan(expected[i])) {
                EXPECT_TRUE(std::isnan(values[i])) << label << " index=" << i;
            } else {
                EXPECT_EQ(values[i], expected[i]) << label << " index=" << i;
            }
        }
    }

    class TensorVulkanIndex : public testing::Test {
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

    TEST_F(TensorVulkanIndex, GatherSelectAndTakeMatchHostReferences) {
        // Catches a gather that decodes output coordinates in the wrong order, a
        // take that ignores negative indices, and an index_select that mixes up
        // the outer and inner extents; sizes cover one element, a partial group,
        // one group plus a tail and more than one group limit.
        constexpr std::array sizes{size_t{1}, size_t{7}, size_t{4099}, size_t{300000}};
        for (const size_t count : sizes) {
            const std::vector<float> values = pattern(count);
            const Tensor source = upload_vulkan(Tensor::from_vector(values, {count}, Device::CPU));
            std::vector<int> picks = pseudo_indices(count + 3, count, 7u);
            for (size_t i = 0; i + 1 < picks.size(); i += 5) {
                picks[i] = -static_cast<int>(picks[i]) - 1;
            }
            std::vector<float> taken(picks.size());
            for (size_t i = 0; i < picks.size(); ++i) {
                int index = picks[i];
                if (index < 0) {
                    index += static_cast<int>(count);
                }
                index = std::clamp(index, 0, static_cast<int>(count) - 1);
                taken[i] = values[static_cast<size_t>(index)];
            }
            const Tensor indices = upload_vulkan(Tensor::from_vector(picks, {picks.size()}, Device::CPU));
            expect_exact(source.take(indices), taken, "take n=" + std::to_string(count));

            const std::vector<int> in_range = pseudo_indices(count + 3, count, 11u);
            std::vector<float> gathered(in_range.size());
            for (size_t i = 0; i < in_range.size(); ++i) {
                gathered[i] = values[static_cast<size_t>(in_range[i])];
            }
            const Tensor valid = upload_vulkan(Tensor::from_vector(in_range, {in_range.size()}, Device::CPU));
            expect_exact(source.gather(0, valid), gathered, "gather n=" + std::to_string(count));
            expect_exact(source.index_select(0, valid), gathered, "index_select n=" + std::to_string(count));
        }

        constexpr size_t rows = 37;
        constexpr size_t columns = 65;
        const std::vector<float> matrix = pattern(rows * columns);
        const Tensor source = upload_vulkan(Tensor::from_vector(matrix, {rows, columns}, Device::CPU));
        const std::vector<int> column_picks = pseudo_indices(rows * 9, columns, 3u);
        const Tensor column_index = upload_vulkan(Tensor::from_vector(column_picks, {rows, 9}, Device::CPU));
        std::vector<float> gathered(rows * 9);
        for (size_t r = 0; r < rows; ++r) {
            for (size_t k = 0; k < 9; ++k) {
                gathered[r * 9 + k] = matrix[r * columns + static_cast<size_t>(column_picks[r * 9 + k])];
            }
        }
        expect_exact(source.gather(1, column_index), gathered, "gather dim 1");

        const std::vector<int> row_picks = pseudo_indices(11, rows, 5u);
        const Tensor row_index = upload_vulkan(Tensor::from_vector(row_picks, {11}, Device::CPU));
        std::vector<float> selected_rows(11 * columns);
        for (size_t k = 0; k < 11; ++k) {
            std::copy_n(matrix.begin() + static_cast<long>(row_picks[k]) * columns, columns,
                        selected_rows.begin() + static_cast<long>(k) * columns);
        }
        expect_exact(source.index_select(0, row_index), selected_rows, "index_select dim 0");
        const std::vector<int> inner_picks = pseudo_indices(5, columns, 9u);
        const Tensor inner_index = upload_vulkan(Tensor::from_vector(inner_picks, {5}, Device::CPU));
        std::vector<float> selected_columns(rows * 5);
        for (size_t r = 0; r < rows; ++r) {
            for (size_t k = 0; k < 5; ++k) {
                selected_columns[r * 5 + k] = matrix[r * columns + static_cast<size_t>(inner_picks[k])];
            }
        }
        expect_exact(source.index_select(1, inner_index), selected_columns, "index_select dim 1");

        std::vector<int> integers(rows * columns);
        for (size_t i = 0; i < integers.size(); ++i) {
            integers[i] = static_cast<int>(i) * 7 - 1000;
        }
        const Tensor int_source = upload_vulkan(Tensor::from_vector(integers, {rows, columns}, Device::CPU));
        std::vector<float> int_selected(rows * 5);
        for (size_t r = 0; r < rows; ++r) {
            for (size_t k = 0; k < 5; ++k) {
                int_selected[r * 5 + k] = static_cast<float>(integers[r * columns + static_cast<size_t>(inner_picks[k])]);
            }
        }
        expect_exact(int_source.index_select(1, inner_index), int_selected, "index_select Int32");
    }

    TEST_F(TensorVulkanIndex, BoundaryModesClampWrapAndAssert) {
        // Catches a clamp or wrap that mishandles negative indices, and an
        // assert mode that lets an out-of-range index read silently instead of
        // recording a device fault the host turns into an error.
        const std::vector<float> values = pattern(50);
        const Tensor source = upload_vulkan(Tensor::from_vector(values, {50}, Device::CPU));
        const std::vector<int> wild{-3, 0, 49, 50, 77, -120, 12};
        const Tensor indices = upload_vulkan(Tensor::from_vector(wild, {wild.size()}, Device::CPU));
        std::vector<float> clamped(wild.size());
        std::vector<float> wrapped(wild.size());
        for (size_t i = 0; i < wild.size(); ++i) {
            clamped[i] = values[static_cast<size_t>(std::clamp(wild[i], 0, 49))];
            wrapped[i] = values[static_cast<size_t>(((wild[i] % 50) + 50) % 50)];
        }
        expect_exact(source.index_select(0, indices, BoundaryMode::Clamp), clamped, "clamp");
        expect_exact(source.index_select(0, indices, BoundaryMode::Wrap), wrapped, "wrap");
        expect_exact(source.gather(0, indices, BoundaryMode::Clamp), clamped, "gather clamp");
        expect_exact(source.gather(0, indices, BoundaryMode::Wrap), wrapped, "gather wrap");
        // The launch records the first out-of-range index; the next
        // synchronization raises it as a Vulkan-domain error and clears it.
        EXPECT_THROW(static_cast<void>(source.index_select(0, indices, BoundaryMode::Assert)),
                     std::exception);
        const std::vector<int> fine{1, 2, 3};
        const Tensor fine_indices = upload_vulkan(Tensor::from_vector(fine, {3}, Device::CPU));
        expect_exact(source.index_select(0, fine_indices, BoundaryMode::Assert),
                     {values[1], values[2], values[3]}, "assert after fault is consumed");
        // gather validates its indices on the host before any launch.
        EXPECT_THROW(static_cast<void>(source.gather(0, indices, BoundaryMode::Assert)), std::exception);
    }

    TEST_F(TensorVulkanIndex, ScatterFamilyAssignsAndAccumulates) {
        // Catches a scatter that indexes the source by the destination
        // position, an index_add that loses concurrent updates on duplicate
        // indices, and an integer accumulation that goes through float.
        constexpr size_t count = 4099;
        const std::vector<float> base = pattern(count);
        const std::vector<float> source = pattern(count, 53);
        const std::vector<int> targets = pseudo_indices(count, count, 21u);
        const Tensor index = upload_vulkan(Tensor::from_vector(targets, {count}, Device::CPU));
        const Tensor values = upload_vulkan(Tensor::from_vector(source, {count}, Device::CPU));

        std::vector<double> accumulated(base.begin(), base.end());
        for (size_t i = 0; i < count; ++i) {
            accumulated[static_cast<size_t>(targets[i])] += source[i];
        }
        {
            Tensor destination = upload_vulkan(Tensor::from_vector(base, {count}, Device::CPU));
            destination.index_add_(0, index, values);
            const std::vector<float> result = download(destination);
            for (size_t i = 0; i < count; ++i) {
                EXPECT_NEAR(result[i], accumulated[i], 1e-4 * (1.0 + std::abs(accumulated[i])))
                    << "index_add float index=" << i;
            }
        }

        std::vector<int> integer_base(count);
        std::vector<int> integer_source(count);
        std::vector<int> integer_expected(count);
        for (size_t i = 0; i < count; ++i) {
            integer_base[i] = static_cast<int>(i % 13);
            integer_source[i] = static_cast<int>(i % 29) - 14;
            integer_expected[i] = integer_base[i];
        }
        for (size_t i = 0; i < count; ++i) {
            integer_expected[static_cast<size_t>(targets[i])] += integer_source[i];
        }
        Tensor integers = upload_vulkan(Tensor::from_vector(integer_base, {count}, Device::CPU));
        integers.index_add_(0, index, upload_vulkan(Tensor::from_vector(integer_source, {count}, Device::CPU)));
        expect_exact(integers, std::vector<float>(integer_expected.begin(), integer_expected.end()),
                     "index_add Int32");

        constexpr size_t rows = 19;
        constexpr size_t columns = 33;
        const std::vector<float> matrix = pattern(rows * columns);
        const std::vector<int> row_targets{4, 0, 18, 7, 11};
        const std::vector<float> rows_source = pattern(row_targets.size() * columns, 31);
        std::vector<float> copied = matrix;
        for (size_t k = 0; k < row_targets.size(); ++k) {
            std::copy_n(rows_source.begin() + static_cast<long>(k) * columns, columns,
                        copied.begin() + static_cast<long>(row_targets[k]) * columns);
        }
        Tensor destination = upload_vulkan(Tensor::from_vector(matrix, {rows, columns}, Device::CPU));
        destination.index_copy_(0, upload_vulkan(Tensor::from_vector(row_targets, {row_targets.size()}, Device::CPU)),
                                upload_vulkan(Tensor::from_vector(rows_source, {row_targets.size(), columns}, Device::CPU)));
        expect_exact(destination, copied, "index_copy dim 0");

        const std::vector<int> column_targets{2, 30, 15};
        const std::vector<float> column_source = pattern(rows * column_targets.size(), 41);
        std::vector<float> column_copied = matrix;
        for (size_t r = 0; r < rows; ++r) {
            for (size_t k = 0; k < column_targets.size(); ++k) {
                column_copied[r * columns + static_cast<size_t>(column_targets[k])] =
                    column_source[r * column_targets.size() + k];
            }
        }
        Tensor column_destination = upload_vulkan(Tensor::from_vector(matrix, {rows, columns}, Device::CPU));
        column_destination.index_copy_(
            1, upload_vulkan(Tensor::from_vector(column_targets, {column_targets.size()}, Device::CPU)),
            upload_vulkan(Tensor::from_vector(column_source, {rows, column_targets.size()}, Device::CPU)));
        expect_exact(column_destination, column_copied, "index_copy dim 1");

        std::vector<float> filled = matrix;
        for (const int target : column_targets) {
            for (size_t r = 0; r < rows; ++r) {
                filled[r * columns + static_cast<size_t>(target)] = -2.5f;
            }
        }
        Tensor fill_destination = upload_vulkan(Tensor::from_vector(matrix, {rows, columns}, Device::CPU));
        fill_destination.index_fill_(
            1, upload_vulkan(Tensor::from_vector(column_targets, {column_targets.size()}, Device::CPU)), -2.5f);
        expect_exact(fill_destination, filled, "index_fill dim 1");

        // Duplicate targets: the CPU reference assigns in position order, so the
        // last source position wins; the Vulkan path must agree exactly.
        std::vector<float> scattered = base;
        for (size_t i = 0; i < count; ++i) {
            scattered[static_cast<size_t>(targets[i])] = source[i];
        }
        Tensor scatter_destination = upload_vulkan(Tensor::from_vector(base, {count}, Device::CPU));
        scatter_destination.scatter_(0, index, values, ScatterMode::None);
        expect_exact(scatter_destination, scattered, "scatter assign with duplicates");
        Tensor cpu_destination = Tensor::from_vector(base, {count}, Device::CPU);
        cpu_destination.scatter_(0, Tensor::from_vector(targets, {count}, Device::CPU),
                                 Tensor::from_vector(source, {count}, Device::CPU), ScatterMode::None);
        EXPECT_EQ(download(scatter_destination), cpu_destination.to_vector());
    }

    TEST_F(TensorVulkanIndex, FusedGatherAppliesTheUnaryAfterTheRead) {
        // Catches a fused gather that applies the unary to the index instead of
        // the gathered value (the expression validates the indices on the host).
        const std::vector<float> values = pattern(4099);
        const Tensor source = upload_vulkan(Tensor::from_vector(values, {4099}, Device::CPU));
        const std::vector<int> picks = pseudo_indices(6000, 4099, 17u);
        const Tensor indices = upload_vulkan(Tensor::from_vector(picks, {picks.size()}, Device::CPU));
        std::vector<float> expected(picks.size());
        for (size_t i = 0; i < picks.size(); ++i) {
            expected[i] = std::abs(values[static_cast<size_t>(picks[i])]);
        }
        expect_exact(source.gather_lazy(indices).map(lfs::core::ops::abs_op{}).eval(), expected, "fused abs");
    }

    class TensorVulkanIndexNoAtomicFloat : public TensorVulkanIndex {
    protected:
        void SetUp() override {
            setenv("LFS_VULKAN_FORCE_NO_ATOMIC_FLOAT", "1", 1);
            TensorVulkanIndex::SetUp();
        }

        void TearDown() override {
            TensorVulkanIndex::TearDown();
            unsetenv("LFS_VULKAN_FORCE_NO_ATOMIC_FLOAT");
        }
    };

    TEST_F(TensorVulkanIndexNoAtomicFloat, IndexAddFallbackMatchesTheSequentialCpuReference) {
        // Catches a fallback that still uses atomics (nondeterministic float
        // order) or a sorted-run pass that skips a run start or a duplicate.
        constexpr size_t count = 4099;
        const std::vector<float> base = pattern(count);
        const std::vector<float> source = pattern(count, 53);
        const std::vector<int> targets = pseudo_indices(count, 97, 21u);
        Tensor destination = upload_vulkan(Tensor::from_vector(base, {count}, Device::CPU));
        ASSERT_FALSE(internal::vulkan_device_caps_for_testing().shader_atomic_float);
        destination.index_add_(0, upload_vulkan(Tensor::from_vector(targets, {count}, Device::CPU)),
                               upload_vulkan(Tensor::from_vector(source, {count}, Device::CPU)));
        Tensor reference = Tensor::from_vector(base, {count}, Device::CPU);
        reference.index_add_(0, Tensor::from_vector(targets, {count}, Device::CPU),
                             Tensor::from_vector(source, {count}, Device::CPU));
        EXPECT_EQ(download(destination), reference.to_vector());

        constexpr size_t rows = 23;
        constexpr size_t columns = 17;
        const std::vector<float> matrix = pattern(rows * columns);
        const std::vector<int> row_targets = pseudo_indices(40, rows, 5u);
        const std::vector<float> rows_source = pattern(40 * columns, 31);
        Tensor matrix_destination = upload_vulkan(Tensor::from_vector(matrix, {rows, columns}, Device::CPU));
        matrix_destination.index_add_(0, upload_vulkan(Tensor::from_vector(row_targets, {40}, Device::CPU)),
                                      upload_vulkan(Tensor::from_vector(rows_source, {40, columns}, Device::CPU)));
        Tensor matrix_reference = Tensor::from_vector(matrix, {rows, columns}, Device::CPU);
        matrix_reference.index_add_(0, Tensor::from_vector(row_targets, {40}, Device::CPU),
                                    Tensor::from_vector(rows_source, {40, columns}, Device::CPU));
        EXPECT_EQ(download(matrix_destination), matrix_reference.to_vector());
    }

    class TensorVulkanMask : public TensorVulkanIndex {};

    TEST_F(TensorVulkanMask, MaskedFillSelectAndScatterAcrossSizes) {
        // Catches a compaction scan that drops the carry between chunks, a fill
        // that writes unmasked elements, and a masked scatter that reads the
        // source by the destination position.
        constexpr std::array sizes{size_t{1}, size_t{7}, size_t{4099}, size_t{300000}};
        for (const size_t count : sizes) {
            const std::vector<float> values = pattern(count);
            const Tensor source = upload_vulkan(Tensor::from_vector(values, {count}, Device::CPU));
            const Tensor mask = source.gt(0.0f);
            std::vector<float> selected;
            std::vector<float> filled = values;
            for (size_t i = 0; i < count; ++i) {
                if (values[i] > 0.0f) {
                    selected.push_back(values[i]);
                    filled[i] = 9.0f;
                }
            }
            const std::string label = " n=" + std::to_string(count);
            const Tensor picked = source.masked_select(mask);
            ASSERT_EQ(picked.numel(), selected.size()) << label;
            if (!selected.empty()) {
                expect_exact(picked, selected, "masked_select" + label);
            }
            Tensor target = source.clone();
            target.masked_fill_(mask, 9.0f);
            expect_exact(target, filled, "masked_fill" + label);
            if (!selected.empty()) {
                std::vector<float> replacement(selected.size());
                for (size_t i = 0; i < replacement.size(); ++i) {
                    replacement[i] = -static_cast<float>(i) - 0.5f;
                }
                std::vector<float> scattered = values;
                size_t next = 0;
                for (size_t i = 0; i < count; ++i) {
                    if (values[i] > 0.0f) {
                        scattered[i] = replacement[next++];
                    }
                }
                Tensor destination = source.clone();
                destination[mask] = upload_vulkan(Tensor::from_vector(replacement, {replacement.size()}, Device::CPU));
                expect_exact(destination, scattered, "masked_scatter" + label);
            }
        }
        std::vector<int> integers(4099);
        for (size_t i = 0; i < integers.size(); ++i) {
            integers[i] = static_cast<int>(i % 5) - 2;
        }
        Tensor int_target = upload_vulkan(Tensor::from_vector(integers, {4099}, Device::CPU));
        const Tensor int_mask = int_target.gt(0.0f);
        int_target.masked_fill_(int_mask, 7.0f);
        std::vector<float> int_expected(4099);
        for (size_t i = 0; i < integers.size(); ++i) {
            int_expected[i] = integers[i] > 0 ? 7.0f : static_cast<float>(integers[i]);
        }
        expect_exact(int_target, int_expected, "masked_fill Int32");
    }

    TEST_F(TensorVulkanMask, NonzeroAndLiveAndWhere) {
        // Catches nonzero indices written in the wrong order, and_live keeping
        // dead entries, and a where that broadcasts the condition on the wrong axis.
        std::vector<float> values = pattern(4099);
        for (size_t i = 0; i < values.size(); i += 3) {
            values[i] = 0.0f;
        }
        const Tensor source = upload_vulkan(Tensor::from_vector(values, {4099}, Device::CPU));
        std::vector<float> expected_positions;
        for (size_t i = 0; i < values.size(); ++i) {
            if (values[i] != 0.0f) {
                expected_positions.push_back(static_cast<float>(i));
            }
        }
        const Tensor positions = source.nonzero();
        ASSERT_EQ(gpu_backend_of(positions), GpuBackend::Vulkan);
        EXPECT_EQ(download(positions), expected_positions);
        const Tensor bool_positions = source.gt(0.0f).nonzero();
        std::vector<float> expected_positive;
        for (size_t i = 0; i < values.size(); ++i) {
            if (values[i] > 0.0f) {
                expected_positive.push_back(static_cast<float>(i));
            }
        }
        EXPECT_EQ(download(bool_positions), expected_positive);

        Tensor mask = source.gt(0.0f);
        const Tensor live = source.lt(2.0f);
        std::vector<float> expected_live(values.size());
        for (size_t i = 0; i < values.size(); ++i) {
            expected_live[i] = (values[i] > 0.0f && values[i] < 2.0f) ? 1.0f : 0.0f;
        }
        mask.and_live_(live);
        expect_exact(mask, expected_live, "and_live");

        constexpr size_t rows = 21;
        constexpr size_t columns = 300;
        const std::vector<float> x = pattern(rows * columns);
        std::vector<float> y(columns);
        std::vector<float> condition_values(rows);
        for (size_t c = 0; c < columns; ++c) {
            y[c] = -static_cast<float>(c);
        }
        for (size_t r = 0; r < rows; ++r) {
            condition_values[r] = (r % 3 == 0) ? 1.0f : 0.0f;
        }
        std::vector<float> expected_where(rows * columns);
        for (size_t r = 0; r < rows; ++r) {
            for (size_t c = 0; c < columns; ++c) {
                expected_where[r * columns + c] = condition_values[r] != 0.0f ? x[r * columns + c] : y[c];
            }
        }
        const Tensor condition = upload_vulkan(Tensor::from_vector(condition_values, {rows, 1}, Device::CPU)).gt(0.5f);
        const Tensor chosen = Tensor::where(condition,
                                            upload_vulkan(Tensor::from_vector(x, {rows, columns}, Device::CPU)),
                                            upload_vulkan(Tensor::from_vector(y, {1, columns}, Device::CPU)));
        expect_exact(chosen, expected_where, "where");
    }

    class TensorVulkanSort : public TensorVulkanIndex {};

    // Values are compared exactly; the returned order must reproduce the
    // sorted values from the input, tie order among equal keys is not compared.
    void expect_sorted(const Tensor& sorted, const Tensor& order, const std::vector<float>& values,
                       const bool descending, const std::string& label) {
        std::vector<float> expected_values = values;
        if (descending) {
            std::sort(expected_values.begin(), expected_values.end(), std::greater<float>());
        } else {
            std::sort(expected_values.begin(), expected_values.end());
        }
        expect_exact(sorted, expected_values, label + " values");
        const std::vector<float> actual_order = download(order);
        ASSERT_EQ(actual_order.size(), values.size()) << label;
        std::vector<bool> used(values.size(), false);
        for (size_t i = 0; i < values.size(); ++i) {
            const auto position = static_cast<size_t>(actual_order[i]);
            ASSERT_LT(position, values.size()) << label << " index=" << i;
            EXPECT_FALSE(used[position]) << label << " duplicate position " << position;
            used[position] = true;
            EXPECT_EQ(values[position], expected_values[i]) << label << " index=" << i;
        }
    }

    TEST_F(TensorVulkanSort, OneDimensionalSortMatchesAcrossBothPaths) {
        // Catches a key mapping that misorders negative values or infinities,
        // an order array that does not permute the input, and a radix pass that
        // drops elements beyond the last full block; sizes cover the
        // shared-memory bitonic path and the multi-block radix path.
        constexpr std::array sizes{size_t{1}, size_t{7}, size_t{2048}, size_t{2049}, size_t{4099}, size_t{300000}};
        for (const size_t count : sizes) {
            std::vector<float> values = pattern(count, 41);
            if (count > 10) {
                values[count - 2] = -std::numeric_limits<float>::infinity();
                values[9] = std::numeric_limits<float>::infinity();
            }
            const Tensor source = upload_vulkan(Tensor::from_vector(values, {count}, Device::CPU));
            for (const bool descending : {false, true}) {
                const auto [sorted, order] = source.sort(0, descending);
                expect_sorted(sorted, order, values, descending,
                              std::string(descending ? "descending" : "ascending") + " n=" + std::to_string(count));
            }
        }
    }

    TEST_F(TensorVulkanSort, NaNsSortLastAscendingAndZerosOfBothSignsKeepInputOrder) {
        // Owner decision 4: NaN (either sign) after every number in ascending
        // order and before every number in descending order; -0 and +0 compare
        // equal and keep their input order. Both the shared-memory path and the
        // radix path are covered; a key that separates the zeros or lets a
        // negative NaN sort first fails here.
        for (const size_t count : {size_t{64}, size_t{5000}}) {
            std::vector<float> values(count);
            for (size_t i = 0; i < count; ++i) {
                values[i] = static_cast<float>(static_cast<int>(i % 23) - 11) * 0.5f;
            }
            const uint32_t negative_nan_bits = 0xFFC00000u;
            values[3] = std::numeric_limits<float>::quiet_NaN();
            values[count / 2] = std::bit_cast<float>(negative_nan_bits);
            values[count - 1] = std::numeric_limits<float>::quiet_NaN();
            for (size_t i = 5; i < count; i += 7) {
                values[i] = (i % 2 == 0) ? 0.0f : -0.0f;
            }
            std::vector<size_t> zero_positions;
            for (size_t i = 0; i < count; ++i) {
                if (values[i] == 0.0f) {
                    zero_positions.push_back(i);
                }
            }
            const Tensor source = upload_vulkan(Tensor::from_vector(values, {count}, Device::CPU));
            for (const bool descending : {false, true}) {
                const auto [sorted, order] = source.sort(0, descending);
                const std::vector<float> out = sorted.cpu().to_vector();
                const std::vector<int64_t> idx = order.cpu().to_vector_int64();
                ASSERT_EQ(out.size(), count);
                const std::string label = descending ? "descending" : "ascending";
                for (size_t k = 0; k < 3; ++k) {
                    const size_t position = descending ? k : count - 1 - k;
                    EXPECT_TRUE(std::isnan(out[position])) << label << " n=" << count << " NaN slot " << position;
                }
                std::vector<size_t> zero_order;
                for (size_t k = 0; k < count; ++k) {
                    if (out[k] == 0.0f) {
                        zero_order.push_back(static_cast<size_t>(idx[k]));
                    }
                }
                EXPECT_EQ(zero_order, zero_positions) << label << " n=" << count << ": zeros must keep input order";
                for (size_t k = 1; k < count; ++k) {
                    if (std::isnan(out[k - 1]) || std::isnan(out[k])) {
                        continue;
                    }
                    if (descending) {
                        EXPECT_GE(out[k - 1], out[k]) << label << " n=" << count << " at " << k;
                    } else {
                        EXPECT_LE(out[k - 1], out[k]) << label << " n=" << count << " at " << k;
                    }
                }
            }
        }
    }

    TEST_F(TensorVulkanSort, TwoDimensionalSortHandlesInnerStridesAndLongLines) {
        // Catches lines that read across each other when the sort dimension is
        // not the innermost one, and a radix path that mixes blocks of
        // different lines.
        struct Case {
            std::vector<size_t> shape;
            int dim;
        };
        const std::array cases{
            Case{{33, 127}, 1},
            Case{{127, 33}, 0},
            Case{{3, 5000}, 1},
            Case{{2500, 4}, 0},
        };
        for (const Case& test : cases) {
            const size_t rows = test.shape[0];
            const size_t columns = test.shape[1];
            const std::vector<float> values = pattern(rows * columns, 61);
            const Tensor source = upload_vulkan(Tensor::from_vector(values, TensorShape(test.shape), Device::CPU));
            const auto [sorted, order] = source.sort(test.dim, false);
            const std::vector<float> sorted_values = download(sorted);
            const std::vector<float> sorted_order = download(order);
            const size_t line_count = test.dim == 1 ? rows : columns;
            const size_t line_length = test.dim == 1 ? columns : rows;
            for (size_t line = 0; line < line_count; ++line) {
                std::vector<float> line_values(line_length);
                for (size_t k = 0; k < line_length; ++k) {
                    line_values[k] = test.dim == 1 ? values[line * columns + k] : values[k * columns + line];
                }
                std::vector<float> expected = line_values;
                std::sort(expected.begin(), expected.end());
                std::vector<bool> used(line_length, false);
                for (size_t k = 0; k < line_length; ++k) {
                    const size_t at = test.dim == 1 ? line * columns + k : k * columns + line;
                    ASSERT_EQ(sorted_values[at], expected[k])
                        << "shape " << rows << "x" << columns << " dim " << test.dim << " line " << line << " k " << k;
                    const auto position = static_cast<size_t>(sorted_order[at]);
                    ASSERT_LT(position, line_length) << "line " << line << " k " << k;
                    ASSERT_FALSE(used[position]) << "line " << line << " duplicate position " << position;
                    used[position] = true;
                    ASSERT_EQ(line_values[position], expected[k])
                        << "shape " << rows << "x" << columns << " dim " << test.dim << " line " << line << " k " << k;
                }
            }
        }
    }

} // namespace
