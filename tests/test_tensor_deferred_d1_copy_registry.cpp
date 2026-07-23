/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/tensor.hpp"
#include "core/tensor/internal/lazy_config.hpp"
#include "core/tensor/internal/lazy_executor.hpp"
#include "core/tensor/internal/lazy_ir.hpp"

#include <array>
#include <barrier>
#include <cuda_runtime.h>
#include <exception>
#include <gtest/gtest.h>
#include <optional>
#include <span>
#include <thread>
#include <utility>
#include <vector>

using namespace lfs::core;

namespace {

    class LazyTestGuard {
    public:
        LazyTestGuard() {
            internal::clear_lazy_ir_for_testing();
            internal::lazy_executor_clear_registry_for_testing();
            internal::lazy_executor_reset_diagnostics_for_testing();
            internal::lazy_executor_set_debug_dump_override_for_testing(std::nullopt);
            internal::lazy_executor_set_pointwise_fusion_override_for_testing(std::nullopt);
            internal::lazy_executor_set_size_heuristic_override_for_testing(false);
            internal::lazy_executor_set_size_threshold_override_for_testing(std::nullopt);
            internal::lazy_ir_set_node_limit_override_for_testing(std::nullopt);
            Tensor::reset_lazy_telemetry();
        }

        ~LazyTestGuard() {
            internal::clear_lazy_ir_for_testing();
            internal::lazy_executor_clear_registry_for_testing();
            internal::lazy_executor_reset_diagnostics_for_testing();
            internal::lazy_executor_set_debug_dump_override_for_testing(std::nullopt);
            internal::lazy_executor_set_pointwise_fusion_override_for_testing(std::nullopt);
            internal::lazy_executor_set_size_heuristic_override_for_testing(std::nullopt);
            internal::lazy_executor_set_size_threshold_override_for_testing(std::nullopt);
            internal::lazy_ir_set_node_limit_override_for_testing(std::nullopt);
            Tensor::reset_lazy_telemetry();
        }
    };

    bool has_cuda_device() {
        int device_count = 0;
        const cudaError_t status = cudaGetDeviceCount(&device_count);
        return status == cudaSuccess && device_count > 0;
    }

    const void* data_pointer(const Tensor& tensor) {
        return tensor.data_ptr();
    }

    const void* storage_pointer(const Tensor& tensor) {
        return tensor.storage_ptr();
    }

    void report_thread_exception(const std::exception_ptr& error) {
        if (!error) {
            return;
        }
        try {
            std::rethrow_exception(error);
        } catch (const std::exception& exception) {
            ADD_FAILURE() << exception.what();
        } catch (...) {
            ADD_FAILURE() << "worker threw a non-standard exception";
        }
    }

} // namespace

TEST(DeferredD1CopyRegistryTest, CopyOfDeferredSharesMaterialization) {
    LazyTestGuard guard;

    Tensor a = Tensor::ones({8}, Device::CPU, DataType::Float32).add(2.0f);
    Tensor b = a;
    ASSERT_TRUE(a.is_deferred());
    ASSERT_TRUE(b.is_deferred());

    const auto before = internal::lazy_executor_diagnostics_snapshot_for_testing();
    const auto a_values = a.to_vector();
    const auto b_values = b.to_vector();
    const auto after = internal::lazy_executor_diagnostics_snapshot_for_testing();

    EXPECT_EQ(a_values, std::vector<float>(8, 3.0f));
    EXPECT_EQ(b_values, a_values);
    EXPECT_EQ(data_pointer(a), data_pointer(b));
    EXPECT_EQ(after.executed_nodes - before.executed_nodes, 1u);
}

TEST(DeferredD1CopyRegistryTest, AssignCopySharesMaterialization) {
    LazyTestGuard guard;

    Tensor a = Tensor::ones({8}, Device::CPU, DataType::Float32).mul(4.0f);
    Tensor b;
    b = a;
    ASSERT_TRUE(a.is_deferred());
    ASSERT_TRUE(b.is_deferred());

    const auto before = internal::lazy_executor_diagnostics_snapshot_for_testing();
    const auto b_values = b.to_vector();
    const auto a_values = a.to_vector();
    const auto after = internal::lazy_executor_diagnostics_snapshot_for_testing();

    EXPECT_EQ(a_values, std::vector<float>(8, 4.0f));
    EXPECT_EQ(b_values, a_values);
    EXPECT_EQ(data_pointer(a), data_pointer(b));
    EXPECT_EQ(after.executed_nodes - before.executed_nodes, 1u);
}

TEST(DeferredD1CopyRegistryTest, DestroyOriginalCopyStillMaterializes) {
    LazyTestGuard guard;

    Tensor copy;
    {
        Tensor original = Tensor::ones({6}, Device::CPU, DataType::Float32).add(5.0f);
        copy = original;
        ASSERT_TRUE(copy.is_deferred());
        ASSERT_GT(internal::lazy_executor_registered_node_count_for_testing(), 0u);
    }

    EXPECT_GT(internal::lazy_executor_registered_node_count_for_testing(), 0u);
    Tensor sibling = copy;
    const auto copy_values = copy.to_vector();
    const auto sibling_values = sibling.to_vector();

    EXPECT_EQ(copy_values, std::vector<float>(6, 6.0f));
    EXPECT_EQ(sibling_values, copy_values);
    EXPECT_EQ(data_pointer(copy), data_pointer(sibling));
    EXPECT_EQ(internal::lazy_executor_registered_node_count_for_testing(), 0u);
}

TEST(DeferredD1CopyRegistryTest, DeferredShapeChainNoDoubleExecuteViewFirst) {
    LazyTestGuard guard;

    Tensor base = Tensor::from_vector(
                      std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f},
                      {2, 3}, Device::CPU)
                      .add(10.0f);
    Tensor sliced = base.slice(0, 1, 2);
    const std::array<int, 2> axes{1, 0};
    Tensor view = sliced.permute(std::span<const int>(axes));
    ASSERT_TRUE(base.is_deferred());
    ASSERT_TRUE(view.is_deferred());

    const auto before = Tensor::lazy_telemetry_snapshot();
    const float* view_data = view.ptr<float>();
    std::vector<float> view_values;
    for (size_t row = 0; row < view.shape()[0]; ++row) {
        view_values.push_back(view_data[row * view.stride(0)]);
    }
    const auto after_view = Tensor::lazy_telemetry_snapshot();
    const auto base_values = base.to_vector();
    const auto after = Tensor::lazy_telemetry_snapshot();

    EXPECT_EQ(base_values,
              (std::vector<float>{11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f}));
    EXPECT_EQ(view_values, (std::vector<float>{14.0f, 15.0f, 16.0f}));
    EXPECT_EQ(storage_pointer(base), storage_pointer(view));
    EXPECT_NE(data_pointer(base), data_pointer(view));
    EXPECT_GT(after_view.materializations - before.materializations, 0u);
    EXPECT_EQ(after.materializations, after_view.materializations);
}

TEST(DeferredD1CopyRegistryTest, DeferredShapeChainNoDoubleExecuteSourceFirst) {
    LazyTestGuard guard;

    Tensor base = Tensor::from_vector(
                      std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f},
                      {2, 3}, Device::CPU)
                      .add(10.0f);
    Tensor base_sibling = base;
    Tensor sliced = base.slice(0, 1, 2);
    const std::array<int, 2> axes{1, 0};
    Tensor view = sliced.permute(std::span<const int>(axes));
    ASSERT_TRUE(base.is_deferred());
    ASSERT_TRUE(view.is_deferred());

    const auto base_values = base.to_vector();
    const auto after_base = Tensor::lazy_telemetry_snapshot();
    const auto sibling_values = base_sibling.to_vector();
    const auto after_sibling = Tensor::lazy_telemetry_snapshot();
    const float* view_data = view.ptr<float>();
    std::vector<float> view_values;
    for (size_t row = 0; row < view.shape()[0]; ++row) {
        view_values.push_back(view_data[row * view.stride(0)]);
    }
    const auto after = Tensor::lazy_telemetry_snapshot();

    EXPECT_EQ(base_values,
              (std::vector<float>{11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f}));
    EXPECT_EQ(sibling_values, base_values);
    EXPECT_EQ(view_values, (std::vector<float>{14.0f, 15.0f, 16.0f}));
    EXPECT_EQ(data_pointer(base), data_pointer(base_sibling));
    EXPECT_EQ(storage_pointer(base), storage_pointer(view));
    EXPECT_NE(data_pointer(base), data_pointer(view));
    EXPECT_EQ(after_sibling.materializations, after_base.materializations);
    EXPECT_GE(after.materializations, after_sibling.materializations);
}

TEST(DeferredD1CopyRegistryTest, InPlaceCatCapacityIndependent) {
    if (!has_cuda_device()) {
        GTEST_SKIP() << "CUDA device is required for capacity-backed in-place cat";
    }
    LazyTestGuard guard;

    Tensor reserved = Tensor::from_vector(
        std::vector<float>{0.0f, 1.0f}, {2, 1}, Device::CUDA);
    reserved.reserve(8);
    Tensor alias = reserved;
    Tensor tail = Tensor::from_vector(
        std::vector<float>{2.0f, 3.0f}, {2, 1}, Device::CUDA);
    Tensor result = Tensor::cat({alias, tail}, 0);
    Tensor extra = Tensor::from_vector(
        std::vector<float>{4.0f}, {1, 1}, Device::CUDA);
    Tensor grown = Tensor::cat({result, extra}, 0);

    EXPECT_EQ(grown.to_vector(),
              (std::vector<float>{0.0f, 1.0f, 2.0f, 3.0f, 4.0f}));
    EXPECT_EQ(reserved.capacity(), 8u);
    EXPECT_EQ(alias.capacity(), 8u);
    EXPECT_EQ(result.capacity(), 8u);
    EXPECT_EQ(grown.capacity(), 8u);
    EXPECT_EQ(reserved.logical_size(), 2u);
    EXPECT_EQ(alias.logical_size(), 2u);
    EXPECT_EQ(result.logical_size(), 4u);
    EXPECT_EQ(grown.logical_size(), 5u);
    EXPECT_EQ(storage_pointer(reserved), storage_pointer(alias));
    EXPECT_EQ(storage_pointer(alias), storage_pointer(result));
    EXPECT_EQ(storage_pointer(result), storage_pointer(grown));
}

TEST(DeferredD1CopyRegistryTest, NamePreservedPerTensor) {
    LazyTestGuard guard;

    Tensor a = Tensor::ones({4}, Device::CPU, DataType::Float32).add(2.0f);
    a.set_name("x");
    Tensor b = a;
    b.set_name("y");

    const auto a_values = a.to_vector();
    const auto b_values = b.to_vector();

    EXPECT_EQ(a_values, std::vector<float>(4, 3.0f));
    EXPECT_EQ(b_values, a_values);
    EXPECT_EQ(data_pointer(a), data_pointer(b));
    EXPECT_EQ(a.name(), "x");
    EXPECT_EQ(b.name(), "y");
}

TEST(DeferredD1CopyRegistryTest, MoveFromIsNotDeferred) {
    LazyTestGuard guard;

    Tensor ctor_source = Tensor::ones({5}, Device::CPU, DataType::Float32).add(1.0f);
    Tensor ctor_alias = ctor_source;
    Tensor move_constructed = std::move(ctor_source);
    EXPECT_FALSE(ctor_source.is_deferred());
    EXPECT_FALSE(ctor_source.is_valid());

    Tensor assign_source = Tensor::ones({5}, Device::CPU, DataType::Float32).mul(3.0f);
    Tensor assign_alias = assign_source;
    Tensor move_assigned;
    move_assigned = std::move(assign_source);
    EXPECT_FALSE(assign_source.is_deferred());
    EXPECT_FALSE(assign_source.is_valid());

    EXPECT_EQ(move_constructed.to_vector(), std::vector<float>(5, 2.0f));
    EXPECT_EQ(ctor_alias.to_vector(), std::vector<float>(5, 2.0f));
    EXPECT_EQ(data_pointer(move_constructed), data_pointer(ctor_alias));
    EXPECT_EQ(move_assigned.to_vector(), std::vector<float>(5, 3.0f));
    EXPECT_EQ(assign_alias.to_vector(), std::vector<float>(5, 3.0f));
    EXPECT_EQ(data_pointer(move_assigned), data_pointer(assign_alias));
}

TEST(DeferredD1CopyRegistryTest, ConcurrentSiblingMaterialize) {
    LazyTestGuard guard;

    Tensor a = Tensor::ones({64}, Device::CPU, DataType::Float32).add(7.0f);
    Tensor b = a;
    std::barrier start(3);
    std::vector<float> a_values;
    std::vector<float> b_values;
    std::exception_ptr a_error;
    std::exception_ptr b_error;
    const auto before = internal::lazy_executor_diagnostics_snapshot_for_testing();

    std::thread a_thread([&] {
        start.arrive_and_wait();
        try {
            a_values = a.to_vector();
        } catch (...) {
            a_error = std::current_exception();
        }
    });
    std::thread b_thread([&] {
        start.arrive_and_wait();
        try {
            b_values = b.to_vector();
        } catch (...) {
            b_error = std::current_exception();
        }
    });
    start.arrive_and_wait();
    a_thread.join();
    b_thread.join();

    report_thread_exception(a_error);
    report_thread_exception(b_error);
    const auto after = internal::lazy_executor_diagnostics_snapshot_for_testing();
    EXPECT_EQ(a_values, std::vector<float>(64, 8.0f));
    EXPECT_EQ(b_values, a_values);
    EXPECT_EQ(data_pointer(a), data_pointer(b));
    EXPECT_EQ(after.executed_nodes - before.executed_nodes, 1u);
}
