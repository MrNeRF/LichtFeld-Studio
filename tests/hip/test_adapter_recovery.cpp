/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "core/detail/gpu_backend_ops.hpp"
#include "core/tensor/backend/hip/hip_backend_ops.hpp"
#include "runtime_fault_control.hpp"
#include <gtest/gtest.h>
using namespace lfs::core;
using namespace lfs::core::internal;
static std::filesystem::path module_path;
TEST(HipAdapterRecovery, PartialShutdownCannotReuseReleasedCacheEntries) {
    lfs::test::FaultControl faults(module_path);
    for (const int fault : {2, 9}) {
        auto* adapter = new HipBackendOps(module_path, 0);
        std::unique_ptr<GpuBackendOps> ops(adapter);
        adapter->ensure_initialized();
        auto storage = ops->allocate(16, 16, {});
        // Checked indexing creates the native device/pinned error buffers.
        auto indices = ops->allocate(4, 16, {});
        const int zero = 0;
        indices.dtype = DataType::Int32;
        ops->copy_host_to_device({.src = {.data = const_cast<int*>(&zero)}, .dst = indices, .bytes = sizeof(zero)});
        auto output = ops->allocate(4, 16, {});
        const StridedLayout values_layout{.rank = 1, .dims = {4}, .strides = {1}, .element_count = 4};
        const StridedLayout indices_layout{.rank = 1, .dims = {1}, .strides = {1}, .element_count = 1};
        ops->gather(storage, indices, output, values_layout, indices_layout,
                    {.dim = 0, .input_size = 4, .index_size = 1, .total_elements = 1}, {});
        ops->synchronize_stream({});
        ops->deallocate(output, {});
        // No live Tensor storage may remain when shutdown is attempted.
        ops->deallocate(indices, {});
        ops->deallocate(storage, {});
        faults.arm(fault);
        EXPECT_THROW(ops->shutdown(), std::exception);
        // Allocation/cache state remains usable after either cache release or
        // native teardown fails. A fresh handle must never alias a dead buffer.
        auto next = ops->allocate(16, 16, {});
        const float values[]{1, 2, 3, 4};
        ops->copy_host_to_device({.src = {.data = const_cast<float*>(values)}, .dst = next, .bytes = sizeof(values)});
        float result[4]{};
        ops->copy_device_to_host({.src = next, .dst = {.data = result}, .bytes = sizeof(result)});
        EXPECT_EQ(result[0], 1);
        EXPECT_EQ(result[3], 4);
        ops->deallocate(next, {});
        ops->shutdown();
        EXPECT_EQ(faults.resources(), 0);
    }
}
int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    if (argc != 2)
        return 2;
    module_path = std::filesystem::absolute(argv[1]);
    return RUN_ALL_TESTS();
}
