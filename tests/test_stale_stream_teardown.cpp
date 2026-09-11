/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

// Trainer teardown can destroy CUDA streams while SplatData tensors still
// stamp those handles. After that, .cpu() / shN_canonical() must not crash
// or latch cudaErrorInvalidResourceHandle for later LFS_CUDA_LAUNCH_CHECK.

#include "core/sh_value_quant.hpp"
#include "core/splat_data.hpp"
#include "core/tensor.hpp"
#include "core/tensor/internal/cuda_stream_context.hpp"
#include "core/tensor/internal/memory_pool.hpp"
#include "core/tensor/internal/stream_lifetime.hpp"

#include <cstdint>
#include <cuda_runtime.h>
#include <gtest/gtest.h>

using namespace lfs::core;

namespace {

    class StaleStreamTeardownTest : public ::testing::Test {
    protected:
        void SetUp() override {
            int device_count = 0;
            if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
                GTEST_SKIP() << "No CUDA device";
            }
            ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
        }
    };

    void stamp_if_cuda(Tensor& tensor, cudaStream_t stream) {
        if (tensor.is_valid() && tensor.device() == Device::CUDA) {
            tensor.set_stream(stream);
        }
    }

} // namespace

TEST_F(StaleStreamTeardownTest, DeadHomeStreamCpuDoesNotPoisonThread) {
    cudaStream_t user_stream = nullptr;
    ASSERT_EQ(cudaStreamCreate(&user_stream), cudaSuccess);

    Tensor tensor;
    {
        CUDAStreamGuard guard(user_stream);
        tensor = Tensor::full({8}, 3.25f, Device::CUDA);
        tensor = tensor * 2.0f;
    }
    ASSERT_TRUE(tensor.is_valid());
    ASSERT_EQ(tensor.stream(), user_stream);

    CudaMemoryPool::instance().release_stream(user_stream);
    ASSERT_EQ(cudaStreamDestroy(user_stream), cudaSuccess);

    const Tensor host = tensor.cpu();
    ASSERT_TRUE(host.is_valid());
    ASSERT_EQ(host.numel(), 8);
    const float* values = host.ptr<float>();
    ASSERT_NE(values, nullptr);
    for (int i = 0; i < 8; ++i) {
        EXPECT_FLOAT_EQ(values[i], 6.5f);
    }

    EXPECT_EQ(cudaGetLastError(), cudaSuccess);

    const Tensor fresh = Tensor::ones({32}, Device::CUDA);
    EXPECT_EQ(fresh.count_nonzero(), 32u);
    EXPECT_EQ(cudaGetLastError(), cudaSuccess);
}

TEST_F(StaleStreamTeardownTest, DetachFromStreamsCoversBoundsThenCanonicalSucceeds) {
    cudaStream_t user_stream = nullptr;
    ASSERT_EQ(cudaStreamCreate(&user_stream), cudaSuccess);

    constexpr size_t n = 4;
    constexpr int sh_degree = 1;

    SplatData model(sh_degree,
                    Tensor::zeros({n, size_t{3}}, Device::CUDA),
                    Tensor::zeros({n, size_t{1}, size_t{3}}, Device::CUDA),
                    Tensor::zeros({n, size_t{3}, size_t{3}}, Device::CUDA),
                    Tensor::zeros({n, size_t{3}}, Device::CUDA),
                    Tensor::zeros({n, size_t{4}}, Device::CUDA),
                    Tensor::zeros({n, size_t{1}}, Device::CUDA),
                    1.0f);

    const auto coeffs_rest = static_cast<std::uint32_t>(model.max_sh_coeffs_rest());
    ASSERT_GT(coeffs_rest, 0u);
    model.shN() = Tensor::zeros({sh_value_quant::sh_value_u16_count(n, coeffs_rest)},
                                Device::CUDA, DataType::Float16);
    model.shN_value_bounds() =
        Tensor::zeros({sh_value_quant::n_bounds_for_prims(n), size_t{2}}, Device::CUDA);
    model.deleted() = Tensor::zeros_bool({n}, Device::CUDA);
    model._densification_info = Tensor::zeros({n}, Device::CUDA);

    stamp_if_cuda(model.means_raw(), user_stream);
    stamp_if_cuda(model.sh0_raw(), user_stream);
    stamp_if_cuda(model.shN_raw(), user_stream);
    stamp_if_cuda(model.shN_value_bounds(), user_stream);
    stamp_if_cuda(model.scaling_raw(), user_stream);
    stamp_if_cuda(model.rotation_raw(), user_stream);
    stamp_if_cuda(model.opacity_raw(), user_stream);
    stamp_if_cuda(model.deleted(), user_stream);
    stamp_if_cuda(model._densification_info, user_stream);

    ASSERT_EQ(model.shN_value_bounds().stream(), user_stream);

    model.detach_from_streams();

    const Tensor* members[] = {
        &model.means_raw(),
        &model.sh0_raw(),
        &model.shN_raw(),
        &model.shN_value_bounds(),
        &model.scaling_raw(),
        &model.rotation_raw(),
        &model.opacity_raw(),
        &model.deleted(),
        &model._densification_info,
    };
    for (const Tensor* tensor : members) {
        ASSERT_TRUE(tensor->is_valid());
        ASSERT_EQ(tensor->device(), Device::CUDA);
        EXPECT_EQ(tensor->stream(), nullptr);
    }
    EXPECT_EQ(model.shN_value_bounds().stream(), nullptr);

    CudaMemoryPool::instance().release_stream(user_stream);
    ASSERT_EQ(cudaStreamDestroy(user_stream), cudaSuccess);

    EXPECT_NO_THROW({
        const Tensor canonical = model.shN_canonical();
        EXPECT_TRUE(canonical.is_valid());
        EXPECT_EQ(canonical.numel(), n * static_cast<size_t>(coeffs_rest) * 3);
    });
    EXPECT_NO_THROW({
        EXPECT_TRUE(model.shN_value_bounds().cpu().is_valid());
        EXPECT_TRUE(model.shN_raw().cpu().is_valid());
    });
    EXPECT_EQ(cudaGetLastError(), cudaSuccess);
}

TEST_F(StaleStreamTeardownTest, RetiredStreamRegistryReuseCycle) {
    const auto fake = reinterpret_cast<cudaStream_t>(static_cast<uintptr_t>(0x1));
    retire_stream(fake);
    EXPECT_TRUE(is_stream_retired(fake));
    unretire_stream(fake);
    EXPECT_FALSE(is_stream_retired(fake));

    retire_stream(nullptr);
    EXPECT_FALSE(is_stream_retired(nullptr));
    EXPECT_FALSE(is_stream_retired(fake));

    cudaStream_t stream_a = nullptr;
    ASSERT_EQ(cudaStreamCreate(&stream_a), cudaSuccess);

    Tensor tensor;
    {
        CUDAStreamGuard guard(stream_a);
        tensor = Tensor::full({4}, 1.0f, Device::CUDA);
    }
    ASSERT_TRUE(tensor.is_valid());
    ASSERT_EQ(tensor.stream(), stream_a);

    CudaMemoryPool::instance().release_stream(stream_a);
    EXPECT_TRUE(is_stream_retired(stream_a));
    ASSERT_EQ(cudaStreamDestroy(stream_a), cudaSuccess);
    EXPECT_TRUE(is_stream_retired(stream_a));

    cudaStream_t stream_b = nullptr;
    ASSERT_EQ(cudaStreamCreate(&stream_b), cudaSuccess);
    retire_stream(stream_b);
    EXPECT_TRUE(is_stream_retired(stream_b));
    tensor.set_stream(stream_b);
    EXPECT_FALSE(is_stream_retired(stream_b));
    CudaMemoryPool::instance().release_stream(stream_b);
    ASSERT_EQ(cudaStreamDestroy(stream_b), cudaSuccess);

    // Pool-level APIs must un-retire a live-by-contract handle. Tests talk to
    // the pool directly (no Tensor::set_stream / setCurrentCUDAStream), and the
    // driver can recycle a previously retired pointer value.
    cudaStream_t stream_s = nullptr;
    ASSERT_EQ(cudaStreamCreate(&stream_s), cudaSuccess);
    void* live = CudaMemoryPool::instance().allocate(256, stream_s);
    ASSERT_NE(live, nullptr);

    retire_stream(stream_s);
    EXPECT_TRUE(is_stream_retired(stream_s));
    CudaMemoryPool::instance().record_stream(live, stream_s);
    EXPECT_FALSE(is_stream_retired(stream_s));

    retire_stream(stream_s);
    EXPECT_TRUE(is_stream_retired(stream_s));
    void* reused = CudaMemoryPool::instance().allocate(256, stream_s);
    ASSERT_NE(reused, nullptr);
    EXPECT_FALSE(is_stream_retired(stream_s));
    CudaMemoryPool::instance().deallocate(reused, stream_s);

    CudaMemoryPool::instance().deallocate(live, stream_s);
    CudaMemoryPool::instance().release_stream(stream_s);
    ASSERT_EQ(cudaStreamDestroy(stream_s), cudaSuccess);
}
