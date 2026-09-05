/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/tensor.hpp"
#include "core/tensor/backend/vulkan/vk_context.hpp"
#include "core/tensor/internal/cuda_stream_context.hpp"
#include "core/tensor_backend.hpp"

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <cstdint>
#include <cstdlib>
#include <string>
#include <thread>
#include <vector>

namespace {

    using namespace lfs::core;

    // One negative case per mixed-backend validator family listed in plan D1.
    // Each catches a validator that lets storage of two GPU backends reach a
    // kernel, which would hand a Vulkan device address to CUDA or the reverse.
    class TensorBackendValidators : public testing::Test {
    protected:
        void SetUp() override {
            if (!gpu_backend_available(GpuBackend::Vulkan)) {
                GTEST_SKIP() << "Vulkan backend unavailable";
            }
            {
                GpuBackendScope scope(GpuBackend::CUDA);
                cuda_ = Tensor::full({2, 3}, 1.0f, Device::CUDA);
                cuda_indices_ = Tensor::from_vector(std::vector<int>{0, 1}, {2}, Device::CUDA);
            }
            {
                GpuBackendScope scope(GpuBackend::Vulkan);
                vulkan_ = Tensor::full({2, 3}, 2.0f, Device::CUDA);
                vulkan_mask_ = vulkan_.gt(0.0f).contiguous();
            }
        }

        void TearDown() override {
            EXPECT_TRUE(shutdown_gpu_backend(GpuBackend::Vulkan).has_value());
        }

        template <typename Operation>
        void expect_mixed_backend_error(const char* const family, Operation&& operation) {
            try {
                operation();
                FAIL() << family << ": mixed-backend operation did not throw";
            } catch (const std::exception& error) {
                EXPECT_NE(std::string(error.what()).find("matching GPU backends"),
                          std::string::npos)
                    << family << ": " << error.what();
            }
        }

        Tensor cuda_;
        Tensor cuda_indices_;
        Tensor vulkan_;
        Tensor vulkan_mask_;
    };

    TEST_F(TensorBackendValidators, BinaryArithmeticRejectsMixedBackends) {
        expect_mixed_backend_error("binary", [&] { static_cast<void>((cuda_ + vulkan_).to_vector()); });
    }

    TEST_F(TensorBackendValidators, TernaryWhereRejectsMixedBackends) {
        expect_mixed_backend_error("where", [&] {
            static_cast<void>(Tensor::where(vulkan_mask_, cuda_, cuda_).to_vector());
        });
    }

    TEST_F(TensorBackendValidators, MaskedFillRejectsMixedBackends) {
        expect_mixed_backend_error("masked_fill_", [&] { cuda_.masked_fill_(vulkan_mask_, 0.0f); });
    }

    TEST_F(TensorBackendValidators, IndexSelectRejectsMixedBackends) {
        expect_mixed_backend_error("index_select", [&] {
            static_cast<void>(vulkan_.index_select(0, cuda_indices_).to_vector());
        });
    }

    TEST_F(TensorBackendValidators, CatRejectsMixedBackends) {
        expect_mixed_backend_error("cat", [&] {
            static_cast<void>(Tensor::cat({cuda_, vulkan_}, 0).to_vector());
        });
    }

    TEST_F(TensorBackendValidators, CopyFromRejectsMixedBackends) {
        expect_mixed_backend_error("copy_from", [&] { cuda_.copy_from(vulkan_); });
    }

    TEST_F(TensorBackendValidators, LazyExpressionRejectsMixedBackendsAtMaterialization) {
        expect_mixed_backend_error("lazy", [&] {
            Tensor large_cuda;
            Tensor large_vulkan;
            {
                GpuBackendScope scope(GpuBackend::CUDA);
                large_cuda = Tensor::full({4096}, 1.0f, Device::CUDA);
            }
            {
                GpuBackendScope scope(GpuBackend::Vulkan);
                large_vulkan = Tensor::full({4096}, 2.0f, Device::CUDA);
            }
            static_cast<void>(large_cuda.add(large_vulkan).to_vector());
        });
    }

    TEST_F(TensorBackendValidators, MatmulRejectsMixedBackends) {
        Tensor vulkan_rhs;
        {
            GpuBackendScope scope(GpuBackend::Vulkan);
            vulkan_rhs = Tensor::full({3, 2}, 1.0f, Device::CUDA);
        }
        expect_mixed_backend_error("matmul", [&] {
            static_cast<void>(cuda_.mm(vulkan_rhs).to_vector());
        });
    }

    TEST_F(TensorBackendValidators, DeferredExpressionsCarryTheBackendOfTheirLeaves) {
        // Catches a deferred (lazy) tensor reporting the default backend before it
        // materializes: validators then rejected a Vulkan operand against its own
        // deferred result, and the pointwise adapter was chosen from the wrong tag.
        GpuBackendScope scope(GpuBackend::Vulkan);
        const Tensor base = Tensor::ones({100, 100}, Device::CUDA);
        Tensor chain = base;
        for (int step = 0; step < 8; ++step) {
            chain = chain.add(0.001f).mul(1.001f);
            EXPECT_EQ(gpu_backend_of(chain), GpuBackend::Vulkan) << "step " << step;
        }
        const Tensor difference = chain.sub(base);
        EXPECT_EQ(gpu_backend_of(difference), GpuBackend::Vulkan);
        const std::vector<float> values = difference.to_vector();
        ASSERT_EQ(values.size(), 10000u);
        EXPECT_NEAR(values[0], 0.016064f, 1e-4f);
        const Tensor mask = base.gt(0.5f);
        EXPECT_EQ(gpu_backend_of(mask), GpuBackend::Vulkan);
        const Tensor selected = base.mul(mask.to(DataType::Float32)).add(1.0f);
        EXPECT_EQ(gpu_backend_of(selected), GpuBackend::Vulkan);
        EXPECT_EQ(selected.to_vector(), std::vector<float>(10000, 2.0f));
        const Tensor view = base.transpose(0, 1);
        EXPECT_EQ(gpu_backend_of(view), GpuBackend::Vulkan);
        Tensor target = Tensor::zeros({100, 100}, Device::CUDA);
        target.copy_from(view);
        EXPECT_EQ(target.to_vector(), std::vector<float>(10000, 1.0f));
    }

    TEST_F(TensorBackendValidators, VulkanTensorsNeverTakeACudaStreamIdentity) {
        // Catches Vulkan storage taking part in CUDA stream bookkeeping: a deferred
        // result or a prepared input that adopts the current CUDA stream makes every
        // later cross-stream check bridge, which is a queue submit per operation.
        GpuBackendScope scope(GpuBackend::Vulkan);
        cudaStream_t stream = nullptr;
        ASSERT_EQ(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), cudaSuccess);
        {
            CUDAStreamGuard guard(stream);
            const Tensor a = Tensor::ones({1024, 1024}, Device::CUDA);
            const Tensor b = Tensor::full({1024, 1024}, 2.0f, Device::CUDA);
            EXPECT_TRUE(a.stream() == nullptr);
            Tensor c = a.add(b);
            ASSERT_TRUE(c.is_deferred());
            EXPECT_TRUE(c.stream() == nullptr);
            EXPECT_FLOAT_EQ(c.to_vector()[0], 3.0f);
            EXPECT_TRUE(c.stream() == nullptr);
            Tensor d = c.mul(a);
            static_cast<void>(d.to_vector());
            EXPECT_TRUE(d.stream() == nullptr);
            const uint64_t completed_before = internal::vulkan_completed_timeline_for_testing();
            Tensor x = Tensor::ones({16, 16}, Device::CUDA);
            const Tensor y = Tensor::full({16, 16}, 0.5f, Device::CUDA);
            for (int i = 0; i < 8; ++i) {
                x = x.add(y);
            }
            EXPECT_FLOAT_EQ(x.to_vector()[0], 5.0f);
            // Eight eager adds and one readback stay within a couple of submits.
            EXPECT_LE(internal::vulkan_completed_timeline_for_testing() - completed_before, 3u);
        }
        ASSERT_EQ(cudaStreamDestroy(stream), cudaSuccess);
    }

    TEST_F(TensorBackendValidators, DeferredViewsCarryTheBackendOfTheirSource) {
        // Catches a view deferral site that drops the tag of its deferred source.
        GpuBackendScope scope(GpuBackend::Vulkan);
        const Tensor base = Tensor::ones({64, 128}, Device::CUDA);
        const Tensor chain = base.add(1.0f);
        ASSERT_TRUE(chain.is_deferred());
        const Tensor permuted = chain.permute({1, 0});
        const Tensor sliced = chain.slice(0, 2, 10);
        const Tensor reshaped = chain.reshape({128, 64});
        const Tensor expanded = chain.unsqueeze(0).broadcast_to(TensorShape({3, 64, 128}));
        for (const Tensor* view : {&permuted, &sliced, &reshaped, &expanded}) {
            ASSERT_TRUE(view->is_deferred());
            EXPECT_EQ(gpu_backend_of(*view), GpuBackend::Vulkan);
        }
        EXPECT_FLOAT_EQ(permuted.to_vector()[0], 2.0f);
        EXPECT_FLOAT_EQ(sliced.to_vector()[0], 2.0f);
        EXPECT_FLOAT_EQ(reshaped.to_vector()[0], 2.0f);
        EXPECT_FLOAT_EQ(expanded.to_vector()[0], 2.0f);
        for (const Tensor* view : {&permuted, &sliced, &reshaped, &expanded}) {
            EXPECT_EQ(gpu_backend_of(*view), GpuBackend::Vulkan);
        }
    }

    TEST_F(TensorBackendValidators, DeferredTensorsMaterializeOnTheirLeafBackendOnAnyThread) {
        // Catches a materializer that allocates on the calling thread's default
        // backend instead of the leaf backend the tag promised.
        Tensor deferred;
        {
            GpuBackendScope scope(GpuBackend::Vulkan);
            deferred = Tensor::ones({512, 512}, Device::CUDA).add(0.5f).mul(2.0f);
        }
        ASSERT_TRUE(deferred.is_deferred());
        EXPECT_EQ(gpu_backend_of(deferred), GpuBackend::Vulkan);
        std::vector<float> values;
        std::thread worker([&] {
            EXPECT_EQ(default_gpu_backend(), GpuBackend::CUDA);
            values = deferred.to_vector();
        });
        worker.join();
        ASSERT_EQ(values.size(), 262144u);
        EXPECT_FLOAT_EQ(values[7], 3.0f);
        EXPECT_EQ(gpu_backend_of(deferred), GpuBackend::Vulkan);
    }

    TEST_F(TensorBackendValidators, InPlaceWriteOnALeafPreservesTheDeferredSnapshotBackend) {
        // Catches snapshot preservation cloning the leaf on the scope default
        // instead of the leaf backend.
        Tensor leaf;
        Tensor deferred;
        {
            GpuBackendScope scope(GpuBackend::Vulkan);
            leaf = Tensor::ones({512, 512}, Device::CUDA);
            deferred = leaf.add(1.0f);
        }
        leaf.mul_(10.0f);
        EXPECT_EQ(gpu_backend_of(deferred), GpuBackend::Vulkan);
        EXPECT_FLOAT_EQ(deferred.to_vector()[0], 2.0f);
        EXPECT_EQ(gpu_backend_of(deferred), GpuBackend::Vulkan);
        EXPECT_EQ(gpu_backend_of(leaf), GpuBackend::Vulkan);
    }

    TEST_F(TensorBackendValidators, ProcessDefaultIsReadOnceAndInvalidValuesAreRejected) {
        // Catches the selector re-reading LFS_TENSOR_BACKEND after the first
        // resolution, and an unknown value silently mapping to CUDA.
        const GpuBackend resolved = default_gpu_backend();
        setenv("LFS_TENSOR_BACKEND", resolved == GpuBackend::CUDA ? "vulkan" : "cuda", 1);
        EXPECT_EQ(default_gpu_backend(), resolved);
        unsetenv("LFS_TENSOR_BACKEND");
        const auto rejected = set_default_gpu_backend(resolved == GpuBackend::CUDA
                                                          ? GpuBackend::Vulkan
                                                          : GpuBackend::CUDA);
        EXPECT_FALSE(rejected.has_value()) << "changing the default after first use must fail";
    }

    TEST_F(TensorBackendValidators, ExternalOwnerStaysCudaUnderVulkanScope) {
        // Catches from_external_owner honouring the scope instead of its CUDA
        // pointer contract.
        Tensor owner_source;
        {
            GpuBackendScope scope(GpuBackend::CUDA);
            owner_source = Tensor::full({8}, 3.0f, Device::CUDA);
        }
        GpuBackendScope scope(GpuBackend::Vulkan);
        auto keep_alive = std::make_shared<Tensor>(owner_source);
        const Tensor external = Tensor::from_external_owner(
            owner_source.data_ptr(), {8}, Device::CUDA, DataType::Float32, keep_alive);
        EXPECT_EQ(gpu_backend_of(external), GpuBackend::CUDA);
        EXPECT_EQ(external.to_vector(), std::vector<float>(8, 3.0f));
    }

} // namespace
