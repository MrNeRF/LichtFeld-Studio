/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/tensor.hpp"
#include "core/tensor_backend.hpp"

#include <gtest/gtest.h>

#include <cstdlib>
#include <string>
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
