/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

// The Vulkan binding guard (vk::address in vk_ops_common.hpp) with its limit lowered to 1 MiB, so it runs on any
// device without a 4 GiB buffer. The guard is off on MoltenVK by default; each test installs the lowered limit after
// the context has started (a starting context sets the guard from the environment) and restores it afterwards.

#include "core/tensor.hpp"
#include "core/tensor/backend/vulkan/vk_context.hpp"
#include "core/tensor/backend/vulkan/vk_memory.hpp"
#include "core/tensor/backend/vulkan/vk_ops_common.hpp"
#include "core/tensor_backend.hpp"
#if LFS_BUILD_TRAINER
#include "lfs/training/ops/mcmc.hpp"
#include "lfs/training/ops/registry.hpp"
#endif

#include "cuda_backend_test.hpp"
#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <exception>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
    using lfs::core::DataType;
    using lfs::core::Device;
    using lfs::core::GpuBackend;
    using lfs::core::Tensor;
    namespace internal = lfs::core::internal;
    namespace vk = lfs::core::internal::vk;
#if LFS_BUILD_TRAINER
    namespace ops = lfs::gpu_ops;
#endif

    constexpr uint64_t kLimit = uint64_t{1} << 20;
    constexpr size_t kFloats = 2 * kLimit / sizeof(float); // a 2 MiB tensor

    // The guard's error, or an empty string when the call went through.
    template <class F>
    std::string refusal(F&& call) {
        try {
            call();
        } catch (const std::exception& error) {
            return error.what();
        }
        return {};
    }

    class TensorVulkanAddressGuard : public testing::Test {
    protected:
        void SetUp() override {
            if (!lfs::core::gpu_backend_available(GpuBackend::Vulkan))
                GTEST_SKIP() << "Vulkan device unavailable";
            session_.emplace(GpuBackend::Vulkan);
            ASSERT_TRUE(session_->switched());
            context_ = internal::acquire_vulkan_context();
            previous_ = internal::set_vulkan_address_guard_for_testing({.limit = kLimit, .report = false});
        }

        void TearDown() override {
            if (context_) {
                (void)internal::set_vulkan_address_guard_for_testing(previous_);
                context_.reset();
            }
            session_.reset();
        }

        std::optional<lfs::test::DefaultGpuBackendForTesting> session_;
        std::shared_ptr<internal::VulkanContext> context_;
        internal::VulkanAddressGuard previous_{};
    };

    TEST_F(TensorVulkanAddressGuard, RefusesABindingPastTheLimitAndNamesItsLauncher) {
        const Tensor values = Tensor::zeros({kFloats}, Device::GPU);
        const std::string error = refusal([&] { (void)vk::address(internal::storage_ref(values)); });
        ASSERT_FALSE(error.empty()) << "a 2 MiB binding passed a 1 MiB guard";
        EXPECT_NE(error.find("test_tensor_vulkan_address_guard.cpp"), std::string::npos) << error;
        EXPECT_NE(error.find("LFS_VULKAN_ADDRESS_GUARD=0"), std::string::npos) << error;
    }

    TEST_F(TensorVulkanAddressGuard, LetsWideBindingsAndBindingsWithinTheLimitThrough) {
        const Tensor values = Tensor::zeros({kFloats}, Device::GPU);
        const auto storage = internal::storage_ref(values);
        const uint64_t base = storage.meta->gpu_descriptor.base_address + storage.byte_offset;
        EXPECT_EQ(vk::wide_address(storage), base);
        const Tensor small = Tensor::zeros({kLimit / sizeof(float)}, Device::GPU);
        EXPECT_EQ(refusal([&] { (void)vk::address(internal::storage_ref(small)); }), "");
    }

    // A view reaches only its own elements, however large its storage: the case of a tensor in a reserved capacity.
    TEST_F(TensorVulkanAddressGuard, MeasuresAViewByItsElementsNotItsStorage) {
        const Tensor values = Tensor::zeros({4 * kFloats}, Device::GPU); // 8 MiB
        const size_t quarter = kLimit / sizeof(float) / 4;
        EXPECT_EQ(refusal([&] { (void)vk::address(internal::storage_ref(values.slice(0, 0, quarter))); }), "");
        EXPECT_EQ(refusal([&] {
                      (void)vk::address(internal::storage_ref(values.slice(0, 4 * kFloats - quarter, 4 * kFloats)));
                  }),
                  "");
        EXPECT_NE(refusal([&] { (void)vk::address(internal::storage_ref(values.slice(0, quarter, 4 * kFloats))); }), "");
        // A strided view reaches from its first element to its last.
        const Tensor rows = values.reshape({4 * kFloats / 1024, 1024});
        EXPECT_NE(refusal([&] { (void)vk::address(internal::storage_ref(rows.slice(1, 0, 1))); }), "");
        EXPECT_EQ(refusal([&] { (void)vk::address(internal::storage_ref(rows.slice(0, 0, 64).slice(1, 0, 1))); }), "");
    }

    // The reach shrinks with an offset moved by hand: 0.5 MiB of the view is left, though 6.5 MiB of storage is.
    TEST_F(TensorVulkanAddressGuard, FollowsAnOffsetMovedByHand) {
        const Tensor values = Tensor::zeros({4 * kFloats}, Device::GPU); // 8 MiB
        auto storage = internal::storage_ref(values.slice(0, 0, kFloats));
        EXPECT_NE(refusal([&] { (void)vk::address(storage); }), "");
        storage.byte_offset += kLimit + kLimit / 2;
        EXPECT_EQ(refusal([&] { (void)vk::address(storage); }), "");
    }

    // A ref without an end is let through, refused neither in the default mode nor in report mode.
    TEST_F(TensorVulkanAddressGuard, LetsARefOfUnknownExtentThrough) {
        const Tensor values = Tensor::zeros({kFloats}, Device::GPU);
        auto storage = internal::storage_ref(values);
        storage.byte_end = 0;
        EXPECT_EQ(refusal([&] { (void)vk::address(storage); }), "");
        (void)internal::set_vulkan_address_guard_for_testing({.limit = kLimit, .report = true});
        EXPECT_EQ(refusal([&] { (void)vk::address(storage); }), "");
    }

    TEST_F(TensorVulkanAddressGuard, ReportModeLogsInsteadOfThrowing) {
        (void)internal::set_vulkan_address_guard_for_testing({.limit = kLimit, .report = true});
        const Tensor values = Tensor::zeros({kFloats}, Device::GPU);
        EXPECT_EQ(refusal([&] { (void)vk::address(internal::storage_ref(values)); }), "");
    }

    TEST_F(TensorVulkanAddressGuard, ScratchCarriesItsExactSize) {
        const auto scratch = context_->memory().allocate(2 * kLimit, 16, {});
        EXPECT_EQ(scratch.byte_end, 2 * kLimit);
        EXPECT_NE(refusal([&] { (void)vk::address(scratch); }), "");
        context_->memory().deallocate(scratch);
    }

    TEST_F(TensorVulkanAddressGuard, OffTurnsItOff) {
        (void)internal::set_vulkan_address_guard_for_testing({});
        const Tensor values = Tensor::zeros({kFloats}, Device::GPU);
        EXPECT_EQ(refusal([&] { (void)vk::address(internal::storage_ref(values)); }), "");
    }

    // A launcher that binds each dispatch at its own piece of a buffer is measured by that piece; the address is the
    // ref's address plus the piece's offset, and a piece ends at the ref's end at the latest.
    TEST_F(TensorVulkanAddressGuard, MeasuresAPieceByItsOwnBytes) {
        const Tensor values = Tensor::zeros({kFloats}, Device::GPU); // 2 MiB
        const auto storage = internal::storage_ref(values);
        uint64_t piece = 0;
        EXPECT_EQ(refusal([&] { piece = vk::address(vk::storage_piece(storage, kLimit / 2, kLimit)); }), "");
        EXPECT_EQ(piece, vk::wide_address(storage) + kLimit / 2);
        EXPECT_NE(refusal([&] { (void)vk::address(vk::storage_piece(storage, 0, kLimit + sizeof(float))); }), "");
        // From 1.5 MiB, 2 MiB would end at 3.5 MiB: it ends at the ref's 2 MiB, so it reaches 0.5 MiB.
        EXPECT_EQ(refusal([&] { (void)vk::address(vk::storage_piece(storage, kLimit + kLimit / 2, 2 * kLimit)); }), "")
            << "the piece was measured past the ref's end";
    }

    // The fill is in the address form: past the limit it runs, contiguous (fill_ of floats, Tensor::full of int64)
    // and through a strided view, and every element is written.
    TEST_F(TensorVulkanAddressGuard, AFillPastTheLimitRuns) {
        Tensor floats, longs, pairs;
        EXPECT_EQ(refusal([&] {
                      floats = Tensor::empty({kFloats}, Device::GPU, DataType::Float32);
                      floats.fill_(0.5f);
                      floats = floats.cpu();
                      longs = Tensor::full({kFloats / 2}, -3.0f, Device::GPU, DataType::Int64).cpu();
                      pairs = Tensor::zeros({kFloats / 2, 2}, Device::GPU);
                      pairs.slice(1, 1, 2).fill_(2.5f);
                      pairs = pairs.cpu();
                  }),
                  "");
        ASSERT_EQ(floats.numel(), kFloats);
        ASSERT_EQ(longs.numel(), kFloats / 2);
        ASSERT_EQ(pairs.numel(), kFloats);
        size_t wrong = 0;
        for (size_t i = 0; i < kFloats; ++i) {
            wrong += floats.ptr<float>()[i] != 0.5f;
            wrong += pairs.ptr<float>()[i] != (i % 2 == 1 ? 2.5f : 0.0f);
            wrong += i < kFloats / 2 && longs.ptr<int64_t>()[i] != -3;
        }
        EXPECT_EQ(wrong, 0u);
    }

    // The scan is in the address form: a cumulative sum past the limit runs through its tile totals.
    TEST_F(TensorVulkanAddressGuard, ACumsumPastTheLimitRuns) {
        Tensor sums;
        EXPECT_EQ(refusal([&] { sums = Tensor::ones({kFloats}, Device::GPU).cumsum(0).cpu(); }), "");
        ASSERT_EQ(sums.numel(), kFloats);
        size_t wrong = 0;
        for (size_t i = 0; i < kFloats; ++i)
            wrong += sums.ptr<float>()[i] != static_cast<float>(i + 1);
        EXPECT_EQ(wrong, 0u);
    }

#if LFS_BUILD_TRAINER
    // A launcher in the address form takes bindings past the limit: the SH decode reads a 3 MiB fp32 store through a
    // wide binding.
    TEST_F(TensorVulkanAddressGuard, AnAddressFormLauncherRunsPastTheLimit) {
        const ops::ShOps* sh = lfs::training::training_ops(GpuBackend::Vulkan).sh;
        if (sh == nullptr)
            GTEST_SKIP() << "Sh slot is empty";
        constexpr uint32_t kRest = 15;
        constexpr size_t kRows = 16384; // 12 float4 slots a row: 3 MiB
        std::vector<float> host(kRows * 12 * 4);
        for (size_t i = 0; i < host.size(); ++i)
            host[i] = static_cast<float>(i % 1021);
        Tensor cpu = Tensor::empty({host.size()}, Device::CPU, DataType::Float32);
        ASSERT_EQ(cpu.bytes(), host.size() * sizeof(float));
        std::memcpy(cpu.data_ptr(), host.data(), cpu.bytes());
        const Tensor store = cpu.gpu();
        ASSERT_GT(store.bytes(), kLimit);
        Tensor decoded = Tensor::zeros({kRest * 3}, Device::GPU);
        const std::string error = refusal([&] {
            sh->decode_range(store, Tensor{}, decoded,
                             {.canonical_float_offset = (kRows - 1) * kRest * 3,
                              .float_count = kRest * 3,
                              .primitives = kRows,
                              .destination_rest = kRest,
                              .layout_rest = kRest,
                              .storage = ops::ShStorage::Float32});
            (void)decoded.cpu();
        });
        EXPECT_EQ(error, "");
        // The last row is lane 31 of its block of 32; its value c is in slot c / 4 of that lane.
        const Tensor result = decoded.cpu();
        const float* values = static_cast<const float*>(result.data_ptr());
        const size_t block = (kRows - 1) / 32, lane = (kRows - 1) % 32;
        for (uint32_t c = 0; c < kRest * 3; ++c)
            EXPECT_EQ(values[c], host[((block * 12 + c / 4) * 32 + lane) * 4 + c % 4]) << "cell " << c;
    }

    template <class T>
    Tensor device_copy(const std::vector<T>& values, const DataType dtype) {
        Tensor host = Tensor::empty({values.size()}, Device::CPU, dtype);
        std::memcpy(host.data_ptr(), values.data(), values.size() * sizeof(T));
        return host.gpu();
    }

    // MCMC's draw is in the address form: its weights, opacities, alive rows and cumulative sum pass the limit. The
    // only weight is on the last category, which is also the last alive row, so every draw must take it.
    TEST_F(TensorVulkanAddressGuard, MCMCsDrawPastTheLimitRuns) {
        const ops::McmcOps* mcmc = lfs::training::training_ops(GpuBackend::Vulkan).mcmc;
        if (mcmc == nullptr)
            GTEST_SKIP() << "MCMC slot is empty";
        constexpr size_t kCategories = kFloats; // 2 MiB of weights and of opacities
        constexpr size_t kSamples = 64;
        std::vector<float> host_weights(kCategories, 0.0f), host_opacity(kCategories, 0.0f);
        host_weights.back() = 2.0f;
        host_opacity.back() = 0.75f;
        std::vector<int64_t> host_alive(kCategories / 2); // 2 MiB of rows: 1, 3, ..., kCategories - 1
        for (size_t i = 0; i < host_alive.size(); ++i)
            host_alive[i] = static_cast<int64_t>(2 * i + 1);
        const Tensor weights = device_copy(host_weights, DataType::Float32);
        const Tensor opacity = device_copy(host_opacity, DataType::Float32);
        const Tensor alive = device_copy(host_alive, DataType::Int64);
        const Tensor scales = Tensor::zeros({kCategories, 3}, Device::GPU);
        for (const bool by_alive : {false, true}) {
            Tensor indices = Tensor::empty({kSamples}, Device::GPU, DataType::Int64);
            Tensor sampled_opacity = Tensor::empty({kSamples}, Device::GPU, DataType::Float32);
            Tensor sampled_scales = Tensor::empty({kSamples, 3}, Device::GPU, DataType::Float32);
            Tensor drawn, drawn_opacity, drawn_scales;
            EXPECT_EQ(refusal([&] {
                          mcmc->sample(weights, opacity, scales, by_alive ? alive : Tensor{}, indices, sampled_opacity,
                                       sampled_scales,
                                       by_alive ? ops::SampleDomain::AliveIndices : ops::SampleDomain::All, 4202);
                          drawn = indices.cpu();
                          drawn_opacity = sampled_opacity.cpu();
                          drawn_scales = sampled_scales.cpu();
                      }),
                      "")
                << (by_alive ? "alive rows" : "all rows");
            ASSERT_EQ(drawn.numel(), kSamples);
            size_t wrong = 0;
            for (size_t i = 0; i < kSamples; ++i) {
                wrong += drawn.ptr<int64_t>()[i] != static_cast<int64_t>(kCategories - 1);
                wrong += drawn_opacity.ptr<float>()[i] != 0.75f;
                for (size_t d = 0; d < 3; ++d)
                    wrong += std::fabs(drawn_scales.ptr<float>()[i * 3 + d] - 1.0f) > 1e-6f; // exp(0)
            }
            EXPECT_EQ(wrong, 0u) << (by_alive ? "alive rows" : "all rows");
        }
    }
#endif
} // namespace
