/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/cuda_error.hpp"
#include "core/memory_pressure.hpp"
#include "core/tensor.hpp"

#include <gtest/gtest.h>

#include <atomic>
#include <string>
#include <vector>

using namespace lfs::core;

namespace {

    AllocationFailure device_failure(size_t requested = 0) {
        return AllocationFailure{
            .domain = MemoryDomain::CudaDevice,
            .requested_bytes = requested,
            .alignment = 0,
            .device = 0,
            .stream = 0,
            .label = "test",
            .operation = "test.alloc",
            .native_error = 0,
        };
    }

} // namespace

class MemoryPressureTest : public ::testing::Test {
protected:
    void SetUp() override {
        MemoryPressureCoordinator::instance().reset_for_testing();
    }
    void TearDown() override {
        MemoryPressureCoordinator::instance().reset_for_testing();
    }

    MemoryPressureCoordinator& coordinator() { return MemoryPressureCoordinator::instance(); }
};

class MemoryPressureCudaUnavailableTest : public ::testing::Test {
protected:
    void SetUp() override {
        reset_cuda_diagnostics_for_testing();
        MemoryPressureCoordinator::instance().reset_for_testing();
    }

    void TearDown() override {
        MemoryPressureCoordinator::instance().reset_for_testing();
        reset_cuda_diagnostics_for_testing();
    }
};

TEST_F(MemoryPressureCudaUnavailableTest, LatchedAllocationFailsFastTyped) {
    int device_count = 0;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
        GTEST_SKIP() << "no CUDA device";
    }

    auto& pressure = MemoryPressureCoordinator::instance();
    pressure.set_allocation_probe(
        [](const MemoryDomain domain, size_t) { return domain == MemoryDomain::CudaDevice; });
    ASSERT_TRUE(latch_cuda_unavailable(cudaErrorInitializationError));
    const uint64_t episodes_before = pressure.episode_count();

    EXPECT_THROW(Tensor::zeros({1024}, Device::CUDA), MemoryAllocationError);
    EXPECT_EQ(pressure.episode_count(), episodes_before);
}

TEST_F(MemoryPressureTest, TypedErrorCarriesMetadata) {
    AllocationFailure failure = device_failure(1234);
    MemoryAllocationError error(failure);
    EXPECT_EQ(error.domain(), MemoryDomain::CudaDevice);
    EXPECT_EQ(error.requested_bytes(), 1234u);
    EXPECT_NE(std::string(error.what()).find("out of memory"), std::string::npos);
}

TEST_F(MemoryPressureTest, ClientsReclaimInPriorityOrder) {
    std::vector<int> order;
    coordinator().set_free_memory_probe([](MemoryDomain) -> size_t { return 0; });
    for (int priority : {30, 10, 20}) {
        coordinator().register_client(PressureClient{
            .name = "c",
            .priority = priority,
            .domain = MemoryDomain::CudaDevice,
            .affinity = PressureAffinity::ImmediateThreadSafe,
            .estimate = nullptr,
            .shrink = [&order, priority](const PressureRequest&) {
                order.push_back(priority);
                return ReclaimResult{};
            },
        });
    }
    coordinator().run_episode(device_failure(0), PressureContext::ImmediateOnly);
    ASSERT_EQ(order.size(), 3u);
    EXPECT_EQ(order[0], 10);
    EXPECT_EQ(order[1], 20);
    EXPECT_EQ(order[2], 30);
}

TEST_F(MemoryPressureTest, StopsReclaimingOnceTargetIsMet) {
    const size_t target = coordinator().reserve_bytes(); // requested == 0
    std::atomic<size_t> fake_free{0};
    coordinator().set_free_memory_probe([&fake_free](MemoryDomain) { return fake_free.load(); });

    std::vector<int> ran;
    coordinator().register_client(PressureClient{
        .name = "high-priority",
        .priority = 10,
        .domain = MemoryDomain::CudaDevice,
        .affinity = PressureAffinity::ImmediateThreadSafe,
        .estimate = nullptr,
        .shrink = [&](const PressureRequest&) {
            ran.push_back(10);
            fake_free.store(target); // relieves the pressure
            return ReclaimResult{};
        },
    });
    coordinator().register_client(PressureClient{
        .name = "low-priority",
        .priority = 20,
        .domain = MemoryDomain::CudaDevice,
        .affinity = PressureAffinity::ImmediateThreadSafe,
        .estimate = nullptr,
        .shrink = [&](const PressureRequest&) {
            ran.push_back(20);
            return ReclaimResult{};
        },
    });

    coordinator().run_episode(device_failure(0), PressureContext::ImmediateOnly);
    ASSERT_EQ(ran.size(), 1u);
    EXPECT_EQ(ran[0], 10);
}

TEST_F(MemoryPressureTest, NoProgressTerminatesAndReportsNoBytes) {
    coordinator().set_free_memory_probe([](MemoryDomain) -> size_t { return 0; });
    int calls = 0;
    coordinator().register_client(PressureClient{
        .name = "unable",
        .priority = 10,
        .domain = MemoryDomain::CudaDevice,
        .affinity = PressureAffinity::ImmediateThreadSafe,
        .estimate = nullptr,
        .shrink = [&calls](const PressureRequest&) {
            ++calls;
            return ReclaimResult{};
        },
    });
    const size_t freed = coordinator().run_episode(device_failure(0), PressureContext::ImmediateOnly);
    EXPECT_EQ(calls, 1);
    EXPECT_EQ(freed, 0u);
}

TEST_F(MemoryPressureTest, AbundantMemoryRecordsNoEpisode) {
    const size_t target = coordinator().reserve_bytes();
    coordinator().set_free_memory_probe([target](MemoryDomain) { return target + 1; });
    int calls = 0;
    coordinator().register_client(PressureClient{
        .name = "unused",
        .priority = 10,
        .domain = MemoryDomain::CudaDevice,
        .affinity = PressureAffinity::ImmediateThreadSafe,
        .estimate = nullptr,
        .shrink = [&calls](const PressureRequest&) { ++calls; return ReclaimResult{}; },
    });
    coordinator().run_episode(device_failure(0), PressureContext::ImmediateOnly);
    EXPECT_EQ(calls, 0);
    EXPECT_EQ(coordinator().episode_count(), 0u);
    EXPECT_FALSE(coordinator().pressure_active());
}

TEST_F(MemoryPressureTest, ReentrantEpisodeIsRefused) {
    coordinator().set_free_memory_probe([](MemoryDomain) -> size_t { return 0; });
    std::vector<std::string> ran;
    size_t inner_result = 12345;
    coordinator().register_client(PressureClient{
        .name = "reenters",
        .priority = 10,
        .domain = MemoryDomain::CudaDevice,
        .affinity = PressureAffinity::ImmediateThreadSafe,
        .estimate = nullptr,
        .shrink = [&](const PressureRequest&) {
            ran.push_back("A");
            inner_result = MemoryPressureCoordinator::instance().run_episode(
                device_failure(0), PressureContext::ImmediateOnly);
            return ReclaimResult{};
        },
    });
    coordinator().register_client(PressureClient{
        .name = "second",
        .priority = 20,
        .domain = MemoryDomain::CudaDevice,
        .affinity = PressureAffinity::ImmediateThreadSafe,
        .estimate = nullptr,
        .shrink = [&](const PressureRequest&) {
            ran.push_back("B");
            return ReclaimResult{};
        },
    });
    coordinator().run_episode(device_failure(0), PressureContext::ImmediateOnly);
    EXPECT_EQ(inner_result, 0u); // reentry refused
    ASSERT_EQ(ran.size(), 2u);   // no recursive re-run of the registry
    EXPECT_EQ(coordinator().episode_count(), 1u);
}

TEST_F(MemoryPressureTest, FailingClientDoesNotAbortEpisode) {
    coordinator().set_free_memory_probe([](MemoryDomain) -> size_t { return 0; });
    bool second_ran = false;
    coordinator().register_client(PressureClient{
        .name = "throws",
        .priority = 10,
        .domain = MemoryDomain::CudaDevice,
        .affinity = PressureAffinity::ImmediateThreadSafe,
        .estimate = nullptr,
        .shrink = [](const PressureRequest&) -> ReclaimResult {
            throw std::runtime_error("client failure");
        },
    });
    coordinator().register_client(PressureClient{
        .name = "survivor",
        .priority = 20,
        .domain = MemoryDomain::CudaDevice,
        .affinity = PressureAffinity::ImmediateThreadSafe,
        .estimate = nullptr,
        .shrink = [&second_ran](const PressureRequest&) {
            second_ran = true;
            return ReclaimResult{};
        },
    });
    EXPECT_NO_THROW(coordinator().run_episode(device_failure(0), PressureContext::ImmediateOnly));
    EXPECT_TRUE(second_ran);
}

TEST_F(MemoryPressureTest, HostClientsDoNotRunForDeviceFailure) {
    coordinator().set_free_memory_probe([](MemoryDomain) -> size_t { return 0; });
    int host_calls = 0;
    coordinator().register_client(PressureClient{
        .name = "pinned",
        .priority = 10,
        .domain = MemoryDomain::PinnedHost,
        .affinity = PressureAffinity::ImmediateThreadSafe,
        .estimate = nullptr,
        .shrink = [&host_calls](const PressureRequest&) { ++host_calls; return ReclaimResult{}; },
    });

    coordinator().run_episode(device_failure(0), PressureContext::ImmediateOnly);
    EXPECT_EQ(host_calls, 0);

    AllocationFailure host = device_failure(0);
    host.domain = MemoryDomain::PinnedHost;
    coordinator().run_episode(host, PressureContext::ImmediateOnly);
    EXPECT_EQ(host_calls, 1);
}

TEST_F(MemoryPressureTest, ContextGatesRenderSafeClients) {
    coordinator().set_free_memory_probe([](MemoryDomain) -> size_t { return 0; });
    int calls = 0;
    coordinator().register_client(PressureClient{
        .name = "render-safe",
        .priority = 10,
        .domain = MemoryDomain::CudaDevice,
        .affinity = PressureAffinity::RenderSafePoint,
        .estimate = nullptr,
        .shrink = [&calls](const PressureRequest&) { ++calls; return ReclaimResult{}; },
    });

    coordinator().run_episode(device_failure(0), PressureContext::ImmediateOnly);
    EXPECT_EQ(calls, 0);

    coordinator().run_episode(device_failure(0), PressureContext::RenderThread);
    EXPECT_EQ(calls, 1);
}

TEST_F(MemoryPressureTest, PreflightUsesReclaimableEstimates) {
    const size_t reserve = coordinator().reserve_bytes();
    coordinator().set_free_memory_probe([](MemoryDomain) -> size_t { return 100; });
    coordinator().register_client(PressureClient{
        .name = "estimator",
        .priority = 10,
        .domain = MemoryDomain::CudaDevice,
        .affinity = PressureAffinity::ImmediateThreadSafe,
        .estimate = [](const PressureRequest&) -> size_t { return 100; },
        .shrink = [](const PressureRequest&) { return ReclaimResult{}; },
    });

    OperationMemoryPlan tight;
    tight.persistent_device_bytes = 300; // required = 300 + reserve; free+reclaim = 200
    EXPECT_FALSE(coordinator().preflight(tight, MemoryDomain::CudaDevice).ok);

    OperationMemoryPlan loose;
    loose.persistent_device_bytes = 0; // required = reserve; but effective_free only 200
    const PreflightResult loose_result = coordinator().preflight(loose, MemoryDomain::CudaDevice);
    EXPECT_EQ(loose_result.reclaimable_bytes, 100u);
    EXPECT_EQ(loose_result.effective_free_bytes, 200u);
    EXPECT_EQ(loose_result.required_peak_bytes, reserve);
}

TEST_F(MemoryPressureTest, PressureLeaseReleasesOnSustainedHeadroom) {
    const size_t target = coordinator().reserve_bytes();
    std::atomic<size_t> fake_free{0};
    coordinator().set_free_memory_probe([&fake_free](MemoryDomain) { return fake_free.load(); });

    coordinator().run_episode(device_failure(0), PressureContext::ImmediateOnly);
    EXPECT_TRUE(coordinator().pressure_active());

    fake_free.store(target * 2); // well above the 20% hysteresis
    coordinator().maybe_recover();
    EXPECT_FALSE(coordinator().pressure_active());
}

TEST_F(MemoryPressureTest, InjectedFailureThrowsTypedErrorFromTensorPath) {
    int device_count = 0;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
        GTEST_SKIP() << "no CUDA device";
    }
    constexpr size_t kThreshold = 64ull * 1024 * 1024;
    coordinator().set_allocation_probe(
        [](MemoryDomain domain, size_t bytes) {
            return domain == MemoryDomain::CudaDevice && bytes >= kThreshold;
        });

    // 128 MiB of Float32 storage: over the injected threshold, so both the
    // initial attempt and the post-reclaim retry fail, surfacing a typed error
    // with no null pointer reaching a kernel.
    EXPECT_THROW(Tensor::zeros({32 * 1024 * 1024}, Device::CUDA), MemoryAllocationError);

    coordinator().set_allocation_probe(nullptr);
    EXPECT_NO_THROW({
        Tensor recovered = Tensor::zeros({32 * 1024 * 1024}, Device::CUDA);
        EXPECT_EQ(recovered.numel(), 32u * 1024 * 1024);
    });
}
