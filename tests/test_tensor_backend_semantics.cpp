/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

// Edge-case semantics every tensor backend must share: scalars, empty
// operands, C pow, unsigned and half dtypes, saturating casts and fault
// reporting. Each case runs on the CPU and on every available GPU backend.

#include "core/error.hpp"
#include "core/gpu_device_runtime.hpp"
#include "core/nn/ops.hpp"
#include "core/tensor.hpp"
#include "core/tensor/backend/backend_driver.hpp"
#include "core/tensor/backend/deferred_target.hpp"
#include "core/tensor/backend/gpu_backend_ops.hpp"
#include "core/tensor/backend/native_allocation.hpp"
#include "core/tensor/internal/expression_runtime.hpp"
#include "core/tensor/internal/lazy_executor.hpp"
#include "core/tensor_backend.hpp"
#include "core/tensor_cuda_interop.hpp"
#include "core/tensor_execution.hpp"
#include "core/tensor_fused.hpp"
#include "core/tensor_upload.hpp"
#include "tensor_backend_contract.hpp"

#include <gtest/gtest.h>

#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <format>
#include <functional>
#include <future>
#include <iostream>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace {

    using lfs::core::DataType;
    using lfs::core::Device;
    using lfs::core::GpuBackend;
    using lfs::core::Tensor;
    using lfs::core::TensorShape;

    constexpr float kInf = std::numeric_limits<float>::infinity();
    constexpr float kNan = std::numeric_limits<float>::quiet_NaN();

    struct DeferredSizeGuard {
        DeferredSizeGuard() { lfs::core::internal::lazy_executor_set_size_heuristic_override_for_testing(false); }
        ~DeferredSizeGuard() { lfs::core::internal::lazy_executor_set_size_heuristic_override_for_testing(std::nullopt); }
    };

    // Exercise failure/retry through the regular expression conversion, without
    // adding a production API that mutates the private materializer state.
    struct DeferredContractExpr : lfs::core::TensorExpr<DeferredContractExpr> {
        Tensor value;
        std::function<Tensor()> evaluate;
        DeferredContractExpr(Tensor tensor, std::function<Tensor()> callback)
            : value(std::move(tensor)), evaluate(std::move(callback)) {}
        const TensorShape& shape_impl() const { return value.shape(); }
        Device device_impl() const { return value.device(); }
        DataType dtype_impl() const { return value.dtype(); }
        auto gpu_backend_impl() const { return lfs::core::gpu_backend_of(value); }
        auto stream_hint_impl() const { return value.stream(); }
        DeferredContractExpr snapshot_impl() const { return *this; }
        Tensor eval_impl() const { return evaluate(); }
    };

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

    class TensorBackendSemantics : public lfs::test::TensorBackendContract {
    protected:
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

    TEST_P(TensorBackendSemantics, AllocatedViewRetainsStorageAfterOwnerDestruction) {
        Tensor view;
        const void* storage = nullptr;
        {
            auto owner = Tensor::full({4}, 7.f, device());
            storage = owner.storage_ptr();
            view = owner.slice(0, 1, 3);
            EXPECT_EQ(view.storage_ptr(), storage);
        }
        // A fresh allocation must not reuse the still-live view's storage.
        const auto other = Tensor::full({4}, 11.f, device());
        expect_floats(host<float>(view), {7, 7});
        EXPECT_EQ(view.storage_ptr(), storage);
        view.fill_(9.f);
        expect_floats(host<float>(view), {9, 9});
        expect_floats(host<float>(other), {11, 11, 11, 11});
    }

    TEST_P(TensorBackendSemantics, MovedAllocationRetainsStorageAfterAliasesAreReleased) {
        auto original = Tensor::full({3}, 5.f, device());
        auto alias = original;
        auto moved = std::move(original);
        auto independent = moved.clone();
        alias = Tensor{};
        original = Tensor{};
        moved.fill_(8.f);
        expect_floats(host<float>(moved), {8, 8, 8});
        moved = Tensor{};
        expect_floats(host<float>(independent), {5, 5, 5});
    }

    TEST_P(TensorBackendSemantics, NullHomeNeedsNoCanonicalizationOrImplicitBinding) {
        using namespace lfs::core;
        const auto tensor = Tensor::full({1}, 1.f, Device::CPU);
        for (const auto backend : kGpuBackends) {
            auto& driver = internal::backend_driver(backend);
            EXPECT_EQ(driver.canonical_home_target(nullptr), nullptr);
            EXPECT_FALSE(driver.bind_tensor_target(tensor, true).has_value());
            EXPECT_EQ(internal::canonical_deferred_home(backend, nullptr), nullptr);
            EXPECT_FALSE(internal::deferred_home_needs_binding(backend, nullptr));
            const auto marker = reinterpret_cast<void*>(uintptr_t{1});
            EXPECT_EQ(internal::canonical_deferred_home(backend, marker),
                      driver.canonical_home_target(marker));
        }
    }

#if LFS_HAS_CUDA
    TEST_P(TensorBackendSemantics, DeferredCudaHomePreservesAnExplicitCurrentStream) {
        using namespace lfs::core;
        if (GetParam().backend != GpuBackend::CUDA)
            GTEST_SKIP() << "CUDA implicit home policy";
        const DeferredSizeGuard guard;
        TensorCudaStream home;
        TensorCudaStream current;
        Tensor deferred;
        {
            const CUDAStreamGuard scope(home.get());
            deferred = Tensor::full({3}, 2.f, Device::GPU).add(1.f).mul(2.f).sub(3.f);
        }
        ASSERT_TRUE(deferred.is_deferred());
        ASSERT_EQ(deferred.stream(), home.get());
        EXPECT_TRUE(internal::deferred_home_needs_binding(GpuBackend::CUDA, home.get()));
        {
            const CUDAStreamGuard scope(current.get());
            EXPECT_FALSE(internal::deferred_home_needs_binding(GpuBackend::CUDA, home.get()));
            auto& driver = internal::backend_driver(GpuBackend::CUDA);
            EXPECT_FALSE(driver.bind_tensor_target(deferred, true).has_value());
            auto alias = deferred;
            (void)deferred.data_ptr();
            EXPECT_EQ(getCurrentCUDAStream(), current.get());
            EXPECT_EQ(deferred.stream(), current.get());
            expect_floats(host<float>(alias), {3, 3, 3});
            EXPECT_EQ(alias.stream(), current.get());
        }
        EXPECT_EQ(getCurrentCUDAStream(), nullptr);
    }

    TEST_P(TensorBackendSemantics, DeferredCudaHomeRestoresAfterMaterializerFailure) {
        using namespace lfs::core;
        if (GetParam().backend != GpuBackend::CUDA)
            GTEST_SKIP() << "CUDA implicit home policy";
        const DeferredSizeGuard guard;
        TensorCudaStream home;
        Tensor deferred;
        const auto attempts = std::make_shared<int>(0);
        {
            const CUDAStreamGuard scope(home.get());
            auto value = Tensor::full({3}, 2.f, Device::GPU);
            deferred = DeferredContractExpr(value, [value, attempts, target = home.get()] {
                EXPECT_EQ(getCurrentCUDAStream(), target);
                if (++*attempts == 1)
                    throw std::runtime_error("CUDA deferred contract retry");
                return value;
            });
        }
        auto alias = deferred;
        EXPECT_THROW((void)deferred.data_ptr(), std::runtime_error);
        EXPECT_EQ(getCurrentCUDAStream(), nullptr);
        expect_floats(host<float>(alias), {2, 2, 2});
        EXPECT_EQ(getCurrentCUDAStream(), nullptr);
        EXPECT_EQ(*attempts, 2);
    }

    TEST_P(TensorBackendSemantics, DeferredCudaRetiredHomeDoesNotBindOrSurvivePublication) {
        using namespace lfs::core;
        if (GetParam().backend != GpuBackend::CUDA)
            GTEST_SKIP() << "CUDA implicit home policy";
        const DeferredSizeGuard guard;
        Tensor deferred;
        cudaStream_t retired = nullptr;
        {
            TensorCudaStream home;
            retired = home.get();
            const CUDAStreamGuard scope(retired);
            deferred = Tensor::full({3}, 2.f, Device::GPU).add(1.f).mul(2.f).sub(3.f);
        }
        ASSERT_TRUE(deferred.is_deferred());
        ASSERT_TRUE(is_stream_retired(retired));
        EXPECT_FALSE(internal::deferred_home_needs_binding(GpuBackend::CUDA, retired));
        EXPECT_FALSE(internal::backend_driver(GpuBackend::CUDA).bind_tensor_target(deferred, true).has_value());
        auto alias = deferred;
        expect_floats(host<float>(deferred), {3, 3, 3});
        expect_floats(host<float>(alias), {3, 3, 3});
        EXPECT_EQ(deferred.stream(), nullptr);
        EXPECT_EQ(alias.stream(), nullptr);
        EXPECT_EQ(getCurrentCUDAStream(), nullptr);
    }
#endif

    TEST_P(TensorBackendSemantics, ConcurrentActivationAndCacheResetPreserveUsableAdapters) {
        using namespace lfs::core;
        if (!GetParam().backend)
            GTEST_SKIP() << "Native GPU activation";
        const auto backend = *GetParam().backend;
        auto* original = &internal::backend_ops(backend);
        for (int round = 0; round < 2; ++round) {
            internal::clear_backend_activation_cache();
            // Destroy the promise before joining workers if launching one throws.
            // wait() also releases on a broken promise, so no worker is stranded.
            std::array<std::future<internal::GpuBackendOps*>, 4> workers;
            std::promise<void> start;
            const auto ready = start.get_future().share();
            for (auto& worker : workers) {
                worker = std::async(std::launch::async, [backend, ready] {
                    ready.wait();
                    auto& ops = internal::backend_ops(backend);
                    ops.synchronize_device();
                    return &ops;
                });
            }
            start.set_value();
            for (auto& worker : workers)
                EXPECT_EQ(worker.get(), original);
        }
        expect_floats(host<float>(make<float>(DataType::Float32, {2}, {2, 3}).add(1.f)), {3, 4});
    }

    TEST_P(TensorBackendSemantics, DeferredAliasesPublishOnceWithNoHomeTarget) {
        const DeferredSizeGuard guard;
        const auto source = make<float>(DataType::Float32, {3}, {2, 3, 4});
        auto deferred = source.add(1.f).mul(2.f).sub(3.f);
        ASSERT_TRUE(deferred.is_deferred());
        std::array<Tensor, 4> aliases{deferred, deferred, deferred, deferred};
        EXPECT_EQ(deferred.stream(), nullptr);
        const auto* storage = deferred.data_ptr();
        for (auto& alias : aliases) {
            EXPECT_EQ(alias.data_ptr(), storage);
            EXPECT_EQ(alias.stream(), nullptr);
            expect_placement(alias);
            expect_floats(host<float>(alias), {3, 5, 7});
        }
    }

    TEST_P(TensorBackendSemantics, DeferredFailureReleasesTheGateAndPreservesSiblingRetry) {
        const DeferredSizeGuard guard;
        const auto value = make<float>(DataType::Float32, {3}, {2, 3, 4});
        const auto attempts = std::make_shared<int>(0);
        Tensor deferred = DeferredContractExpr(value, [value, attempts] {
            if (++*attempts == 1)
                throw std::runtime_error("deferred contract retry");
            return value;
        });
        auto alias = deferred;
        EXPECT_THROW((void)deferred.data_ptr(), std::runtime_error);
        EXPECT_TRUE(deferred.is_deferred());
        EXPECT_TRUE(alias.is_deferred());
        expect_floats(host<float>(alias), {2, 3, 4});
        expect_floats(host<float>(deferred), {2, 3, 4});
        EXPECT_EQ(*attempts, 2);
        EXPECT_EQ(alias.storage_ptr(), deferred.storage_ptr());
    }

    TEST_P(TensorBackendSemantics, ConcurrentDeferredAliasesShareOnePublishedResult) {
        const DeferredSizeGuard guard;
        for (int round = 0; round < 4; ++round) {
            auto deferred = make<float>(DataType::Float32, {3}, {2, 3, 4}).add(1.f);
            ASSERT_TRUE(deferred.is_deferred());
            std::array<std::future<Tensor>, 4> workers;
            std::promise<void> start;
            const auto ready = start.get_future().share();
            const auto before = lfs::core::internal::lazy_executor_diagnostics_snapshot_for_testing();
            for (auto& worker : workers) {
                worker = std::async(std::launch::async, [alias = deferred, ready]() mutable {
                    ready.wait();
                    (void)std::as_const(alias).data_ptr();
                    return alias;
                });
            }
            start.set_value();
            const void* storage = nullptr;
            for (auto& worker : workers) {
                auto alias = worker.get();
                if (!storage)
                    storage = alias.storage_ptr();
                EXPECT_EQ(alias.storage_ptr(), storage);
                expect_placement(alias);
                expect_floats(host<float>(alias), {3, 4, 5});
            }
            const auto after = lfs::core::internal::lazy_executor_diagnostics_snapshot_for_testing();
            EXPECT_EQ(after.executed_nodes - before.executed_nodes, 1u);
            EXPECT_EQ(std::as_const(deferred).data_ptr(), storage);
        }
    }

    TEST_P(TensorBackendSemantics, DeferredAliasesRetainHomeOutsideTheProducerScope) {
        using namespace lfs::core;
        if (!GetParam().backend)
            GTEST_SKIP() << "Native GPU execution target";
        const DeferredSizeGuard guard;
        TensorWorkQueue queue(*GetParam().backend);
        Tensor source;
        Tensor deferred;
        {
            const TensorWorkQueue::Scope scope(queue);
            source = Tensor::full({3}, 2.f, Device::GPU);
            deferred = source.add(1.f).mul(2.f).sub(3.f);
        }
        ASSERT_TRUE(deferred.is_deferred());
        const auto home = source.execution_target();
        std::array<Tensor, 4> aliases{deferred, deferred, deferred, deferred};
        const auto* storage = deferred.data_ptr();
        EXPECT_EQ(deferred.execution_target(), home);
        for (auto& alias : aliases) {
            EXPECT_EQ(alias.data_ptr(), storage);
            EXPECT_EQ(alias.execution_target(), home);
            expect_floats(host<float>(alias), {3, 3, 3});
        }
        queue.wait();
    }

    TEST_P(TensorBackendSemantics, DeferredAliasOutlivesTheFirstPublishedHandle) {
        const DeferredSizeGuard guard;
        Tensor pending;
        const void* storage = nullptr;
        {
            const auto source = make<float>(DataType::Float32, {3}, {2, 3, 4});
            auto deferred = source.add(1.f).mul(2.f).sub(3.f);
            ASSERT_TRUE(deferred.is_deferred());
            pending = deferred;
            storage = deferred.data_ptr();
            ASSERT_TRUE(pending.is_deferred());
        }
        // Publication must transfer only its local ownership snapshot, never
        // steal the cached result needed by a still-deferred sibling.
        const auto other = Tensor::full({3}, 11.f, device());
        EXPECT_NE(other.storage_ptr(), storage);
        EXPECT_EQ(std::as_const(pending).data_ptr(), storage);
        expect_placement(pending);
        expect_floats(host<float>(pending), {3, 5, 7});
        auto view = pending.slice(0, 1, 3);
        pending = Tensor{};
        expect_floats(host<float>(view), {5, 7});
        view.fill_(9.f);
        expect_floats(host<float>(view), {9, 9});
        expect_floats(host<float>(other), {11, 11, 11});
    }

    TEST_P(TensorBackendSemantics, UncompiledDriversDoNotAdvertiseNativeTargetsOrOwnership) {
        using namespace lfs::core;
        for (const auto backend : kGpuBackends) {
            if (gpu_backend_compiled(backend))
                continue;
            auto& driver = internal::backend_driver(backend);
            EXPECT_EQ(driver.current_target(), nullptr);
            EXPECT_EQ(driver.canonical_home_target(reinterpret_cast<void*>(uintptr_t{1})), nullptr);
            const auto access = driver.renderer_access();
            EXPECT_FALSE(access.shares_vulkan);
            EXPECT_FALSE(access.needs_vulkan_interop);
            EXPECT_FALSE(access.metal);
            EXPECT_FALSE(driver.needs_thread_binding());
            EXPECT_THROW((void)driver.selected_device(), TensorError);
            EXPECT_THROW((void)driver.allocation_bytes(16), TensorError);
            const auto empty = driver.empty_tensor_allocation(internal::TensorAllocationKind::Direct);
            EXPECT_EQ(empty.storage.backend, backend);
            EXPECT_EQ(empty.storage.data, nullptr);
            EXPECT_EQ(empty.target, nullptr);
            EXPECT_FALSE(empty.owner);
            EXPECT_STREQ(empty.external_kind, "");
        }
        if constexpr (!gpu_backend_compiled(GpuBackend::CUDA))
            EXPECT_EQ(selected_cuda_device(), 0);
    }

    TEST_P(TensorBackendSemantics, CachedWorkerReopensAfterAnotherThreadShutsDown) {
        if (!GetParam().backend)
            GTEST_SKIP() << "GPU context lifecycle";
        const auto backend = *GetParam().backend;
        std::promise<std::exception_ptr> ready;
        auto initialized = ready.get_future();
        std::promise<void> resume;
        auto resumed = resume.get_future();
        auto worker = std::async(std::launch::async, [&] {
            bool announced = false;
            try {
                auto& cached = lfs::core::internal::backend_ops(backend);
                cached.synchronize_device();
                ready.set_value(nullptr);
                announced = true;
                resumed.wait();
                // Synchronization, not allocation or a new scope, must reopen
                // the worker's cached adapter after the other thread shuts down.
                cached.synchronize_device();
                cached.device_barrier();
                cached.synchronize_stream({});
                lfs::core::gpu_device_barrier(backend);
                return std::exception_ptr{};
            } catch (...) {
                if (!announced)
                    ready.set_value(std::current_exception());
                return std::current_exception();
            }
        });
        const auto initialization_error = initialized.get();
        const auto shutdown = initialization_error ? lfs::Status{} : lfs::core::shutdown_gpu_backend(backend);
        // Always release/join the worker before assertions can exit the test.
        resume.set_value();
        const auto synchronization_error = worker.get();
        if (initialization_error)
            EXPECT_NO_THROW(std::rethrow_exception(initialization_error));
        EXPECT_TRUE(shutdown);
        if (synchronization_error)
            EXPECT_NO_THROW(std::rethrow_exception(synchronization_error));
        EXPECT_NO_THROW(Tensor::full({3}, 7.f, Device::GPU).cpu());
    }

    TEST_P(TensorBackendSemantics, EmptyDirectAllocationUsesItsNativeLabel) {
        if (!GetParam().backend)
            GTEST_SKIP() << "GPU allocation metadata";
        const auto backend = *GetParam().backend;
        const auto allocation = lfs::core::internal::backend_driver(backend).empty_tensor_allocation(
            lfs::core::internal::TensorAllocationKind::Direct);
        EXPECT_EQ(allocation.storage.backend, backend);
        EXPECT_EQ(allocation.storage.data, nullptr);
        EXPECT_STREQ(allocation.external_kind, lfs::core::internal::allocation_label(allocation.storage));
    }

    TEST_P(TensorBackendSemantics, ExpressionCacheRejectsAnInvalidFreshKernel) {
        if (GetParam().backend)
            GTEST_SKIP() << "Device-independent expression cache contract";
        lfs::core::internal::ExpressionCache cache;
        const auto key = "invalid-fresh-contract-" + std::to_string(
                                                         std::chrono::steady_clock::now().time_since_epoch().count());
        EXPECT_THROW(cache.get(key, [] { return std::vector<char>{'x'}; }, [](std::span<const char>) { return std::shared_ptr<lfs::core::internal::CompiledExpression>{}; }), std::runtime_error);
        EXPECT_EQ(cache.stats().entries, 0);
    }

    TEST_P(TensorBackendSemantics, CheckedIndexCompletionUsesTheSelectedNativeBackend) {
        if (!GetParam().backend)
            GTEST_SKIP() << "Native GPU completion contract";
        const auto input = make<float>(DataType::Float32, {3}, {2, 4, 6});
        const auto indices = Tensor::from_vector(std::vector<int32_t>{2, 0}, TensorShape{2}, device());
        const auto output = input.index_select(0, indices);
        lfs::core::internal::backend_driver(*GetParam().backend).complete_checked_index(output.stream(), LFS_SOURCE_SITE_CURRENT());
        expect_floats(host<float>(output), {6, 2});
    }

    TEST_P(TensorBackendSemantics, NormalReseedingPreservesOddAndEvenSequences) {
        auto& generator = lfs::core::RandomGenerator::instance();
        const auto old_seed = generator.get_seed();
        struct ResetSeed {
            uint64_t value;
            ~ResetSeed() { lfs::core::RandomGenerator::instance().manual_seed(value); }
        } reset{old_seed};
        for (const size_t count : {size_t{1}, size_t{2}, size_t{3}, size_t{16}}) {
            generator.manual_seed(731);
            const auto first = Tensor::normal({count}, 0.5f, 0.75f, device());
            const auto expected = host<float>(first);
            generator.manual_seed(731);
            const auto repeated = Tensor::normal({count}, 0.5f, 0.75f, device());
            EXPECT_EQ(expected, host<float>(repeated));
            for (const auto value : expected)
                EXPECT_TRUE(std::isfinite(value));
            expect_placement(first);
            expect_placement(repeated);
        }
    }

    TEST_P(TensorBackendSemantics, ExactAllocationPreservesPlacementAndEmptyStorage) {
        if (!GetParam().backend)
            GTEST_SKIP() << "GPU allocation API";
        auto tensor = Tensor::empty_exact({3, 2}, DataType::Float32);
        expect_placement(tensor);
        EXPECT_EQ(tensor.shape(), (TensorShape{3, 2}));
        tensor.fill_(7.f);
        expect_floats(host<float>(tensor), {7, 7, 7, 7, 7, 7});
        const auto empty = Tensor::empty_exact({0, 2}, DataType::Float32);
        EXPECT_TRUE(empty.is_valid());
        EXPECT_EQ(empty.numel(), 0);
        expect_placement(empty);
    }

    TEST_P(TensorBackendSemantics, FloatBoolConversionUsesNonzeroIncludingNan) {
        const auto source = make<float>(DataType::Float32, {6}, {0.f, -0.f, 0.25f, -0.25f, kNan, kInf});
        const auto output = source.to(DataType::Bool);
        expect_placement(output);
        EXPECT_EQ(host<uint8_t>(output), (std::vector<uint8_t>{0, 0, 1, 1, 1, 1}));
    }

    TEST_P(TensorBackendSemantics, FusedOutputsRetainTheirQueueOrdering) {
        if (!GetParam().backend)
            GTEST_SKIP() << "GPU expression API";
        const auto backend = *GetParam().backend;
        lfs::core::TensorWorkQueue producer(backend);
        lfs::core::TensorWorkQueue consumer(backend);
        Tensor input, first, second, observed;
        producer.execute([&] {
            input = Tensor::full({4}, 2.f, Device::GPU);
            first = Tensor::full({4}, -1.f, Device::GPU);
        });
        consumer.execute([&] { second = Tensor::full({4}, -2.f, Device::GPU); });
        producer.wait();
        consumer.wait();

        lfs::core::fused::Builder builder(1);
        const auto value = builder.input(DataType::Float32, 1);
        const auto gain = builder.input(DataType::Float32, 1);
        builder.output(value.load() + gain.at({0}), DataType::Float32);
        builder.output(value.load() * gain.at({0}), DataType::Float32);
        const lfs::core::fused::Kernel compiled(builder);
        const auto host_gain = Tensor::full({1}, 3.f, Device::CPU);
        compiled({4}, {input, host_gain}, {first, second});
        consumer.execute([&] { observed = second.add(1.f); });
        consumer.wait();
        expect_floats(host<float>(first), {5, 5, 5, 5});
        expect_floats(host<float>(observed), {7, 7, 7, 7});
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

    TEST_P(TensorBackendSemantics, MultinomialRejectsUnsafeCountsBeforeDispatch) {
        if (!GetParam().backend)
            GTEST_SKIP() << "GPU indexing limits do not apply to CPU sampling";
        const auto weights = Tensor::ones({1}, device());
        const auto output = Tensor::empty({1}, device(), DataType::Int64);
        const bool signed_counts = *GetParam().backend == GpuBackend::CUDA;
        const size_t maximum = std::numeric_limits<uint32_t>::max();
        // The last safe count solves count + ceil(count / 1024) <= UINT32_MAX.
        const size_t last_safe = (maximum / 1025) * 1024 + (maximum % 1025) - 1;
        for (const bool replacement : {false, true}) {
            for (const size_t count : {signed_counts ? size_t{std::numeric_limits<int>::max()} + 1 : last_safe + 1,
                                       size_t{4294000000}, maximum}) {
                SCOPED_TRACE(count);
                try {
                    lfs::core::internal::backend_ops_for(weights).multinomial(
                        lfs::core::internal::storage_ref(weights), lfs::core::internal::storage_ref(output),
                        {.count = count, .sample_count = 1, .replacement = replacement}, {});
                    FAIL() << "Unsafe count reached dispatch";
                } catch (const lfs::Exception& error) {
                    EXPECT_EQ(error.error().code(), lfs::ErrorCode::BoundsViolation);
                    EXPECT_EQ(error.error().domain(), lfs::ErrorDomain::Tensor);
                }
            }
        }
        if (signed_counts) {
            EXPECT_THROW(lfs::core::internal::backend_ops_for(weights).multinomial(
                             lfs::core::internal::storage_ref(weights), lfs::core::internal::storage_ref(output),
                             {.count = 1, .sample_count = size_t{std::numeric_limits<int>::max()} + 1, .replacement = true}, {}),
                         lfs::Exception);
        }
        EXPECT_EQ(Tensor::multinomial(weights, 1, true).to_vector_int64(), std::vector<int64_t>{0});
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

    TEST_P(TensorBackendSemantics, TypedAxisReductionOperationsAreDistinct) {
        const Tensor integers = make<int32_t>(DataType::Int32, {2, 3}, {3, 0, 7, 0, 0, 0});
        EXPECT_EQ(host<int32_t>(integers.min(1)), (std::vector<int32_t>{0, 0}));
        EXPECT_EQ(host<int32_t>(integers.max(1)), (std::vector<int32_t>{7, 0}));
        const Tensor booleans = integers.to(DataType::Bool);
        EXPECT_EQ(host<uint8_t>(booleans.any(1)), (std::vector<uint8_t>{1, 0}));
        EXPECT_EQ(host<uint8_t>(booleans.all(1)), (std::vector<uint8_t>{0, 0}));
        const Tensor half = make<float>(DataType::Float32, {2, 3}, {3, 0, 7, -2, -5, -1}).to(DataType::Float16);
        expect_floats(host<float>(half.min(1).to(DataType::Float32)), {0, -5});
        expect_floats(host<float>(half.max(1).to(DataType::Float32)), {7, -1});
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

    TEST_P(TensorBackendSemantics, BoolCastsUseNonzeroSemantics) {
        const Tensor floats = make<float>(DataType::Float32, {6}, {-1.f, -0.f, 0.f, 1.f, std::numeric_limits<float>::quiet_NaN(), std::numeric_limits<float>::infinity()});
        const std::vector<uint8_t> expected{1, 0, 0, 1, 1, 1};
        EXPECT_EQ(host<uint8_t>(floats.to(DataType::Bool)), expected);
        EXPECT_EQ(host<uint8_t>(floats.to(DataType::Float16).to(DataType::Bool)), expected);
        const Tensor integers = make<int32_t>(DataType::Int32, {3}, {-7, 0, 9});
        EXPECT_EQ(host<uint8_t>(integers.to(DataType::Bool)), (std::vector<uint8_t>{1, 0, 1}));
        EXPECT_EQ(host<uint8_t>(integers.to(DataType::Int64).to(DataType::Bool)), (std::vector<uint8_t>{1, 0, 1}));
        const Tensor bytes = make<uint8_t>(DataType::UInt8, {4}, {0, 1, 2, 255});
        EXPECT_EQ(host<uint8_t>(bytes.to(DataType::Bool)), (std::vector<uint8_t>{0, 1, 1, 1}));
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
                             testing::ValuesIn(lfs::test::tensor_backend_targets()),
                             lfs::test::tensor_backend_parameter_name);

} // namespace
