/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/error.hpp"
#include "core/tensor.hpp"
#include "core/tensor/backend/gpu_backend_ops.hpp"
#include "core/tensor/backend/vulkan/vk_context.hpp"
#include "core/tensor/backend/vulkan/vk_shader_table.hpp"
#include "core/tensor_backend.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <latch>
#include <limits>
#include <mutex>
#include <ranges>
#include <set>
#include <span>
#include <string_view>
#include <thread>
#include <vector>

namespace {
    using namespace lfs::core;

    class TensorVulkanRuntime : public testing::Test {
    protected:
        void SetUp() override {
            ASSERT_TRUE(gpu_backend_available(GpuBackend::Vulkan));
        }

        void TearDown() override {
            const auto status = shutdown_gpu_backend(GpuBackend::Vulkan);
            EXPECT_TRUE(status.has_value());
            EXPECT_EQ(internal::vulkan_live_vma_objects_for_testing(), 0u);
            for (const std::string& message :
                 internal::vulkan_validation_messages_for_testing()) {
                ADD_FAILURE() << message;
            }
        }
    };

    TEST_F(TensorVulkanRuntime, EmptyCreatesNativeVulkanStorageDescriptor) {
        // Catches Tensor::empty dropping the native allocation descriptor returned by VMA.
        GpuBackendScope scope(GpuBackend::Vulkan);
        const Tensor tensor = Tensor::empty({17}, Device::GPU, DataType::Float32);
        const internal::StorageRef storage = internal::storage_ref(tensor);
        EXPECT_EQ(gpu_backend_of(tensor), GpuBackend::Vulkan);
        ASSERT_NE(storage.data, nullptr);
        ASSERT_NE(storage.meta, nullptr);
        EXPECT_NE(storage.meta->gpu_descriptor.native_buffer, 0u);
        EXPECT_NE(storage.meta->gpu_descriptor.native_allocation, 0u);
        EXPECT_NE(storage.meta->gpu_descriptor.base_address, 0u);
        EXPECT_EQ(storage.meta->gpu_descriptor.byte_size, 17u * sizeof(float));
        EXPECT_EQ(storage.meta->gpu_descriptor.accounting_kind,
                  StorageAccountingKind::VulkanOwned);
        EXPECT_EQ(tensor.data_ptr(), storage.data);
        EXPECT_EQ(tensor.storage_ptr(), storage.data);
    }

    TEST_F(TensorVulkanRuntime, DeviceIndexOverrideAndCapsAreExposed) {
#if defined(_WIN32)
        _putenv_s("LFS_VULKAN_DEVICE", "0");
#else
        setenv("LFS_VULKAN_DEVICE", "0", 1);
#endif
        const internal::VkDeviceCaps caps =
            internal::vulkan_device_caps_for_testing();
        EXPECT_EQ(caps.device_index, 0u);
        EXPECT_GT(caps.subgroup_size, 0u);
        EXPECT_GT(caps.max_workgroup_invocations, 0u);
        EXPECT_TRUE(std::ranges::any_of(caps.device_uuid,
                                        [](const uint8_t byte) {
                                            return byte != 0;
                                        }));
        const MemoryInfo memory = gpu_backend_memory_info(GpuBackend::Vulkan);
        EXPECT_GT(memory.total_bytes, 0u);
        EXPECT_EQ(memory.device_id, 0);
#if defined(_WIN32)
        _putenv_s("LFS_VULKAN_DEVICE", "");
#else
        unsetenv("LFS_VULKAN_DEVICE");
#endif
    }

    TEST_F(TensorVulkanRuntime, UploadAndDownloadAreBitExactForEveryDtypeAndBoundarySize) {
        // Catches staging-ring wrap, tail-byte, and descriptor-offset errors.
        constexpr std::array sizes{size_t{1}, size_t{7}, size_t{4099}, size_t{1048581}};
        constexpr std::array dtypes{
            DataType::Float32, DataType::Float16, DataType::Int32,
            DataType::Int64, DataType::UInt8, DataType::Bool, DataType::UInt32};
        for (const DataType dtype : dtypes) {
            for (const size_t size : sizes) {
                Tensor cpu = Tensor::empty({size}, Device::CPU, dtype, false);
                auto* const bytes = static_cast<uint8_t*>(cpu.data_ptr());
                for (size_t index = 0; index < cpu.bytes(); ++index) {
                    bytes[index] = static_cast<uint8_t>((index * 37 + 11) & 0xff);
                }
                Tensor vulkan;
                {
                    GpuBackendScope scope(GpuBackend::Vulkan);
                    vulkan = cpu.to(Device::GPU);
                }
                EXPECT_EQ(gpu_backend_of(vulkan), GpuBackend::Vulkan);
                const Tensor downloaded = vulkan.to(Device::CPU);
                ASSERT_EQ(downloaded.bytes(), cpu.bytes());
                EXPECT_EQ(std::memcmp(downloaded.data_ptr(), cpu.data_ptr(), cpu.bytes()), 0)
                    << "dtype=" << static_cast<int>(dtype) << " size=" << size;
            }
        }
    }

    TEST_F(TensorVulkanRuntime, StagingRingWrapWaitsBeforeReusingSlices) {
        // Two 1 MiB staging slices per iteration force the 64 MiB ring to wrap.
        const std::vector<int> expected(256 * 1024, 0x13579bdf);
        for (int iteration = 0; iteration < 34; ++iteration) {
            GpuBackendScope scope(GpuBackend::Vulkan);
            const Tensor vulkan = Tensor::from_vector(
                expected, {expected.size()}, Device::GPU);
            EXPECT_EQ(vulkan.to_vector_int(), expected);
        }
    }

    TEST_F(TensorVulkanRuntime, ZeroOneAndFullUseTransferAndFillPaths) {
        // Catches vkCmdFillBuffer tails and the Slang BDA fill push-constant ABI.
        GpuBackendScope scope(GpuBackend::Vulkan);
        EXPECT_EQ(Tensor::zeros({7}, Device::GPU).to_vector(),
                  std::vector<float>(7, 0.0f));
        EXPECT_EQ(Tensor::ones({4099}, Device::GPU).to_vector(),
                  std::vector<float>(4099, 1.0f));
        EXPECT_EQ(Tensor::full({17}, -3.25f, Device::GPU).to_vector(),
                  std::vector<float>(17, -3.25f));
        EXPECT_EQ(Tensor::full({7}, 9.0f, Device::GPU, DataType::Int32)
                      .to_vector_int(),
                  std::vector<int>(7, 9));
        EXPECT_EQ(Tensor::full({7}, 1.0f, Device::GPU, DataType::Bool)
                      .to_vector_bool(),
                  std::vector<bool>(7, true));
        EXPECT_FLOAT_EQ(Tensor::full({1}, 2.25f, Device::GPU).item<float>(),
                        2.25f);
        EXPECT_EQ(Tensor::full({7}, 5.0f, Device::GPU).cpu().to_vector(),
                  std::vector<float>(7, 5.0f));
    }

    TEST_F(TensorVulkanRuntime, ViewsShareDescriptorAndDownloadInLogicalOrder) {
        // Catches view offsets or strides being interpreted as host pointers.
        GpuBackendScope scope(GpuBackend::Vulkan);
        const Tensor base = Tensor::from_vector(
            std::vector<float>{1, 2, 3, 4, 5, 6}, {2, 3}, Device::GPU);
        const Tensor sliced = base.slice(0, 1, 2);
        const Tensor transposed = base.transpose(0, 1);
        const Tensor row = Tensor::from_vector(
            std::vector<float>{7, 8, 9}, {1, 3}, Device::GPU);
        const Tensor expanded = row.expand({4, 3});
        const Tensor already_contiguous = base.contiguous();
        EXPECT_EQ(internal::storage_ref(base).meta, internal::storage_ref(sliced).meta);
        EXPECT_EQ(internal::storage_ref(base).meta,
                  internal::storage_ref(transposed).meta);
        EXPECT_EQ(internal::storage_ref(base).meta,
                  internal::storage_ref(already_contiguous).meta);
        EXPECT_EQ(sliced.storage_ptr(), base.storage_ptr());
        EXPECT_EQ(reinterpret_cast<uintptr_t>(sliced.data_ptr()),
                  reinterpret_cast<uintptr_t>(base.data_ptr()) + 3 * sizeof(float));
        EXPECT_EQ(sliced.to_vector(), std::vector<float>({4, 5, 6}));
        EXPECT_EQ(transposed.to_vector(),
                  std::vector<float>({1, 4, 2, 5, 3, 6}));
        EXPECT_EQ(expanded.to_vector(),
                  std::vector<float>({7, 8, 9, 7, 8, 9, 7, 8, 9, 7, 8, 9}));
    }

    TEST_F(TensorVulkanRuntime, CloneAndReservePreserveBytesAndGenerationRules) {
        // Catches D2D copies using a BDA as VkBuffer and reserve losing generation state.
        GpuBackendScope scope(GpuBackend::Vulkan);
        Tensor tensor = Tensor::from_vector(
            std::vector<float>{1, 2, 3, 4}, {2, 2}, Device::GPU);
        const Tensor clone = tensor.clone();
        EXPECT_NE(internal::storage_ref(tensor).meta->gpu_descriptor.native_allocation,
                  internal::storage_ref(clone).meta->gpu_descriptor.native_allocation);
        EXPECT_EQ(clone.to_vector(), tensor.to_vector());

        const Tensor stale = tensor.slice(0, 0, 1);
        const auto stale_meta = internal::storage_ref(tensor).meta;
        const uint64_t generation = stale_meta->generation.load();
        tensor.reserve(8);
        EXPECT_EQ(tensor.to_vector(), std::vector<float>({1, 2, 3, 4}));
        EXPECT_GT(stale_meta->generation.load(), generation);
        EXPECT_NE(internal::storage_ref(tensor).meta, stale_meta);
        EXPECT_THROW(static_cast<void>(stale.data_ptr()), std::runtime_error);
    }

    TEST_F(TensorVulkanRuntime, RetiredTierAllocationIsReusedAfterTimelineCompletion) {
        GpuBackendScope scope(GpuBackend::Vulkan);
        uint64_t retired_allocation = 0;
        {
            const Tensor first = Tensor::ones({4099}, Device::GPU);
            EXPECT_EQ(first.to_vector().front(), 1.0f);
            retired_allocation =
                internal::storage_ref(first).meta->gpu_descriptor.native_allocation;
        }
        const Tensor second = Tensor::empty({4099}, Device::GPU);
        EXPECT_EQ(internal::storage_ref(second).meta->gpu_descriptor.native_allocation,
                  retired_allocation);
    }

    TEST_F(TensorVulkanRuntime, RecorderProtectsReadAllocationsUntilTimelineCompletion) {
        // Catches a recorder that stamps allocation lifetime only for writes.
        GpuBackendScope scope(GpuBackend::Vulkan);
        Tensor source = Tensor::empty({4099}, Device::GPU);
        const uint64_t source_allocation =
            internal::storage_ref(source).meta->gpu_descriptor.native_allocation;
        const uint64_t completed_before =
            internal::vulkan_completed_timeline_for_testing();
        const Tensor copy = source.clone();
        source = Tensor{};

        const Tensor before_completion = Tensor::empty({4099}, Device::GPU);
        EXPECT_NE(internal::storage_ref(before_completion)
                      .meta->gpu_descriptor.native_allocation,
                  source_allocation);
        EXPECT_EQ(internal::vulkan_completed_timeline_for_testing(),
                  completed_before);

        static_cast<void>(copy.cpu());
        EXPECT_GT(internal::vulkan_completed_timeline_for_testing(),
                  completed_before);
        const Tensor after_completion = Tensor::empty({4099}, Device::GPU);
        EXPECT_EQ(internal::storage_ref(after_completion)
                      .meta->gpu_descriptor.native_allocation,
                  source_allocation);
    }

    TEST_F(TensorVulkanRuntime, ExplicitCrossBackendCopyIsBitExactBothWays) {
        // Catches copy_to_backend consulting the active scope instead of the requested backend.
        Tensor scoped_same_device_clone;
        {
            GpuBackendScope scope(GpuBackend::Vulkan);
            scoped_same_device_clone =
                Tensor::from_vector(std::vector<float>{4, 3, 2, 1}, {4})
                    .to(Device::GPU);
        }
        EXPECT_EQ(gpu_backend_of(scoped_same_device_clone), GpuBackend::Vulkan);
        EXPECT_EQ(scoped_same_device_clone.to_vector(),
                  std::vector<float>({4, 3, 2, 1}));

        Tensor cuda;
        {
            GpuBackendScope scope(GpuBackend::CUDA);
            cuda = Tensor::from_vector(std::vector<float>{1, -2, 3, 9}, {4},
                                       Device::GPU);
        }
        const Tensor vulkan = internal::copy_to_backend(cuda, GpuBackend::Vulkan);
        const Tensor cuda_roundtrip =
            internal::copy_to_backend(vulkan, GpuBackend::CUDA);
        EXPECT_EQ(gpu_backend_of(vulkan), GpuBackend::Vulkan);
        EXPECT_EQ(gpu_backend_of(cuda_roundtrip), GpuBackend::CUDA);
        EXPECT_EQ(cuda_roundtrip.to_vector(), cuda.to_vector());
    }

    TEST_F(TensorVulkanRuntime, MixedBackendOperandsFailAtTheFacade) {
        // Catches mixed storage reaching either API.
        Tensor cuda;
        Tensor vulkan;
        {
            GpuBackendScope scope(GpuBackend::CUDA);
            cuda = Tensor::ones({4}, Device::GPU);
        }
        {
            GpuBackendScope scope(GpuBackend::Vulkan);
            vulkan = Tensor::ones({4}, Device::GPU);
        }
        try {
            static_cast<void>(cuda + vulkan);
            FAIL() << "mixed-backend binary operation did not throw";
        } catch (const std::exception& error) {
            EXPECT_NE(std::string(error.what()).find("matching GPU backends"),
                      std::string::npos);
        }
    }

    TEST_F(TensorVulkanRuntime, CrossThreadConsumerFlushesUnsubmittedProducer) {
        // Catches pending tokens that only synchronize the consuming thread's recorder.
        std::mutex mutex;
        std::condition_variable condition;
        Tensor shared;
        bool published = false;
        bool consumed = false;
        std::thread producer([&] {
            GpuBackendScope scope(GpuBackend::Vulkan);
            Tensor value = Tensor::full({4099}, 6.5f, Device::GPU);
            {
                std::lock_guard lock(mutex);
                shared = std::move(value);
                published = true;
            }
            condition.notify_all();
            std::unique_lock lock(mutex);
            condition.wait(lock, [&] { return consumed; });
        });
        std::thread consumer([&] {
            std::unique_lock lock(mutex);
            condition.wait(lock, [&] { return published; });
            const Tensor value = shared;
            lock.unlock();
            EXPECT_EQ(value.to_vector(), std::vector<float>(4099, 6.5f));
            {
                std::lock_guard done_lock(mutex);
                consumed = true;
            }
            condition.notify_all();
        });
        producer.join();
        consumer.join();
    }

    TEST_F(TensorVulkanRuntime, ThreadExitFlushesItsOpenRecorder) {
        // Catches a thread-exit token that drops an open command buffer without submitting it.
        Tensor shared;
        std::thread producer([&] {
            GpuBackendScope scope(GpuBackend::Vulkan);
            shared = Tensor::full({4099}, 3.5f, Device::GPU);
        });
        producer.join();
        EXPECT_EQ(shared.to_vector(), std::vector<float>(4099, 3.5f));
    }

    TEST_F(TensorVulkanRuntime, CompletedDeadThreadRecordersAreReclaimed) {
        // Catches exited threads retaining command pools in the recorder registry.
        std::vector<std::thread> threads;
        threads.reserve(64);
        for (int index = 0; index < 64; ++index) {
            threads.emplace_back([] {
                GpuBackendScope scope(GpuBackend::Vulkan);
                const Tensor source = Tensor::empty({64}, Device::GPU);
                const Tensor copy = source.clone();
                static_cast<void>(copy);
            });
        }
        for (std::thread& thread : threads) {
            thread.join();
        }

        internal::backend_ops(GpuBackend::Vulkan).synchronize_device();
        EXPECT_EQ(internal::vulkan_dead_recorder_count_for_testing(), 0u);
    }

    TEST_F(TensorVulkanRuntime, ShutdownWaitsForAutoSubmittedCommands) {
        // 65 fills cross the 64-command auto-flush; shutdown must wait for that
        // submission. The tensors are deliberately kept alive across shutdown, so
        // the leak detector must report exactly them (an accessor that returned
        // zero here was vacuous), and releasing them afterwards must be a no-op.
        std::vector<Tensor> live;
        live.reserve(65);
        {
            GpuBackendScope scope(GpuBackend::Vulkan);
            for (int index = 0; index < 65; ++index) {
                live.push_back(Tensor::full({4099}, static_cast<float>(index),
                                            Device::GPU));
            }
        }
        ASSERT_TRUE(shutdown_gpu_backend(GpuBackend::Vulkan).has_value());
        EXPECT_EQ(internal::vulkan_live_vma_objects_for_testing(), 65u);
        live.clear();
    }

    TEST_F(TensorVulkanRuntime, ShutdownDropsLiveContextAndReinitializes) {
        // Catches call_once state or late deleters retaining a destroyed VkDevice.
        {
            GpuBackendScope scope(GpuBackend::Vulkan);
            const Tensor value = Tensor::ones({17}, Device::GPU);
            EXPECT_EQ(value.to_vector(), std::vector<float>(17, 1.0f));
        }
        ASSERT_TRUE(shutdown_gpu_backend(GpuBackend::Vulkan).has_value());
        EXPECT_EQ(internal::vulkan_live_vma_objects_for_testing(), 0u);
        {
            GpuBackendScope scope(GpuBackend::Vulkan);
            const Tensor value = Tensor::full({7}, 2.0f, Device::GPU);
            EXPECT_EQ(value.to_vector(), std::vector<float>(7, 2.0f));
        }
    }

    TEST_F(TensorVulkanRuntime, SurvivorOfShutdownAndReinitIsReleasedAsNoOp) {
        // Catches deleters that dereference the destroyed allocator record after
        // shutdown, or free a live allocation of the new context whose VMA handle
        // value was recycled from the old one.
        Tensor survivor;
        {
            GpuBackendScope scope(GpuBackend::Vulkan);
            survivor = Tensor::full({33}, 3.0f, Device::GPU);
            EXPECT_EQ(survivor.to_vector(), std::vector<float>(33, 3.0f));
        }
        ASSERT_TRUE(shutdown_gpu_backend(GpuBackend::Vulkan).has_value());
        {
            GpuBackendScope scope(GpuBackend::Vulkan);
            const Tensor fresh = Tensor::full({33}, 5.0f, Device::GPU);
            survivor = Tensor();
            EXPECT_EQ(fresh.to_vector(), std::vector<float>(33, 5.0f));
            EXPECT_EQ(Tensor::full({33}, 6.0f, Device::GPU).to_vector(),
                      std::vector<float>(33, 6.0f));
        }
    }

    TEST_F(TensorVulkanRuntime, OddOffsetViewsAreZeroedWithoutUnalignedFillCommands) {
        // Catches vkCmdFillBuffer issued at a view offset that is not a multiple of
        // four (VUID-vkCmdFillBuffer-dstOffset-00028); the validation collector in
        // TearDown turns the violation into a failure.
        GpuBackendScope scope(GpuBackend::Vulkan);
        for (const size_t start : {size_t{1}, size_t{2}, size_t{3}}) {
            std::vector<float> values(11);
            for (size_t index = 0; index < values.size(); ++index) {
                values[index] = static_cast<float>(index + 1);
            }
            Tensor half = Tensor::from_vector(values, {values.size()}, Device::GPU)
                              .to(DataType::Float16);
            half.slice(0, start, start + 5).zero_();
            std::vector<float> expected = values;
            for (size_t index = start; index < start + 5; ++index) {
                expected[index] = 0.0f;
            }
            EXPECT_EQ(half.to(DataType::Float32).to_vector(), expected) << "start=" << start;

            Tensor bytes = Tensor::from_vector(values, {values.size()}, Device::GPU)
                               .to(DataType::UInt8);
            bytes.slice(0, start, start + 5).zero_();
            std::vector<int> expected_bytes(values.size());
            for (size_t index = 0; index < values.size(); ++index) {
                expected_bytes[index] = index >= start && index < start + 5
                                            ? 0
                                            : static_cast<int>(values[index]);
            }
            EXPECT_EQ(bytes.to(DataType::Int32).to_vector_int(), expected_bytes)
                << "start=" << start;
        }
    }

    TEST_F(TensorVulkanRuntime, DispatchesAboveTheDeviceGroupLimitCoverEveryElement) {
        // Catches a dispatch that asserts or truncates when ceil(count / 256) exceeds
        // maxComputeWorkGroupCount[0] (65535 on lavapipe): fill, pointwise and convert
        // must iterate grid-stride instead. Skips on devices whose limit is not reachable.
        GpuBackendScope scope(GpuBackend::Vulkan);
        const uint64_t limit = internal::vulkan_device_caps_for_testing().max_workgroup_count[0];
        constexpr size_t count = 20'000'003;
        if ((count + 255) / 256 <= limit) {
            GTEST_SKIP() << "device group limit " << limit << " is above the test size";
        }
        const Tensor ones = Tensor::ones({count}, Device::GPU);
        const Tensor twos = ones + 1.0f;
        const Tensor ints = twos.to(DataType::Int32);
        const auto values = ints.to_vector_int();
        ASSERT_EQ(values.size(), count);
        EXPECT_EQ(values.front(), 2);
        EXPECT_EQ(values[count / 2], 2);
        EXPECT_EQ(values.back(), 2);
        EXPECT_EQ(std::count(values.begin(), values.end(), 2), static_cast<ptrdiff_t>(count));
    }

    TEST_F(TensorVulkanRuntime, CrossThreadWriteAfterWriteFollowsProgramOrder) {
        // Catches writes that never consult the target's pending token: a writer whose
        // command buffer was opened earlier (lower timeline value) would be submitted
        // before the producer that wrote first, inverting program order.
        std::mutex mutex;
        std::condition_variable condition;
        Tensor shared;
        bool late_opened = false;
        bool first_written = false;
        bool second_written = false;
        std::thread late_writer([&] {
            GpuBackendScope scope(GpuBackend::Vulkan);
            const Tensor warm = Tensor::full({4099}, 0.5f, Device::GPU);
            {
                std::lock_guard lock(mutex);
                late_opened = true;
            }
            condition.notify_all();
            Tensor target;
            {
                std::unique_lock lock(mutex);
                condition.wait(lock, [&] { return first_written; });
                target = shared;
            }
            target.fill_(2.0f);
            {
                std::lock_guard lock(mutex);
                second_written = true;
            }
            condition.notify_all();
        });
        std::thread first_writer([&] {
            GpuBackendScope scope(GpuBackend::Vulkan);
            {
                std::unique_lock lock(mutex);
                condition.wait(lock, [&] { return late_opened; });
            }
            Tensor value = Tensor::full({4099}, 1.0f, Device::GPU);
            {
                std::lock_guard lock(mutex);
                shared = std::move(value);
                first_written = true;
            }
            condition.notify_all();
            std::unique_lock lock(mutex);
            condition.wait(lock, [&] { return second_written; });
        });
        first_writer.join();
        late_writer.join();
        EXPECT_EQ(shared.to_vector(), std::vector<float>(4099, 2.0f));
    }

    TEST_F(TensorVulkanRuntime, EnvironmentSelectsVulkanForPublicFactories) {
        internal::gpu_backend_reset_for_testing();
#if defined(_WIN32)
        _putenv_s("LFS_TENSOR_BACKEND", "vulkan");
#else
        setenv("LFS_TENSOR_BACKEND", "vulkan", 1);
#endif
        EXPECT_EQ(default_gpu_backend(), GpuBackend::Vulkan);
        const Tensor value = Tensor::full({7}, 4.0f, Device::GPU);
        EXPECT_EQ(gpu_backend_of(value), GpuBackend::Vulkan);
        EXPECT_EQ(value.to_vector(), std::vector<float>(7, 4.0f));
#if defined(_WIN32)
        _putenv_s("LFS_TENSOR_BACKEND", "");
#else
        unsetenv("LFS_TENSOR_BACKEND");
#endif
        internal::gpu_backend_reset_for_testing();
    }

    TEST_F(TensorVulkanRuntime, EmbeddedShadersCarryTheFloatControlsContract) {
        // Catches a module embedded without the finalize step (no execution
        // mode), an fp32 module that regained the Float16 capability through a
        // half intrinsic, and a capability the loader table does not know.
        const std::set<std::string_view> known{"Shader", "Int64", "Int16",
                                               "PhysicalStorageBufferAddresses",
                                               "SignedZeroInfNanPreserve", "Float16",
                                               "AtomicFloat32AddEXT"};
        const std::span<const internal::EmbeddedShader> shaders = internal::embedded_shaders();
        EXPECT_EQ(shaders.size(), 21u);
        for (const internal::EmbeddedShader& shader : shaders) {
            const std::string module(shader.name);
            EXPECT_EQ(internal::find_embedded_shader(shader.name), &shader) << module;
            EXPECT_GT(shader.code.size(), 5u) << module;
            EXPECT_EQ(shader.code[0], 0x07230203u) << module;
            EXPECT_EQ(shader.entry_point, "main") << module;
            EXPECT_EQ(shader.local_size, (std::array<uint32_t, 3>{256, 1, 1})) << module;
            EXPECT_EQ(shader.push_constant_size % 4, 0u) << module;
            for (const std::string_view capability : shader.capabilities) {
                EXPECT_TRUE(known.contains(capability)) << module << " declares " << capability;
            }
            const bool half = std::ranges::find(shader.capabilities, "Float16") != shader.capabilities.end();
            EXPECT_EQ(half, module == "pointwise_half") << module;
            EXPECT_EQ(std::ranges::find(shader.capabilities, "AtomicFloat32AddEXT") !=
                          shader.capabilities.end(),
                      module == "index_atomic")
                << module;
            EXPECT_TRUE(std::ranges::equal(shader.float_widths, shader.signed_zero_inf_nan_preserve))
                << module;
            const bool byte_mover = module == "fill" || module == "cat_pad";
            EXPECT_EQ(std::ranges::find(shader.float_widths, 32u) != shader.float_widths.end(),
                      !byte_mover)
                << module;
            EXPECT_EQ(std::ranges::find(shader.float_widths, 16u) != shader.float_widths.end(), half)
                << module;
        }
        EXPECT_EQ(internal::find_embedded_shader("no_such_module"), nullptr);
        const internal::VkDeviceCaps caps = internal::vulkan_device_caps_for_testing();
        EXPECT_TRUE(caps.float_controls_fp16 || !caps.shader_float16);
    }

    TEST_F(TensorVulkanRuntime, HalfConversionsMatchCudaBitForBitOnEverySpecialValue) {
        // The fp32 modules convert Float16 storage with integer arithmetic; a
        // wrong tie, a dropped subnormal, an early overflow or a NaN turned
        // infinity differs from the CUDA conversion in at least one bit.
        std::vector<float> values{
            0.0f, -0.0f, 1.0f, -1.0f, 65504.0f, 65519.99f, 65520.0f, 65536.0f, 1.0e5f,
            std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity(),
            std::numeric_limits<float>::quiet_NaN(), 6.103515625e-05f, 5.9604645e-08f,
            std::ldexp(1.0f, -25), std::ldexp(1.5f, -25), std::ldexp(1.0f, -26),
            std::ldexp(3.0f, -26), 1.0f + std::ldexp(1.0f, -11), 1.0f + std::ldexp(3.0f, -11),
            1.0f + std::ldexp(1.0f, -11) + std::ldexp(1.0f, -20), 0.1f, 0.3f, 1.0e-3f, 3.14159f,
            10000.7f, -2.5e-5f, 1.2345e-6f};
        for (int i = 0; i < 4096; ++i) {
            values.push_back(std::ldexp(1.0f + static_cast<float>(i) / 4096.0f, (i % 40) - 20));
            values.push_back(-std::ldexp(1.0f + static_cast<float>(i * 7 % 4096) / 4096.0f,
                                         (i % 13) - 26));
        }
        const Tensor cpu = Tensor::from_vector(values, {values.size()}, Device::CPU);
        const auto round_trip = [&](const GpuBackend backend) {
            GpuBackendScope scope(backend);
            return cpu.to(Device::GPU).to(DataType::Float16).to(DataType::Float32).cpu().to_vector();
        };
        const std::vector<float> cuda = round_trip(GpuBackend::CUDA);
        const std::vector<float> vulkan = round_trip(GpuBackend::Vulkan);
        ASSERT_EQ(cuda.size(), vulkan.size());
        size_t mismatches = 0;
        for (size_t i = 0; i < cuda.size(); ++i) {
            const bool both_nan = std::isnan(cuda[i]) && std::isnan(vulkan[i]);
            if (!both_nan && std::bit_cast<uint32_t>(cuda[i]) != std::bit_cast<uint32_t>(vulkan[i])) {
                ++mismatches;
                if (mismatches <= 8) {
                    ADD_FAILURE() << "index " << i << " input " << values[i] << " cuda " << cuda[i]
                                  << " vulkan " << vulkan[i];
                }
            }
        }
        EXPECT_EQ(mismatches, 0u);
    }

    TEST_F(TensorVulkanRuntime, InjectedDeviceLossRaisesTypedErrorsAndShutsDownCleanly) {
        // Catches a lost-device path that throws a boundary assertion instead of
        // the typed DeviceLost error consumers act on, a probe that keeps
        // reporting a dead backend as available, and a shutdown that fails or
        // hangs on a dead context.
        {
            GpuBackendScope scope(GpuBackend::Vulkan);
            const Tensor warm = Tensor::ones({16}, Device::GPU);
            ASSERT_FLOAT_EQ(warm.sum_scalar(), 16.0f);
        }
        internal::vulkan_inject_device_loss_for_testing();
        EXPECT_FALSE(gpu_backend_available(GpuBackend::Vulkan));
        bool typed = false;
        try {
            GpuBackendScope scope(GpuBackend::Vulkan);
            const Tensor lost = Tensor::zeros({16}, Device::GPU);
            (void)lost.sum_scalar();
        } catch (const lfs::Exception& error) {
            typed = error.error().code() == lfs::ErrorCode::DeviceLost;
            EXPECT_TRUE(typed) << error.what();
        } catch (const std::exception& error) {
            ADD_FAILURE() << "untyped failure on a lost device: " << error.what();
        }
        EXPECT_TRUE(typed);
        const auto status = shutdown_gpu_backend(GpuBackend::Vulkan);
        EXPECT_TRUE(status.has_value());
        ASSERT_TRUE(gpu_backend_available(GpuBackend::Vulkan));
        GpuBackendScope scope(GpuBackend::Vulkan);
        const Tensor fresh = Tensor::ones({16}, Device::GPU);
        EXPECT_FLOAT_EQ(fresh.sum_scalar(), 16.0f);
    }

    TEST_F(TensorVulkanRuntime, ThreadsWithPendingWorkExitCleanlyAfterDeviceLoss) {
        // Catches the thread-exit token submitting a pending batch to a lost
        // device from its destructor, which ends the process.
        std::latch recorded(1);
        std::latch lost(1);
        std::thread worker([&] {
            GpuBackendScope scope(GpuBackend::Vulkan);
            const Tensor pending = Tensor::zeros({64}, Device::GPU);
            (void)pending;
            recorded.count_down();
            lost.wait();
        });
        recorded.wait();
        internal::vulkan_inject_device_loss_for_testing();
        lost.count_down();
        worker.join();
        EXPECT_TRUE(shutdown_gpu_backend(GpuBackend::Vulkan).has_value());
        ASSERT_TRUE(gpu_backend_available(GpuBackend::Vulkan));
    }

    TEST_F(TensorVulkanRuntime, TrimReleasesFreeVulkanBlocksOfBothKinds) {
        // Catches a public trim that only reaches the CUDA pool: pooled blocks
        // and readback blocks freed on Vulkan stay cached until the backend
        // shuts down.
        {
            GpuBackendScope scope(GpuBackend::Vulkan);
            const Tensor warm = Tensor::ones({256}, Device::GPU);
            ASSERT_FLOAT_EQ(warm.sum_scalar(), 256.0f);
        }
        Tensor::trim_memory_pool();
        const uint64_t baseline = internal::vulkan_live_vma_objects_for_testing();
        {
            GpuBackendScope scope(GpuBackend::Vulkan);
            std::vector<Tensor> blocks;
            for (size_t i = 0; i < 6; ++i) {
                blocks.push_back(Tensor::ones({4096 + 512 * i}, Device::GPU));
            }
            float total = 0.0f;
            for (const Tensor& block : blocks) {
                total += block.sum_scalar();
            }
            EXPECT_GT(total, 0.0f);
        }
        EXPECT_GT(internal::vulkan_live_vma_objects_for_testing(), baseline);
        Tensor::trim_memory_pool();
        EXPECT_EQ(internal::vulkan_live_vma_objects_for_testing(), baseline);
    }

} // namespace
