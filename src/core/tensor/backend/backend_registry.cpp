/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "../internal/expression_runtime.hpp"
#include "../internal/tensor_impl.hpp"
#include "backend_driver.hpp"
#include "backend_work_queue.hpp"
#include "core/assert.hpp"
#include "gpu_backend_ops.hpp"
#include "native_allocation.hpp"
#include "native_export.hpp"
#include "tensor_vulkan_interop.hpp"
#include <atomic>
#include <format>

namespace lfs::core::internal {
    void BackendDriver::throw_unavailable() {
        throw TensorError(std::format("GPU backend '{}' is unavailable", name()));
    }
    MemoryInfo BackendDriver::memory_info(bool) { return activate().stats(); }
    TensorExportDevice BackendDriver::dlpack_device(const Tensor&) { reject_dlpack_export(name()); }
    Tensor BackendDriver::prepare_dlpack_export(const Tensor&, std::optional<int64_t>) { reject_dlpack_export(name()); }
    std::shared_ptr<TensorVulkanInteropBackend> BackendDriver::make_vulkan_interop(VulkanInteropDevice) {
        throw TensorError("Vulkan tensor interop is unavailable in this build");
    }
    SplatPublication BackendDriver::publication() const { return SplatPublication::Copied; }
    void BackendDriver::complete_checked_index(void* target, SourceSite) {
        activate().synchronize_stream(ExecContext{static_cast<cudaStream_t>(target)});
    }
    namespace {
        class UnavailableDriver final : public BackendDriver {
            GpuBackend backend_;
            const char* name_;

        public:
            explicit UnavailableDriver(GpuBackend backend, const char* name)
                : backend_(backend), name_(name) {}
            std::shared_ptr<TensorVulkanInteropBackend> make_vulkan_interop(VulkanInteropDevice) override { throw_unavailable(); }
            std::unique_ptr<BackendWorkQueue> make_work_queue(const BackendQueueCreateInfo&) override { throw_unavailable(); }
            NativeTensorAllocation empty_tensor_allocation(TensorAllocationKind) override { return {.storage = {.backend = backend_}}; }
            void complete_checked_index(void*, SourceSite) override { throw_unavailable(); }
            std::array<uintptr_t, 2> bind_target(void*) override { throw_unavailable(); }
            void unbind_target(std::array<uintptr_t, 2>) noexcept override {}
            void bridge_targets(void*, void*) override { throw_unavailable(); }
            const char* name() const noexcept override { return name_; }
            bool available() override { return false; }
            bool live() noexcept override { return false; }
            int device_count() override { return 0; }
            std::optional<GpuDeviceInfo> device_info(int) override { return std::nullopt; }
            int selected_device() override {
                throw TensorError(std::format("GPU backend '{}' is not compiled into this build", name_));
            }
            GpuBackendOps& activate() override {
                LFS_ASSERT_MSG(false, std::format("GPU backend '{}' is not compiled into this build", name_));
                std::unreachable();
            }
            MemoryInfo memory_info(bool) override { return {}; }
            MemoryInfo pressure_memory_info(bool) override { return {}; }
            size_t allocation_bytes(size_t) override {
                throw TensorError(std::format("GPU backend '{}' is not compiled into this build", name_));
            }
        };
        const BackendRegistry& drivers() {
            static const BackendRegistry registry = [] {
                // Driver control objects must survive callbacks registered before
                // the first registry lookup, including standalone Python teardown.
                static auto* const unavailable = new std::array{
                    UnavailableDriver{GpuBackend::CUDA, "CUDA"}, UnavailableDriver{GpuBackend::Vulkan, "Vulkan"},
                    UnavailableDriver{GpuBackend::Metal, "Metal"}};
                BackendRegistry result{&(*unavailable)[0], &(*unavailable)[1], &(*unavailable)[2]};
                register_compiled_backends(result);
                return result;
            }();
            return registry;
        }
        // Only the thread-scoped miss path touches TLS. In particular, ELF PIC
        // builds must not pay __tls_get_addr for every process-scoped lookup.
        std::array<std::atomic<GpuBackendOps*>, kGpuBackendCount> process_activated{};
        thread_local std::array<GpuBackendOps*, kGpuBackendCount> thread_activated{};

        // Keep TLS and activation's register spills out of process-cache hits.
        // This local call can be tail-called and does not cross an export boundary.
#if defined(_MSC_VER)
        __declspec(noinline)
#else
        __attribute__((noinline))
#endif
        GpuBackendOps&
        thread_or_uncached_ops(GpuBackend backend) {
            const auto index = static_cast<size_t>(backend);
            auto* ops = thread_activated[index];
            if (!ops) [[unlikely]] {
                auto& driver = backend_driver(backend);
                ops = &driver.activate();
                if (driver.activation_scope() == BackendActivationScope::Process)
                    process_activated[index].store(ops, std::memory_order_release);
                else
                    thread_activated[index] = ops;
            }
            return *ops;
        }
    } // namespace
    BackendDriver& backend_driver(GpuBackend backend) {
        const auto index = static_cast<size_t>(backend);
        LFS_ASSERT_MSG(index < kGpuBackendCount, "Unknown GPU backend");
        return *drivers()[index];
    }
    void clear_backend_activation_cache() noexcept {
        for (auto& ops : process_activated)
            ops.store(nullptr, std::memory_order_release);
        thread_activated.fill(nullptr);
    }
    GpuBackendOps& backend_ops(GpuBackend backend) {
        const auto index = static_cast<size_t>(backend);
        LFS_DEBUG_ASSERT(index < kGpuBackendCount);
        if (auto* ops = process_activated[index].load(std::memory_order_acquire)) [[likely]]
            return *ops;
        return thread_or_uncached_ops(backend);
    }
    GpuBackendOps& backend_ops_for(const Tensor& tensor) {
        LFS_ASSERT_MSG(tensor.is_valid(), "backend_ops_for requires a valid tensor");
        LFS_ASSERT_MSG(tensor.device() == Device::GPU, "backend_ops_for requires GPU storage");
        return backend_ops(gpu_backend_of(tensor).value());
    }
} // namespace lfs::core::internal

namespace lfs::core::internal {
    GpuBackendOps::~GpuBackendOps() = default;

} // namespace lfs::core::internal

namespace lfs::core::internal {
    MemoryInfo BackendDriver::pressure_memory_info(bool) {
        if (const auto device = device_info(-1); device && device->supports_process_memory_budget) {
            MemoryInfo result;
            result.total_bytes = device->process_memory_budget_bytes;
            result.allocated_bytes = device->process_memory_used_bytes;
            result.free_bytes = result.total_bytes > result.allocated_bytes ? result.total_bytes - result.allocated_bytes : 0;
            return result;
        }
        return memory_info(false);
    }
} // namespace lfs::core::internal
