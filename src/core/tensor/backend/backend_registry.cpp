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
#include <format>

namespace lfs::core::internal {
    namespace {
        class UnavailableDriver final : public BackendDriver {
            GpuBackend backend_;
            const char* name_;

        public:
            void* canonical_home_target(void*) const noexcept override { return nullptr; }
            std::optional<std::array<uintptr_t, 2>> bind_tensor_target(const Tensor&, bool) override { return std::nullopt; }
            void synchronize_tensor_target(const Tensor&, void*) override {}
            void shutdown_compatibility_pool() override {}
            void set_pool_iteration(int) override {}
            std::optional<StorageAccountingKind> supplementary_accounting() const noexcept override { return std::nullopt; }

            explicit UnavailableDriver(GpuBackend backend, const char* name)
                : backend_(backend), name_(name) {}
            std::shared_ptr<TensorVulkanInteropBackend> make_vulkan_interop(VulkanInteropDevice) override { throw_unavailable(); }
            SplatPublication publication() const override { return SplatPublication::Copied; }
            std::unique_ptr<BackendWorkQueue> make_work_queue(const BackendQueueCreateInfo&) override { throw_unavailable(); }
            NativeTensorAllocation empty_tensor_allocation(TensorAllocationKind) override { return {.storage = {.backend = backend_}}; }
            void complete_checked_index(void*, SourceSite) override { throw_unavailable(); }
            std::array<uintptr_t, 2> bind_target(void*) override { throw_unavailable(); }
            void unbind_target(std::array<uintptr_t, 2>) noexcept override {}
            void bridge_targets(void*, void*) override { throw_unavailable(); }
            const char* name() const noexcept override { return name_; }
            bool available() override { return false; }
            bool live() noexcept override { return false; }
            [[noreturn]] void throw_unavailable() override {
                throw TensorError(std::format("GPU backend '{}' is unavailable", name()));
            }
            int device_count() override { return 0; }
            std::optional<GpuDeviceInfo> device_info(int) override { return std::nullopt; }
            std::vector<VulkanHardwareDevice> enumerate_hardware() override { return {}; }
            std::optional<std::string> validate_options(const TensorBackendOptions&, int& selection) override {
                selection = -1;
                return {};
            }
            void set_selection(int) noexcept override {}
            int selected_device() override {
                throw TensorError(std::format("GPU backend '{}' is not compiled into this build", name_));
            }
            bool needs_thread_binding() const noexcept override { return false; }
            void bind_thread() override {}
            void* current_target() const noexcept override { return nullptr; }
            BackendRendererAccess renderer_access() const noexcept override { return {}; }
            GpuBackendOps& activate() override {
                LFS_ASSERT_MSG(false, std::format("GPU backend '{}' is not compiled into this build", name_));
                std::unreachable();
            }
            MemoryInfo memory_info(bool) override { return {}; }
            MemoryInfo pressure_memory_info(bool) override { return {}; }
            bool pressure_recovery_blocked() const noexcept override { return false; }
            void register_pressure_clients(MemoryPressureCoordinator&) override {}
            size_t allocation_bytes(size_t) override {
                throw TensorError(std::format("GPU backend '{}' is not compiled into this build", name_));
            }
            lfs::Status shutdown() override { return {}; }
            TensorExportDevice dlpack_device(const Tensor&) override { reject_dlpack_export(name()); }
            Tensor prepare_dlpack_export(const Tensor&, std::optional<int64_t>) override { reject_dlpack_export(name()); }
            BackendExportCapabilities export_capabilities() const noexcept override { return {}; }
        };
        const BackendRegistry& drivers() {
            static const BackendRegistry registry = [] {
                static UnavailableDriver cuda{GpuBackend::CUDA, "CUDA"}, vulkan{GpuBackend::Vulkan, "Vulkan"}, metal{GpuBackend::Metal, "Metal"}, hip{GpuBackend::HIP, "HIP"};
                BackendRegistry result{&cuda, &vulkan, &metal, &hip};
                register_compiled_backends(result);
                return result;
            }();
            return registry;
        }
        thread_local std::array<GpuBackendOps*, kGpuBackendCount> activated{};
    } // namespace
    BackendDriver& backend_driver(GpuBackend backend) {
        const auto index = static_cast<size_t>(backend);
        LFS_ASSERT_MSG(index < kGpuBackendCount, "Unknown GPU backend");
        return *drivers()[index];
    }
    void clear_backend_activation_cache() noexcept { activated.fill(nullptr); }
    GpuBackendOps& backend_ops(GpuBackend backend) {
        const auto index = static_cast<size_t>(backend);
        LFS_DEBUG_ASSERT(index < kGpuBackendCount);
        auto* ops = activated[index];
        if (!ops) [[unlikely]] {
            ops = &backend_driver(backend).activate();
            activated[index] = ops;
        }
        return *ops;
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
    MemoryInfo backend_pressure_memory_info(BackendDriver& driver) {
        if (const auto device = driver.device_info(-1); device && device->supports_process_memory_budget) {
            MemoryInfo result;
            result.total_bytes = device->process_memory_budget_bytes;
            result.allocated_bytes = device->process_memory_used_bytes;
            result.free_bytes = result.total_bytes > result.allocated_bytes ? result.total_bytes - result.allocated_bytes : 0;
            return result;
        }
        return driver.memory_info(false);
    }
} // namespace lfs::core::internal
