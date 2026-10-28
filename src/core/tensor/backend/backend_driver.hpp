/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once

#include "core/gpu_device_info.hpp"
#include "core/source_site.hpp"
#include "core/tensor_backend.hpp"
#include "core/tensor_backend_vulkan.hpp"
#include "core/tensor_protocol_export.hpp"
#include <array>
#include <memory>
#include <optional>
#include <string>

namespace lfs::core {
    struct VulkanInteropDevice;
    class MemoryPressureCoordinator;
    enum class SplatPublication : uint8_t;
    enum class StorageAccountingKind : uint8_t;
} // namespace lfs::core

namespace lfs::core::internal {
    class TensorVulkanInteropBackend;
    class GpuBackendOps;
    class BackendWorkQueue;
    struct NativeTensorAllocation;
    enum class TensorAllocationKind;
    struct BackendQueueCreateInfo;
    using BackendExportCapabilities = TensorExportCapabilities;
    // Inventory/selection never activates Tensor storage. Activation is cold:
    // backend_ops caches the resulting pointer in the calling thread.
    struct BackendRendererAccess {
        bool shares_vulkan = false, needs_vulkan_interop = false, metal = false;
    };
    class BackendDriver {
    public:
        virtual ~BackendDriver() = default;
        virtual const char* name() const noexcept = 0;
        virtual bool available() = 0;
        virtual bool live() noexcept = 0;
        [[noreturn]] virtual void throw_unavailable() = 0;
        virtual int device_count() = 0;
        virtual std::optional<GpuDeviceInfo> device_info(int ordinal) = 0;
        virtual std::vector<VulkanHardwareDevice> enumerate_hardware() = 0;
        virtual std::optional<std::string> validate_options(const TensorBackendOptions&, int& selection) = 0;
        virtual void set_selection(int selection) noexcept = 0;
        virtual int selected_device() = 0;
        virtual bool needs_thread_binding() const noexcept = 0;
        virtual void bind_thread() = 0;
        virtual void* current_target() const noexcept = 0;
        virtual BackendRendererAccess renderer_access() const noexcept = 0;
        virtual std::array<uintptr_t, 2> bind_target(void*) = 0;
        virtual void unbind_target(std::array<uintptr_t, 2>) noexcept = 0;
        virtual void bridge_targets(void* consumer, void* producer) = 0;
        virtual GpuBackendOps& activate() = 0;
        virtual MemoryInfo memory_info(bool include_pool_stats) = 0;
        virtual MemoryInfo pressure_memory_info(bool free_query) = 0;
        virtual bool pressure_recovery_blocked() const noexcept = 0;
        virtual void register_pressure_clients(MemoryPressureCoordinator&) = 0;
        virtual size_t allocation_bytes(size_t bytes) = 0;
        virtual lfs::Status shutdown() = 0;
        virtual BackendExportCapabilities export_capabilities() const noexcept = 0;
        virtual TensorExportDevice dlpack_device(const Tensor&) = 0;
        virtual Tensor prepare_dlpack_export(const Tensor&, std::optional<int64_t>) = 0;
        virtual std::shared_ptr<TensorVulkanInteropBackend> make_vulkan_interop(VulkanInteropDevice) = 0;
        virtual SplatPublication publication() const = 0;
        virtual std::unique_ptr<BackendWorkQueue> make_work_queue(const BackendQueueCreateInfo&) = 0;
        virtual void* canonical_home_target(void*) const noexcept = 0;
        virtual std::optional<std::array<uintptr_t, 2>> bind_tensor_target(const Tensor&, bool implicit) = 0;
        virtual void synchronize_tensor_target(const Tensor&, void*) = 0;
        virtual void shutdown_compatibility_pool() = 0;
        virtual void set_pool_iteration(int) = 0;
        virtual std::optional<StorageAccountingKind> supplementary_accounting() const noexcept = 0;
        virtual NativeTensorAllocation empty_tensor_allocation(TensorAllocationKind) = 0;
        virtual void complete_checked_index(void* target, SourceSite location) = 0;
    };
    class TensorTargetBinding {
        BackendDriver& driver_;
        std::optional<std::array<uintptr_t, 2>> token_;

    public:
        TensorTargetBinding(BackendDriver& driver, const Tensor& tensor, bool implicit)
            : driver_(driver), token_(driver.bind_tensor_target(tensor, implicit)) {}
        ~TensorTargetBinding() {
            if (token_)
                driver_.unbind_target(*token_);
        }
        TensorTargetBinding(const TensorTargetBinding&) = delete;
        TensorTargetBinding& operator=(const TensorTargetBinding&) = delete;
    };
    MemoryInfo backend_pressure_memory_info(BackendDriver&);
    using BackendRegistry = std::array<BackendDriver*, kGpuBackendCount>;
    void register_compiled_backends(BackendRegistry&);
    LFS_CORE_API BackendDriver& backend_driver(GpuBackend);
    LFS_CORE_API void clear_backend_activation_cache() noexcept;
} // namespace lfs::core::internal
