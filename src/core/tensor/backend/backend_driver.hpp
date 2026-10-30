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
} // namespace lfs::core

namespace lfs::core::internal {
    class TensorVulkanInteropBackend;
    class GpuBackendOps;
    class BackendWorkQueue;
    struct NativeTensorAllocation;
    enum class TensorAllocationKind;
    struct BackendQueueCreateInfo;
    // Inventory/selection never activates Tensor storage. Activation is cold:
    // backend_ops caches the resulting pointer at the driver's requested scope.
    // Registered drivers have process lifetime, like the activated ops, so late exit hooks
    // can still call them. Native resources are released explicitly by shutdown.
    struct BackendRendererAccess {
        bool shares_vulkan = false, needs_vulkan_interop = false, metal = false;
    };
    enum class BackendActivationScope { Process,
                                        Thread };
    class BackendDriver {
        const bool preserves_home_;

    protected:
        explicit BackendDriver(bool preserves_home = false) : preserves_home_(preserves_home) {}

    public:
        virtual ~BackendDriver() = default;
        virtual const char* name() const noexcept = 0;
        virtual bool available() = 0;
        virtual bool live() noexcept = 0;
        [[noreturn]] virtual void throw_unavailable();
        virtual int device_count() = 0;
        virtual std::optional<GpuDeviceInfo> device_info(int ordinal) = 0;
        virtual std::vector<VulkanHardwareDevice> enumerate_hardware() { return {}; }
        virtual std::optional<std::string> validate_options(const TensorBackendOptions&, int& selection) {
            selection = -1;
            return {};
        }
        virtual void set_selection(int) noexcept {}
        virtual int selected_device() { return 0; }
        virtual void bind_thread() {}
        virtual void* current_target() const noexcept { return nullptr; }
        virtual BackendRendererAccess renderer_access() const noexcept { return {}; }
        virtual std::array<uintptr_t, 2> bind_target(void*) = 0;
        virtual void unbind_target(std::array<uintptr_t, 2>) noexcept = 0;
        virtual void bridge_targets(void* consumer, void* producer) = 0;
        virtual GpuBackendOps& activate() = 0;
        // This immutable policy is consulted only when filling activation or binding caches.
        // Process-scoped activation must be safe to publish to other threads;
        // thread-scoped activation may also establish native thread state.
        virtual BackendActivationScope activation_scope() const noexcept { return BackendActivationScope::Process; }
        virtual MemoryInfo memory_info(bool include_pool_stats);
        virtual MemoryInfo pressure_memory_info(bool free_query);
        virtual bool pressure_recovery_blocked() const noexcept { return false; }
        virtual void register_pressure_clients(MemoryPressureCoordinator&) {}
        virtual size_t allocation_bytes(size_t bytes) = 0;
        virtual lfs::Status shutdown() { return {}; }
        virtual TensorExportCapabilities export_capabilities() const noexcept { return {}; }
        virtual TensorExportDevice dlpack_device(const Tensor&);
        virtual Tensor prepare_dlpack_export(const Tensor&, std::optional<int64_t>);
        virtual std::shared_ptr<TensorVulkanInteropBackend> make_vulkan_interop(VulkanInteropDevice);
        virtual SplatPublication publication() const;
        virtual std::unique_ptr<BackendWorkQueue> make_work_queue(const BackendQueueCreateInfo&) = 0;
        // A null home remains null, and implicit binding with no home is a no-op.
        void* canonical_home_target(void* home) const noexcept { return preserves_home_ ? home : nullptr; }
        virtual std::optional<std::array<uintptr_t, 2>> bind_tensor_target(const Tensor&, bool) { return std::nullopt; }
        virtual void synchronize_tensor_target(const Tensor&, void*) {}
        virtual void shutdown_compatibility_pool() {}
        virtual void set_pool_iteration(int) {}
        virtual NativeTensorAllocation empty_tensor_allocation(TensorAllocationKind) = 0;
        virtual void complete_checked_index(void* target, SourceSite location);
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
    using BackendRegistry = std::array<BackendDriver*, kGpuBackendCount>;
    void register_compiled_backends(BackendRegistry&);
    LFS_CORE_API BackendDriver& backend_driver(GpuBackend);
    LFS_CORE_API void clear_backend_activation_cache() noexcept;
} // namespace lfs::core::internal
