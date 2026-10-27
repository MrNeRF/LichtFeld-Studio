/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once

#include "core/gpu_device_info.hpp"
#include "core/tensor_backend.hpp"
#include "core/tensor_backend_vulkan.hpp"
#include "core/tensor_protocol_export.hpp"
#include <array>
#include <memory>
#include <optional>
#include <string>

namespace lfs::core {
    struct VulkanInteropDevice;
    enum class SplatPublication : uint8_t;
} // namespace lfs::core

namespace lfs::core::internal {
    class TensorVulkanInteropBackend;
    class GpuBackendOps;
    using BackendExportCapabilities = TensorExportCapabilities;
    // Inventory/selection never activates Tensor storage. Activation is cold:
    // backend_ops caches the resulting pointer in the calling thread.
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
        virtual void bind_thread() = 0;
        virtual GpuBackendOps& activate() = 0;
        virtual MemoryInfo memory_info(bool include_pool_stats) = 0;
        virtual size_t allocation_bytes(size_t bytes) = 0;
        virtual lfs::Status shutdown() = 0;
        virtual BackendExportCapabilities export_capabilities() const noexcept = 0;
        virtual TensorExportDevice dlpack_device(const Tensor&) = 0;
        virtual Tensor prepare_dlpack_export(const Tensor&, std::optional<int64_t>) = 0;
        virtual std::shared_ptr<TensorVulkanInteropBackend> make_vulkan_interop(VulkanInteropDevice) = 0;
        virtual SplatPublication publication() const = 0;
    };
    using BackendRegistry = std::array<BackendDriver*, kGpuBackendCount>;
    void register_compiled_backends(BackendRegistry&);
    LFS_CORE_API BackendDriver& backend_driver(GpuBackend);
    LFS_CORE_API void clear_backend_activation_cache() noexcept;
} // namespace lfs::core::internal
