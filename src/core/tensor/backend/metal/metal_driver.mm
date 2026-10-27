/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "../../internal/tensor_impl.hpp"
#include "../backend_driver.hpp"
#include "../native_export.hpp"
#include "../tensor_vulkan_interop.hpp"
#include "core/assert.hpp"
#include "core/gpu_device_selection.hpp"
#include "metal_backend_ops.hpp"
#include "metal_context.hpp"
#include <Metal/Metal.h>
#include <atomic>
#include <charconv>
#include <cstring>
#include <format>

namespace lfs::core::internal {
namespace {
    class MetalDriver final : public BackendDriver {
    public:
        std::shared_ptr<TensorVulkanInteropBackend> make_vulkan_interop(VulkanInteropDevice device) override { return make_metal_vulkan_interop(device); }
        SplatPublication publication() const override { return SplatPublication::Copied; }
        const char* name() const noexcept override { return "Metal"; }
        bool available() override { return metal_backend_available(); }
        bool live() noexcept override { return metal_backend_live(); }
        [[noreturn]] void throw_unavailable() override
        {
            throw TensorError(std::format("GPU backend '{}' is unavailable", name()));
        }
        int device_count() override { return available() ? 1 : 0; }
        int selected_device() override { return 0; }
        void bind_thread() override { }
        void set_selection(int) noexcept override { }
        std::optional<std::string> validate_options(const TensorBackendOptions&, int& ordinal) override
        {
            ordinal = -1;
            return {};
        }
        GpuBackendOps& activate() override
        {
            if (__builtin_available(macOS 26.0, *)) {
                static MetalBackendOps* const ops = new MetalBackendOps();
                return *ops;
            }
            LFS_ASSERT_MSG(false, "GPU backend 'Metal' needs macOS 26");
            std::unreachable();
        }
        std::vector<VulkanHardwareDevice> enumerate_hardware() override
        {
            @autoreleasepool {
                id<MTLDevice> device = MTLCreateSystemDefaultDevice();
                if (!device)
                    return {};
                VulkanHardwareDevice info;
                info.name = device.name.UTF8String;
                info.discrete = !device.hasUnifiedMemory;
                info.shared_memory = device.hasUnifiedMemory;
                info.total_memory_bytes = device.recommendedMaxWorkingSetSize;
                if (!available())
                    info.tensor_unavailable_reason = "Metal tensor backend needs macOS 26 and a supported device";
                return { std::move(info) };
            }
        }
        std::optional<GpuDeviceInfo> device_info(int ordinal) override
        {
            return ordinal <= 0 ? metal_device_info() : std::nullopt;
        }
        MemoryInfo memory_info(bool) override { return activate().stats(); }
        size_t allocation_bytes(size_t bytes) override { return metal_allocation_bytes(bytes); }
        TensorExportDevice dlpack_device(const Tensor&) override { reject_dlpack_export(name()); }
        Tensor prepare_dlpack_export(const Tensor&, std::optional<int64_t>) override { reject_dlpack_export(name()); }
        BackendExportCapabilities export_capabilities() const noexcept override { return {}; }
        lfs::Status shutdown() override
        {

            try {
                internal::shutdown_metal_backend();
            } catch (...) {
                return lfs::Status::failure(lfs::make_error(lfs::ErrorInit {
                    .code = lfs::ErrorCode::Internal,
                    .domain = lfs::ErrorDomain::Core,
                    .user_message = "Metal tensor backend shutdown failed",
                    .detection = LFS_SOURCE_SITE_CURRENT(),
                }));
            }
            return {};
        }
    };
}
void register_metal_backend(BackendRegistry& registry)
{
    static MetalDriver driver;
    registry[static_cast<size_t>(GpuBackend::Metal)] = &driver;
}
}
