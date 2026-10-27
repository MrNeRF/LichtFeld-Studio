/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "../../internal/tensor_impl.hpp"
#include "../backend_driver.hpp"
#include "../backend_work_queue.hpp"
#include "../native_export.hpp"
#include "../tensor_vulkan_interop.hpp"
#include "core/assert.hpp"
#include "core/gpu_device_selection.hpp"
#include "hip_backend_ops.hpp"
#include <atomic>
#include <charconv>
#include <cstring>
#include <format>
#include <mutex>

namespace lfs::core::internal {
    namespace {
        class HipDriver final : public BackendDriver {
            std::mutex mutex_;
            int requested_ = -1, selected_ = -1;

        public:
            void complete_checked_index(void* target, SourceSite) override { activate().synchronize_stream(ExecContext{static_cast<cudaStream_t>(target)}); }
            MemoryInfo pressure_memory_info(bool) override { return backend_pressure_memory_info(*this); }
            void register_pressure_clients(MemoryPressureCoordinator&) override {}

            void* current_target() const noexcept override { return nullptr; }
            BackendRendererAccess renderer_access() const noexcept override { return {false, true, false}; }

            std::shared_ptr<TensorVulkanInteropBackend> make_vulkan_interop(VulkanInteropDevice device) override { return make_hip_vulkan_interop(device); }
            SplatPublication publication() const override { return SplatPublication::Copied; }
            std::unique_ptr<BackendWorkQueue> make_work_queue(const BackendQueueCreateInfo& info) override { return make_hip_work_queue(info); }
            std::array<uintptr_t, 2> bind_target(void* target) override {
                if (target)
                    throw std::invalid_argument("HIP Tensor accepts only its native default queue");
                const auto previous = reinterpret_cast<uintptr_t>(getCurrentCUDAStream());
                setCurrentCUDAStream(nullptr);
                return {previous, 0};
            }
            void unbind_target(std::array<uintptr_t, 2> token) noexcept override { setCurrentCUDAStream(reinterpret_cast<cudaStream_t>(token[0])); }
            void bridge_targets(void* consumer, void* producer) override { (void)activate().bridge({static_cast<cudaStream_t>(producer)}, {static_cast<cudaStream_t>(consumer)}); }
            const char* name() const noexcept override { return "HIP"; }
            bool available() override {
                try {
                    return selected_device() >= 0;
                } catch (...) { /* LFS-CENSUS-OK(empty-catch): Discovery reports an unavailable backend without creating a context. */
                    return false;
                }
            }
            bool live() noexcept override { return hip_backend_is_live(); }
            [[noreturn]] void throw_unavailable() override {
                throw TensorError(std::format("GPU backend '{}' is unavailable", name()));
            }
            int device_count() override { return hip_device_count(LFS_HIP_RUNTIME_DEFAULT_MODULE); }
            int selected_device() override {
                std::lock_guard lock(mutex_);
                if (selected_ < 0)
                    selected_ = select_hip_device(LFS_HIP_RUNTIME_DEFAULT_MODULE, requested_);
                return selected_;
            }
            void bind_thread() override {}
            std::optional<std::string> validate_options(const TensorBackendOptions& options, int& ordinal) override {
                ordinal = -1;
                if (options.hip_device.empty())
                    return {};
                const auto& value = options.hip_device;
                const auto parsed = std::from_chars(value.data(), value.data() + value.size(), ordinal);
                if (parsed.ec != std::errc{} || parsed.ptr != value.data() + value.size() || ordinal < 0)
                    return "HIP Tensor device selection requires a nonnegative device ordinal";
                return {};
            }
            void set_selection(int ordinal) noexcept override {
                std::lock_guard lock(mutex_);
                requested_ = ordinal;
                selected_ = -1;
            }
            GpuBackendOps& activate() override {
                (void)tensor_backend_options();
                static HipBackendOps* const ops = new HipBackendOps(LFS_HIP_RUNTIME_DEFAULT_MODULE, selected_device());
                ops->ensure_initialized();
                return *ops;
            }
            std::optional<GpuDeviceInfo> device_info(int ordinal) override {
                return hip_gpu_device_info(LFS_HIP_RUNTIME_DEFAULT_MODULE, ordinal < 0 ? selected_device() : ordinal);
            }
            std::vector<VulkanHardwareDevice> enumerate_hardware() override {
                std::vector<VulkanHardwareDevice> result;
                const auto count = device_count();
                for (int ordinal = 0; ordinal < count; ++ordinal) {
                    const auto native = hip_device_info(LFS_HIP_RUNTIME_DEFAULT_MODULE, ordinal);
                    VulkanHardwareDevice info;
                    info.name = native.name;
                    info.index = ordinal;
                    const auto details = device_info(ordinal);
                    if (details)
                        info.total_memory_bytes = details->total_memory_bytes;
                    result.push_back(std::move(info));
                }
                return result;
            }
            MemoryInfo memory_info(bool) override { return activate().stats(); }
            size_t allocation_bytes(size_t bytes) override { return bytes; }
            TensorExportDevice dlpack_device(const Tensor&) override { reject_dlpack_export(name()); }
            Tensor prepare_dlpack_export(const Tensor&, std::optional<int64_t>) override { reject_dlpack_export(name()); }
            BackendExportCapabilities export_capabilities() const noexcept override { return {}; }
            lfs::Status shutdown() override {
                if (!live())
                    return {};
                try {
                    activate().shutdown();
                } catch (const std::exception& error) {
                    return lfs::Status::failure(lfs::make_error(lfs::ErrorInit{
                        .code = lfs::ErrorCode::Internal,
                        .domain = lfs::ErrorDomain::Core,
                        .user_message = "HIP tensor backend shutdown failed",
                        .detail = error.what(),
                        .detection = LFS_SOURCE_SITE_CURRENT(),
                    }));
                }
                return {};
            }
        };
    } // namespace
    void register_hip_backend(BackendRegistry& registry) {
        static HipDriver driver;
        registry[static_cast<size_t>(GpuBackend::HIP)] = &driver;
    }
} // namespace lfs::core::internal
