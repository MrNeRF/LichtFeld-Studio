/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "../internal/expression_runtime.hpp"
#include "../internal/tensor_impl.hpp"
#include "backend_driver.hpp"
#include "backend_work_queue.hpp"
#include "core/assert.hpp"
#include "gpu_backend_ops.hpp"
#include "native_export.hpp"
#include "tensor_vulkan_interop.hpp"
#include <format>

namespace lfs::core::internal {
    namespace {
        class UnavailableDriver final : public BackendDriver {
            const char* name_;
            int default_device_;
            SplatPublication publication_;

        public:
            explicit UnavailableDriver(const char* name, int device = -1, SplatPublication publication = SplatPublication::Copied)
                : name_(name), default_device_(device), publication_(publication) {}
            std::shared_ptr<TensorVulkanInteropBackend> make_vulkan_interop(VulkanInteropDevice) override { throw_unavailable(); }
            SplatPublication publication() const override {
                return publication_ == SplatPublication::RendererStorage && !cuda_vulkan_direct_interop()
                           ? SplatPublication::Copied
                           : publication_;
            }
            std::unique_ptr<BackendWorkQueue> make_work_queue(const BackendQueueCreateInfo&) override { throw_unavailable(); }
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
                if (default_device_ >= 0)
                    return default_device_;
                throw std::runtime_error(std::format("GPU backend '{}' is not compiled into this build", name_));
            }
            void bind_thread() override {}
            void* current_target() const noexcept override { return default_device_ >= 0 ? getCurrentCUDAStream() : nullptr; }
            BackendRendererAccess renderer_access() const noexcept override { return {publication_ != SplatPublication::Copied, publication_ == SplatPublication::RendererStorage, false}; }
            GpuBackendOps& activate() override {
                LFS_ASSERT_MSG(false, std::format("GPU backend '{}' is not compiled into this build", name_));
                std::unreachable();
            }
            MemoryInfo memory_info(bool) override { return {}; }
            size_t allocation_bytes(size_t) override {
                throw std::runtime_error(std::format("GPU backend '{}' is not compiled into this build", name_));
            }
            lfs::Status shutdown() override { return {}; }
            TensorExportDevice dlpack_device(const Tensor&) override { reject_dlpack_export(name()); }
            Tensor prepare_dlpack_export(const Tensor&, std::optional<int64_t>) override { reject_dlpack_export(name()); }
            BackendExportCapabilities export_capabilities() const noexcept override { return {}; }
        };
        const BackendRegistry& drivers() {
            static const BackendRegistry registry = [] {
                static UnavailableDriver cuda{"CUDA", 0, SplatPublication::RendererStorage}, vulkan{"Vulkan", -1, SplatPublication::Shared}, metal{"Metal"}, hip{"HIP"};
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
