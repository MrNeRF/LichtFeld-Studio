/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "vulkan_graphics_context.hpp"

#include "core/gpu_device_selection.hpp"
#include "core/logger.hpp"
#include "core/tensor_backend_vulkan.hpp"
#include "core/vulkan_device_selection.hpp"
#include "core/vulkan_helpers.hpp"
#include "preferences.hpp"

#include <utility>

namespace lfs::vis {
    std::unique_ptr<GraphicsContext> createGraphicsContext() { return std::make_unique<VulkanGraphicsContext>(); }
    namespace {
        lfs::Error graphicsError(const lfs::ErrorCode code, std::string detail) {
            return lfs::make_error({
                .code = code,
                .domain = lfs::ErrorDomain::Vulkan,
                .user_message = "Graphics operation failed",
                .detail = std::move(detail),
                .detection = LFS_SOURCE_SITE_CURRENT(),
            });
        }
    } // namespace

    VulkanGraphicsContext::VulkanGraphicsContext() = default;
    VulkanGraphicsContext::~VulkanGraphicsContext() {
        disconnectTensorBackend();
        if (headless_) {
            if (auto result = lfs::core::shutdown_gpu_backend(lfs::core::GpuBackend::Vulkan); !result)
                LOG_WARN("Failed to shut down tensor backend: {}", result.error().detail());
        }
    }

    bool VulkanGraphicsContext::initialize(SDL_Window* const window,
                                           const int framebuffer_width,
                                           const int framebuffer_height) {
        if (!context_.init(window, framebuffer_width, framebuffer_height, std::exchange(device_hooks_, {})))
            return false;
        return true;
    }

    bool VulkanGraphicsContext::initializeHeadless() {
        headless_ = context_.initHeadless();
        return headless_;
    }
    void VulkanGraphicsContext::publishPrimaryGpuState() {
        lfs::core::set_cuda_vulkan_direct_interop(context_.directTensorInteropEnabled());
        publishRenderingGpuIdentity(lfs::core::gpu_identity_string(context_.deviceUUID()), deviceName(),
                                    context_.tensorBackendDevice().complete || lfs::core::vulkan_backend_adopted());
    }
    void VulkanGraphicsContext::shutdown() { context_.shutdown(); }

    void VulkanGraphicsContext::notifyFramebufferResized(
        const int width, const int height, const GraphicsResizeIntent intent) {
        context_.notifyFramebufferResized(
            width, height,
            intent == GraphicsResizeIntent::Interactive
                ? VulkanContext::ResizeIntent::Interactive
                : VulkanContext::ResizeIntent::Exact);
    }

    bool VulkanGraphicsContext::hasPendingResize() const { return context_.hasPendingSwapchainResize(); }
    bool VulkanGraphicsContext::pendingResizeReady() const { return context_.pendingSwapchainResizeReady(); }
    double VulkanGraphicsContext::secondsUntilPendingResizeReady() const {
        return context_.secondsUntilPendingSwapchainResizeReady();
    }

    lfs::Result<std::optional<GraphicsFrame>> VulkanGraphicsContext::beginFrame(
        const std::array<float, 4>& clear_color) {
        VkClearValue clear{};
        clear.color = {{clear_color[0], clear_color[1], clear_color[2], clear_color[3]}};
        if (!context_.beginFrame(clear, frame_))
            return std::optional<GraphicsFrame>{};
        return std::optional<GraphicsFrame>{GraphicsFrame{
            .frame_slot = frame_.frame_slot,
            .width = frame_.extent.width,
            .height = frame_.extent.height,
            .generation = ++frame_generation_,
        }};
    }

    lfs::Status VulkanGraphicsContext::endFrame() {
        if (context_.endFrame())
            return {};
        return lfs::Status::failure(graphicsError(lfs::ErrorCode::Internal, context_.lastError()));
    }

    const VulkanContext::Frame* VulkanGraphicsContext::referenceFrame(const GraphicsFrame& frame) const noexcept {
        return context_.hasActiveFrame() && frame.generation == frame_generation_ ? &frame_ : nullptr;
    }

    const VulkanContext::Frame* vulkanFrameOrNull(const GraphicsContext* context, const GraphicsFrame& frame) noexcept {
        const auto* native = dynamic_cast<const VulkanGraphicsContext*>(context);
        return native ? native->referenceFrame(frame) : nullptr;
    }

    bool VulkanGraphicsContext::presentBootstrapFrame(const float r, const float g,
                                                      const float b, const float a) {
        return context_.presentBootstrapFrame(r, g, b, a);
    }
    bool VulkanGraphicsContext::hasActiveFrame() const noexcept { return context_.hasActiveFrame(); }

    lfs::Result<WindowCapture> VulkanGraphicsContext::captureFinalFrameRgba() {
        auto capture = context_.captureAndEndActiveFrameRgba();
        if (!capture)
            return graphicsError(lfs::ErrorCode::Internal, std::move(capture.error()));
        return WindowCapture{capture->width, capture->height, std::move(capture->rgba)};
    }

    lfs::Result<WindowCapture> VulkanGraphicsContext::captureSceneRgba() {
        auto capture = context_.captureAndEndActiveFrameRgba(true);
        if (!capture)
            return graphicsError(lfs::ErrorCode::Internal, std::move(capture.error()));
        return WindowCapture{capture->width, capture->height, std::move(capture->rgba)};
    }

    bool VulkanGraphicsContext::waitForNextFrameSlot() { return context_.waitForNextFrameSlot(); }
    bool VulkanGraphicsContext::waitForSubmittedFrames() { return context_.waitForSubmittedFrames(); }
    bool VulkanGraphicsContext::waitIdle() { return context_.deviceWaitIdle(); }
    RendererTerminalState VulkanGraphicsContext::terminalState() const noexcept {
        return context_.rendererTerminalState();
    }
    const std::string& VulkanGraphicsContext::lastError() const noexcept { return context_.lastError(); }
    GraphicsCapabilities VulkanGraphicsContext::capabilities() const noexcept {
        return {
            .native_metal = false,
            .mesh_rendering = true,
            .wireframe = context_.hasFillModeNonSolid(),
            .wide_lines = context_.hasWideLines(),
            .external_memory_interop = context_.externalMemoryInteropEnabled(),
            .external_semaphore_interop = context_.externalSemaphoreInteropEnabled(),
        };
    }
    void VulkanGraphicsContext::flushPipelineCache() { context_.flushPipelineCache(); }
    std::string VulkanGraphicsContext::deviceName() const {
        if (!context_.physicalDevice())
            return {};
        VkPhysicalDeviceProperties properties{};
        vkGetPhysicalDeviceProperties(context_.physicalDevice(), &properties);
        return properties.deviceName;
    }
    lfs::core::SplatTensorAllocator VulkanGraphicsContext::splatTensorAllocator(
        const bool preserve_float_shN) {
        return context_.tensorInterop().splat_allocator(preserve_float_shN);
    }

    void VulkanGraphicsContext::connectTensorBackend() {
        if (!lfs::core::tensor_backend_shares_vulkan_device())
            return;
        const auto backend = lfs::core::configured_gpu_backend();
        if (backend == lfs::core::GpuBackend::CUDA && !context_.directTensorInteropEnabled())
            return;
        const auto requested = backend == lfs::core::GpuBackend::Vulkan
                                   ? lfs::core::tensor_backend_options().vulkan_device
                                   : std::string{};
        if (!requested.empty()) {
            const auto devices = lfs::core::enumerate_vulkan_physical_devices(context_.instance());
            const auto found = std::find(devices.devices.begin(), devices.devices.end(), context_.physicalDevice());
            if (found == devices.devices.end() ||
                !lfs::core::matches_vulkan_device_preference(requested, found - devices.devices.begin(), context_.deviceUUID())) {
                LOG_INFO("Tensor and rendering Vulkan GPUs differ; using host-staged transfers");
                return;
            }
        }
        const auto& device = context_.tensorBackendDevice();
        if (!device.complete) {
            LOG_INFO("Tensor Vulkan backend keeps its own device: the window device has no spare compute queue or lacks a required feature");
            return;
        }
        const lfs::core::VulkanDeviceHandles handles{
            .instance = context_.instance(),
            .physical_device = context_.physicalDevice(),
            .device = context_.device(),
            .queue = device.queue,
            .queue_family = device.queue_family,
            .sharing_queue_families = {context_.graphicsQueueFamily(), context_.computeQueueFamily(), device.queue_family},
            .sharing_queue_family_count = 3,
            .shader_atomic_float = device.shader_atomic_float,
            .memory_budget = false,
            .shader_float64 = device.shader_float64,
            .shader_float16 = device.shader_float16,
            .vulkan_memory_model = device.vulkan_memory_model,
            .vulkan_memory_model_device_scope = device.vulkan_memory_model_device_scope,
            .cooperative_matrix = device.cooperative_matrix,
            .external_memory = context_.externalMemoryInteropEnabled(),
            .external_semaphore = context_.externalSemaphoreInteropEnabled(),
#ifdef __APPLE__
            .metal_objects = context_.metalObjectsInteropEnabled(),
#endif
            .consumer_queue = context_.graphicsQueue(),
            .consumer_queue_mutex = &context_.graphicsQueueMutex(),
        };
        if (const auto status = lfs::core::adopt_vulkan_device(handles, device.subgroup_size_control); !status) {
            LOG_WARN("Tensor Vulkan backend keeps its own device: {}", lfs::format_for_developer(status.error()));
            return;
        }
        tensor_backend_adopted_ = true;
        LOG_INFO("Tensor Vulkan backend runs on the window device (queue family {})", device.queue_family);
    }

    void VulkanGraphicsContext::disconnectTensorBackend() {
        if (!tensor_backend_adopted_)
            return;
        tensor_backend_adopted_ = false;
        if (const auto status = lfs::core::shutdown_gpu_backend(lfs::core::GpuBackend::Vulkan); !status)
            LOG_WARN("Tensor Vulkan backend shutdown failed before the window device is destroyed: {}",
                     lfs::format_for_developer(status.error()));
    }

    VulkanContext* vulkanContextOrNull(GraphicsContext* const context) noexcept {
        const auto implementation = dynamic_cast<VulkanGraphicsContext*>(context);
        return implementation ? &implementation->vulkanContext() : nullptr;
    }

} // namespace lfs::vis
