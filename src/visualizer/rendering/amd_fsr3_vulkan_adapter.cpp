/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "amd_fsr3_vulkan_adapter.hpp"

#include "core/logger.hpp"
#include "rendering/amd_fsr3_contract.hpp"
#include "rendering/scene_upscaler_registry.hpp"
#include "rendering/vulkan_scene_upscaler_adapter.hpp"
#include "window/vulkan_context.hpp"

#include <FidelityFX/host/backends/vk/ffx_vk.h>
#include <FidelityFX/host/ffx_fsr3upscaler.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace lfs::vis {
    namespace {
        constexpr std::string_view ADAPTER_ID = "amd-fsr3";
        constexpr VkFormat OUTPUT_FORMAT = VK_FORMAT_R8G8B8A8_UNORM;
        constexpr std::size_t VIEW_COUNT = static_cast<std::size_t>(TemporalViewId::Count);
        constexpr std::size_t SHARED_RESOURCE_COUNT = 3;

        [[nodiscard]] std::size_t viewIndex(const TemporalViewId view) {
            return static_cast<std::size_t>(view);
        }

        [[nodiscard]] FfxResourceDescription imageDescription(
            const VulkanSceneUpscalerResource& resource,
            const FfxResourceUsage additional_usage = FFX_RESOURCE_USAGE_READ_ONLY) {
            VkImageCreateInfo info{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
            info.imageType = VK_IMAGE_TYPE_2D;
            info.format = resource.format;
            info.extent = {static_cast<std::uint32_t>(resource.allocation_extent.x),
                           static_cast<std::uint32_t>(resource.allocation_extent.y), 1};
            info.mipLevels = 1;
            info.arrayLayers = 1;
            info.samples = VK_SAMPLE_COUNT_1_BIT;
            info.tiling = VK_IMAGE_TILING_OPTIMAL;
            info.usage = VK_IMAGE_USAGE_SAMPLED_BIT;
            if ((additional_usage & FFX_RESOURCE_USAGE_UAV) != 0)
                info.usage |= VK_IMAGE_USAGE_STORAGE_BIT;
            return ffxGetImageResourceDescriptionVK(resource.image, info, additional_usage);
        }

        [[nodiscard]] FfxResourceStates resourceStateForLayout(const VkImageLayout layout) {
            switch (layout) {
            case VK_IMAGE_LAYOUT_GENERAL:
                return FFX_RESOURCE_STATE_UNORDERED_ACCESS;
            case VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL:
                return FFX_RESOURCE_STATE_RENDER_TARGET;
            case VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL:
            case VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL:
                return FFX_RESOURCE_STATE_DEPTH_ATTACHEMENT;
            case VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL:
                return FFX_RESOURCE_STATE_COPY_SRC;
            case VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL:
                return FFX_RESOURCE_STATE_COPY_DEST;
            case VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL:
            case VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL:
            case VK_IMAGE_LAYOUT_DEPTH_READ_ONLY_OPTIMAL:
                return FFX_RESOURCE_STATE_COMPUTE_READ;
            default:
                return FFX_RESOURCE_STATE_COMPUTE_READ;
            }
        }

        void messageCallback(const FfxMsgType type, const wchar_t* const message) {
            std::string text;
            if (message != nullptr) {
                auto* cursor = message;
                while (*cursor != L'\0') {
                    const auto character = *cursor++;
                    text.push_back(character >= 0 && character <= 0x7f
                                       ? static_cast<char>(character)
                                       : '?');
                }
            }
            if (type == FFX_MESSAGE_TYPE_ERROR)
                LOG_ERROR("AMD FSR 3.1 runtime error: {}", text);
            else
                LOG_WARN("AMD FSR 3.1 runtime warning: {}", text);
        }

        class AmdFsr3VulkanAdapter final : public VulkanSceneUpscalerAdapter {
        public:
            ~AmdFsr3VulkanAdapter() override { shutdown(); }

            [[nodiscard]] SceneUpscalerAvailability probe(
                const SceneUpscalerProbeContext& context) const noexcept override {
                if (context.safe_mode)
                    return {.reason = SceneUpscalerAvailabilityReason::SafeMode};
                if (context.graphics_api != SceneUpscalerGraphicsApi::Vulkan)
                    return {.reason = SceneUpscalerAvailabilityReason::GraphicsApiUnsupported};
                return {.reason = SceneUpscalerAvailabilityReason::Ready};
            }

            [[nodiscard]] SceneUpscalerAvailability initialize(VulkanContext& context) noexcept override {
                if (initialized_)
                    return {.reason = SceneUpscalerAvailabilityReason::Ready};
                if (context.physicalDevice() == VK_NULL_HANDLE || context.device() == VK_NULL_HANDLE ||
                    context.allocator() == VK_NULL_HANDLE)
                    return {.reason = SceneUpscalerAvailabilityReason::ProbeFailed};

                device_ = context.device();
                physical_device_ = context.physicalDevice();
                allocator_ = context.allocator();
                const std::size_t max_contexts =
                    VIEW_COUNT * (FFX_FSR3UPSCALER_CONTEXT_COUNT + 1);
                scratch_.resize(ffxGetScratchMemorySizeVK(physical_device_, max_contexts));
                VkDeviceContext device_context{device_, physical_device_, vkGetDeviceProcAddr};
                const auto result = ffxGetInterfaceVK(&backend_,
                                                      ffxGetDeviceVK(&device_context),
                                                      scratch_.data(),
                                                      scratch_.size(),
                                                      max_contexts);
                if (result != FFX_OK) {
                    clearBackend();
                    return {.reason = SceneUpscalerAvailabilityReason::ProbeFailed};
                }
                initialized_ = true;
                LOG_INFO("AMD FidelityFX FSR 3.1 Vulkan adapter initialized");
                return {.reason = SceneUpscalerAvailabilityReason::Ready};
            }

            [[nodiscard]] bool record(const VkCommandBuffer command_buffer,
                                      const VulkanSceneUpscalerDispatch& dispatch) noexcept override {
                if (!initialized_ || command_buffer == VK_NULL_HANDLE || !validAmdFsr3Dispatch(dispatch))
                    return false;
                auto& state = views_[viewIndex(dispatch.view)];
                if (!ensureContext(command_buffer, dispatch, state))
                    return false;

                FfxFsr3UpscalerDispatchDescription parameters{};
                parameters.commandList = ffxGetCommandListVK(command_buffer);
                parameters.color = wrap(dispatch.color, L"LFS FSR input color");
                parameters.depth = wrap(dispatch.depth, L"LFS FSR input depth");
                parameters.motionVectors = wrap(dispatch.motion, L"LFS FSR input motion");
                parameters.output = wrap(state.outputResource(),
                                         L"LFS FSR output",
                                         FFX_RESOURCE_STATE_UNORDERED_ACCESS,
                                         FFX_RESOURCE_USAGE_UAV);
                parameters.dilatedDepth = sharedResource(state, 0);
                parameters.dilatedMotionVectors = sharedResource(state, 1);
                parameters.reconstructedPrevNearestDepth = sharedResource(state, 2);
                parameters.jitterOffset = {dispatch.jitter_pixels.x, dispatch.jitter_pixels.y};
                parameters.motionVectorScale = {1.0f, 1.0f};
                parameters.renderSize = dimensions(dispatch.color.valid_extent);
                parameters.upscaleSize = dimensions(dispatch.output_extent);
                parameters.enableSharpening = false;
                parameters.sharpness = 0.0f;
                parameters.frameTimeDelta = dispatch.frame_time_seconds * 1000.0f;
                parameters.preExposure = dispatch.exposure;
                parameters.reset = state.reset_pending || dispatch.reset_reasons != TemporalResetReason::None;
                parameters.cameraNear = dispatch.camera_near;
                parameters.cameraFar = dispatch.camera_far;
                parameters.cameraFovAngleVertical = dispatch.camera_vertical_fov_radians;
                parameters.viewSpaceToMetersFactor = dispatch.view_space_to_meters;
                const auto result = ffxFsr3UpscalerContextDispatch(&state.context, &parameters);
                if (result != FFX_OK) {
                    LOG_WARN("AMD FSR 3.1 dispatch failed: {}", static_cast<int>(result));
                    return false;
                }
                state.reset_pending = false;
                ++state.generation;
                return true;
            }

            [[nodiscard]] VulkanSceneUpscalerOutput output(const TemporalViewId view) const noexcept override {
                const auto& state = views_[viewIndex(view)];
                return state.output_image == VK_NULL_HANDLE
                           ? VulkanSceneUpscalerOutput{}
                           : VulkanSceneUpscalerOutput{.color = state.outputResource()};
            }

            void reset(const TemporalViewId view, const TemporalResetReason reasons) noexcept override {
                if (reasons != TemporalResetReason::None)
                    views_[viewIndex(view)].reset_pending = true;
            }

            void shutdown() noexcept override {
                if (!initialized_)
                    return;
                if (device_ != VK_NULL_HANDLE)
                    vkDeviceWaitIdle(device_);
                for (auto& view : views_)
                    destroyView(view);
                clearBackend();
                LOG_DEBUG("AMD FidelityFX FSR 3.1 Vulkan adapter shut down");
            }

        private:
            struct ViewState {
                FfxFsr3UpscalerContext context{};
                std::array<FfxResourceInternal, SHARED_RESOURCE_COUNT> shared{};
                std::size_t shared_count = 0;
                FfxUInt32 shared_effect_context_id = 0;
                VkImage output_image = VK_NULL_HANDLE;
                VkImageView output_view = VK_NULL_HANDLE;
                VmaAllocation output_allocation = VK_NULL_HANDLE;
                glm::ivec2 render_extent{0, 0};
                glm::ivec2 output_extent{0, 0};
                bool context_created = false;
                bool shared_context_created = false;
                bool reset_pending = true;
                std::uint64_t generation = 0;

                [[nodiscard]] VulkanSceneUpscalerResource outputResource() const noexcept {
                    return {.image = output_image,
                            .view = output_view,
                            .format = OUTPUT_FORMAT,
                            .layout = VK_IMAGE_LAYOUT_GENERAL,
                            .valid_extent = output_extent,
                            .allocation_extent = output_extent,
                            .generation = generation};
                }
            };

            [[nodiscard]] static FfxDimensions2D dimensions(const glm::ivec2 extent) {
                return {static_cast<std::uint32_t>(extent.x), static_cast<std::uint32_t>(extent.y)};
            }

            [[nodiscard]] static FfxResource wrap(
                const VulkanSceneUpscalerResource& resource,
                const wchar_t* const name,
                const FfxResourceStates state,
                const FfxResourceUsage usage = FFX_RESOURCE_USAGE_READ_ONLY) {
                return ffxGetResourceVK(resource.image, imageDescription(resource, usage), name, state);
            }

            [[nodiscard]] static FfxResource wrap(
                const VulkanSceneUpscalerResource& resource,
                const wchar_t* const name) {
                return wrap(resource, name, resourceStateForLayout(resource.layout));
            }

            [[nodiscard]] FfxResource sharedResource(ViewState& state, const std::size_t index) {
                return backend_.fpGetResource(&backend_, state.shared[index]);
            }

            void clearBackend() noexcept {
                backend_ = {};
                scratch_.clear();
                allocator_ = VK_NULL_HANDLE;
                physical_device_ = VK_NULL_HANDLE;
                device_ = VK_NULL_HANDLE;
                initialized_ = false;
            }

            void destroyView(ViewState& state) noexcept {
                if (state.shared_context_created) {
                    for (std::size_t index = 0; index < state.shared_count; ++index)
                        backend_.fpDestroyResource(&backend_, state.shared[index],
                                                   state.shared_effect_context_id);
                    backend_.fpDestroyBackendContext(&backend_, state.shared_effect_context_id);
                }
                if (state.context_created)
                    ffxFsr3UpscalerContextDestroy(&state.context);
                if (state.output_view != VK_NULL_HANDLE)
                    vkDestroyImageView(device_, state.output_view, nullptr);
                if (state.output_image != VK_NULL_HANDLE)
                    vmaDestroyImage(allocator_, state.output_image, state.output_allocation);
                state = {};
            }

            [[nodiscard]] bool ensureContext(const VkCommandBuffer command_buffer,
                                             const VulkanSceneUpscalerDispatch& dispatch,
                                             ViewState& state) noexcept {
                if (state.context_created && state.render_extent == dispatch.color.valid_extent &&
                    state.output_extent == dispatch.output_extent)
                    return true;
                if (state.context_created || state.output_image != VK_NULL_HANDLE) {
                    vkDeviceWaitIdle(device_);
                    destroyView(state);
                }

                FfxFsr3UpscalerContextDescription description{};
                description.flags = FFX_FSR3UPSCALER_ENABLE_AUTO_EXPOSURE |
                                    FFX_FSR3UPSCALER_ENABLE_DYNAMIC_RESOLUTION;
                if (dispatch.motion_includes_jitter)
                    description.flags |= FFX_FSR3UPSCALER_ENABLE_MOTION_VECTORS_JITTER_CANCELLATION;
                description.maxRenderSize = dimensions(dispatch.color.allocation_extent);
                description.maxUpscaleSize = dimensions(dispatch.output_extent);
                description.fpMessage = &messageCallback;
                description.backendInterface = backend_;
                if (ffxFsr3UpscalerContextCreate(&state.context, &description) != FFX_OK)
                    return false;
                state.context_created = true;
                if (backend_.fpCreateBackendContext(&backend_, FFX_EFFECT_SHAREDRESOURCES, nullptr,
                                                    &state.shared_effect_context_id) != FFX_OK) {
                    destroyView(state);
                    return false;
                }
                state.shared_context_created = true;

                FfxFsr3UpscalerSharedResourceDescriptions shared{};
                if (ffxFsr3UpscalerGetSharedResourceDescriptions(&state.context, &shared) != FFX_OK ||
                    !createShared(shared.dilatedDepth, state) ||
                    !createShared(shared.dilatedMotionVectors, state) ||
                    !createShared(shared.reconstructedPrevNearestDepth, state) ||
                    !createOutput(command_buffer, dispatch.output_extent, state)) {
                    destroyView(state);
                    return false;
                }
                state.render_extent = dispatch.color.valid_extent;
                state.output_extent = dispatch.output_extent;
                state.reset_pending = true;
                ++state.generation;
                LOG_DEBUG("AMD FSR 3.1 context ready: view={} render={}x{} output={}x{}",
                          viewIndex(dispatch.view), state.render_extent.x, state.render_extent.y,
                          state.output_extent.x, state.output_extent.y);
                return true;
            }

            [[nodiscard]] bool createShared(const FfxCreateResourceDescription& description,
                                            ViewState& state) {
                if (state.shared_count >= state.shared.size())
                    return false;
                auto& resource = state.shared[state.shared_count];
                if (backend_.fpCreateResource(&backend_, &description,
                                              state.shared_effect_context_id, &resource) != FFX_OK)
                    return false;
                ++state.shared_count;
                return true;
            }

            [[nodiscard]] bool createOutput(const VkCommandBuffer command_buffer,
                                            const glm::ivec2 extent,
                                            ViewState& state) noexcept {
                VkImageCreateInfo image_info{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
                image_info.imageType = VK_IMAGE_TYPE_2D;
                image_info.format = OUTPUT_FORMAT;
                image_info.extent = {static_cast<std::uint32_t>(extent.x),
                                     static_cast<std::uint32_t>(extent.y), 1};
                image_info.mipLevels = 1;
                image_info.arrayLayers = 1;
                image_info.samples = VK_SAMPLE_COUNT_1_BIT;
                image_info.tiling = VK_IMAGE_TILING_OPTIMAL;
                image_info.usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT |
                                   VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
                VmaAllocationCreateInfo allocation_info{};
                allocation_info.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
                if (vmaCreateImage(allocator_, &image_info, &allocation_info, &state.output_image,
                                   &state.output_allocation, nullptr) != VK_SUCCESS)
                    return false;
                VkImageViewCreateInfo view_info{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
                view_info.image = state.output_image;
                view_info.viewType = VK_IMAGE_VIEW_TYPE_2D;
                view_info.format = OUTPUT_FORMAT;
                view_info.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
                if (vkCreateImageView(device_, &view_info, nullptr, &state.output_view) != VK_SUCCESS)
                    return false;
                VkImageMemoryBarrier barrier{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
                barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
                barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
                barrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
                barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                barrier.image = state.output_image;
                barrier.subresourceRange = view_info.subresourceRange;
                vkCmdPipelineBarrier(command_buffer, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                                     VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr, 0,
                                     nullptr, 1, &barrier);
                return true;
            }

            VkDevice device_ = VK_NULL_HANDLE;
            VkPhysicalDevice physical_device_ = VK_NULL_HANDLE;
            VmaAllocator allocator_ = VK_NULL_HANDLE;
            FfxInterface backend_{};
            std::vector<std::byte> scratch_;
            std::array<ViewState, VIEW_COUNT> views_{};
            bool initialized_ = false;
        };

        SceneUpscalerAdapterFactoryResult makeAmdFsr3Adapter() noexcept {
            try {
                return std::make_unique<AmdFsr3VulkanAdapter>();
            } catch (...) {
                return std::unexpected(SceneUpscalerAvailabilityReason::ProbeFailed);
            }
        }
    } // namespace

    void registerAmdFsr3VulkanAdapter() {
        static const bool registered = optionalSceneUpscalerRegistry().registerAdapter(
            {.id = std::string(ADAPTER_ID),
             .label_key = "preferences.scene_upscaler_amd_fsr3",
             .requirements = {.depth = true, .motion_vectors = true, .jitter = true, .history = true}},
            &makeAmdFsr3Adapter);
        if (!registered)
            LOG_WARN("AMD FidelityFX FSR 3.1 adapter registration was rejected");
    }
} // namespace lfs::vis
