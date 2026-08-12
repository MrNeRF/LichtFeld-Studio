/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "nvidia_dlss_vulkan_adapter.hpp"

#include "core/logger.hpp"
#include "core/user_paths.hpp"
#include "rendering/scene_upscaler_registry.hpp"
#include "rendering/vulkan_scene_upscaler_adapter.hpp"
#include "window/vulkan_context.hpp"

#include <nvsdk_ngx_helpers.h>
#include <nvsdk_ngx_helpers_vk.h>
#include <nvsdk_ngx_vk.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <memory>
#include <mutex>
#include <optional>
#include <string_view>
#include <system_error>

namespace lfs::vis {
    namespace {
        constexpr std::uint32_t NVIDIA_VENDOR_ID = 0x10de;
        constexpr VkFormat OUTPUT_FORMAT = VK_FORMAT_R8G8B8A8_UNORM;
        constexpr std::string_view ADAPTER_ID = "nvidia-dlss";
        std::atomic_bool bootstrap_ready{true};
        std::mutex recommended_scales_mutex;
        std::array<float, 3> recommended_scales{};
        glm::ivec2 recommended_scales_output_extent{0, 0};

        [[nodiscard]] NVSDK_NGX_FeatureDiscoveryInfo discoveryInfo(
            const wchar_t* const application_data_path) noexcept {
            NVSDK_NGX_FeatureDiscoveryInfo info{};
            info.SDKVersion = NVSDK_NGX_Version_API;
            info.FeatureID = NVSDK_NGX_Feature_SuperSampling;
            info.Identifier.IdentifierType = NVSDK_NGX_Application_Identifier_Type_Project_Id;
            info.Identifier.v.ProjectDesc.ProjectId = LFS_NVIDIA_DLSS_PROJECT_ID;
            info.Identifier.v.ProjectDesc.EngineType = NVSDK_NGX_ENGINE_TYPE_CUSTOM;
            info.Identifier.v.ProjectDesc.EngineVersion = "LichtFeld Studio";
            info.ApplicationDataPath = application_data_path;
            return info;
        }

        [[nodiscard]] std::optional<std::wstring> discoveryDataPath() {
            const auto paths = lfs::core::UserPaths::resolve();
            if (!paths)
                return std::nullopt;
            const auto data_path = paths->cacheDir() / "ngx";
            std::error_code error;
            std::filesystem::create_directories(data_path, error);
            if (error) {
                LOG_WARN("Cannot create NGX discovery directory '{}': {}",
                         data_path.string(),
                         error.message());
                return std::nullopt;
            }
            return data_path.wstring();
        }

        [[nodiscard]] std::vector<std::string> extensionNames(
            const uint32_t count,
            const VkExtensionProperties* const properties) {
            std::vector<std::string> names;
            names.reserve(count);
            for (uint32_t index = 0; index < count; ++index)
                names.emplace_back(properties[index].extensionName);
            return names;
        }

        [[nodiscard]] NVSDK_NGX_PerfQuality_Value ngxQuality(
            const SceneTemporalQuality quality) {
            switch (quality) {
            case SceneTemporalQuality::Performance:
                return NVSDK_NGX_PerfQuality_Value_MaxPerf;
            case SceneTemporalQuality::Quality:
                return NVSDK_NGX_PerfQuality_Value_MaxQuality;
            case SceneTemporalQuality::Balanced:
                return NVSDK_NGX_PerfQuality_Value_Balanced;
            }
            return NVSDK_NGX_PerfQuality_Value_Balanced;
        }

        void updateRecommendedScales(NVSDK_NGX_Parameter* const parameters,
                                     const glm::ivec2 output_extent) {
            std::lock_guard lock(recommended_scales_mutex);
            if (recommended_scales_output_extent == output_extent)
                return;

            std::array<float, 3> scales{};
            constexpr std::array qualities{
                NVSDK_NGX_PerfQuality_Value_MaxPerf,
                NVSDK_NGX_PerfQuality_Value_Balanced,
                NVSDK_NGX_PerfQuality_Value_MaxQuality,
            };
            for (std::size_t index = 0; index < qualities.size(); ++index) {
                unsigned int optimal_width = 0;
                unsigned int optimal_height = 0;
                unsigned int maximum_width = 0;
                unsigned int maximum_height = 0;
                unsigned int minimum_width = 0;
                unsigned int minimum_height = 0;
                float sharpness = 0.0f;
                const auto result = NGX_DLSS_GET_OPTIMAL_SETTINGS(
                    parameters,
                    static_cast<unsigned int>(output_extent.x),
                    static_cast<unsigned int>(output_extent.y),
                    qualities[index],
                    &optimal_width,
                    &optimal_height,
                    &maximum_width,
                    &maximum_height,
                    &minimum_width,
                    &minimum_height,
                    &sharpness);
                if (NVSDK_NGX_FAILED(result) || optimal_width == 0 || optimal_height == 0)
                    continue;
                const float width_scale = static_cast<float>(optimal_width) /
                                          static_cast<float>(output_extent.x);
                const float height_scale = static_cast<float>(optimal_height) /
                                           static_cast<float>(output_extent.y);
                scales[index] = std::clamp(std::min(width_scale, height_scale), 0.25f, 1.0f);
            }
            recommended_scales = scales;
            recommended_scales_output_extent = output_extent;
            LOG_DEBUG("NGX DLSS recommended input scales for {}x{}: performance={:.3f}, "
                      "balanced={:.3f}, quality={:.3f}",
                      output_extent.x,
                      output_extent.y,
                      scales[0],
                      scales[1],
                      scales[2]);
        }

        [[nodiscard]] std::size_t viewIndex(const TemporalViewId view) {
            return static_cast<std::size_t>(view);
        }

        class NvidiaDlssVulkanAdapter final : public VulkanSceneUpscalerAdapter {
        public:
            ~NvidiaDlssVulkanAdapter() override {
                shutdown();
            }

            [[nodiscard]] SceneUpscalerAvailability probe(
                const SceneUpscalerProbeContext& context) const noexcept override {
                if (context.safe_mode)
                    return {.reason = SceneUpscalerAvailabilityReason::SafeMode};
                if (context.graphics_api != SceneUpscalerGraphicsApi::Vulkan)
                    return {.reason = SceneUpscalerAvailabilityReason::GraphicsApiUnsupported};
                if (!nvidiaDlssVulkanBootstrapReady())
                    return {.reason = SceneUpscalerAvailabilityReason::RuntimeMissing};
                if (context.vendor_id != NVIDIA_VENDOR_ID)
                    return {.reason = SceneUpscalerAvailabilityReason::DeviceUnsupported};
                return {.reason = SceneUpscalerAvailabilityReason::Ready};
            }

            [[nodiscard]] SceneUpscalerAvailability initialize(
                VulkanContext& context) noexcept override {
                if (initialized_)
                    return {.reason = SceneUpscalerAvailabilityReason::Ready};
                if (context.instance() == VK_NULL_HANDLE ||
                    context.physicalDevice() == VK_NULL_HANDLE ||
                    context.device() == VK_NULL_HANDLE || context.allocator() == VK_NULL_HANDLE) {
                    return {.reason = SceneUpscalerAvailabilityReason::ProbeFailed};
                }

                std::error_code error;
                const auto paths = lfs::core::UserPaths::resolve();
                if (!paths)
                    return {.reason = SceneUpscalerAvailabilityReason::ProbeFailed};
                const auto data_path = paths->cacheDir() / "ngx";
                std::filesystem::create_directories(data_path, error);
                if (error) {
                    LOG_ERROR("Cannot create NGX cache directory '{}': {}",
                              data_path.string(),
                              error.message());
                    return {.reason = SceneUpscalerAvailabilityReason::ProbeFailed};
                }
                std::wstring wide_data_path;
                try {
                    wide_data_path = data_path.wstring();
                } catch (const std::exception& exception) {
                    LOG_ERROR("Cannot convert NGX cache path '{}': {}", data_path.string(), exception.what());
                    return {.reason = SceneUpscalerAvailabilityReason::ProbeFailed};
                }
                const auto result = NVSDK_NGX_VULKAN_Init_with_ProjectID(
                    LFS_NVIDIA_DLSS_PROJECT_ID,
                    NVSDK_NGX_ENGINE_TYPE_CUSTOM,
                    "LichtFeld Studio",
                    wide_data_path.c_str(),
                    context.instance(),
                    context.physicalDevice(),
                    context.device(),
                    vkGetInstanceProcAddr,
                    vkGetDeviceProcAddr);
                if (NVSDK_NGX_FAILED(result)) {
                    LOG_WARN("NGX Vulkan initialization failed: {:#x}",
                             static_cast<unsigned int>(result));
                    return {.reason = SceneUpscalerAvailabilityReason::RuntimeMissing};
                }

                context_ = &context;
                device_ = context.device();
                allocator_ = context.allocator();
                const auto capability_result =
                    NVSDK_NGX_VULKAN_GetCapabilityParameters(&parameters_);
                int available = 0;
                if (NVSDK_NGX_FAILED(capability_result) || parameters_ == nullptr ||
                    NVSDK_NGX_FAILED(NVSDK_NGX_Parameter_GetI(
                        parameters_, NVSDK_NGX_Parameter_SuperSampling_Available, &available)) ||
                    available == 0) {
                    LOG_WARN("NGX reports that Super Sampling is unavailable");
                    if (parameters_ != nullptr)
                        NVSDK_NGX_VULKAN_DestroyParameters(parameters_);
                    NVSDK_NGX_VULKAN_Shutdown1(device_);
                    context_ = nullptr;
                    device_ = VK_NULL_HANDLE;
                    allocator_ = VK_NULL_HANDLE;
                    parameters_ = nullptr;
                    return {.reason = SceneUpscalerAvailabilityReason::DeviceUnsupported};
                }
                initialized_ = true;
                LOG_INFO("NVIDIA DLSS Vulkan adapter initialized");
                return {.reason = SceneUpscalerAvailabilityReason::Ready};
            }

            [[nodiscard]] bool record(
                const VkCommandBuffer command_buffer,
                const VulkanSceneUpscalerDispatch& dispatch) noexcept override {
                if (!initialized_ || parameters_ == nullptr || command_buffer == VK_NULL_HANDLE ||
                    dispatch.color.format == VK_FORMAT_UNDEFINED ||
                    dispatch.depth.format == VK_FORMAT_UNDEFINED ||
                    dispatch.motion.format == VK_FORMAT_UNDEFINED) {
                    return false;
                }
                auto& state = views_[viewIndex(dispatch.view)];
                if (!ensureFeature(command_buffer, dispatch, state))
                    return false;

                const VkImageSubresourceRange color_range{
                    .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                    .baseMipLevel = 0,
                    .levelCount = 1,
                    .baseArrayLayer = 0,
                    .layerCount = 1,
                };
                auto color = NVSDK_NGX_Create_ImageView_Resource_VK(
                    dispatch.color.view,
                    dispatch.color.image,
                    color_range,
                    dispatch.color.format,
                    static_cast<unsigned int>(dispatch.color.valid_extent.x),
                    static_cast<unsigned int>(dispatch.color.valid_extent.y),
                    false);
                auto depth = NVSDK_NGX_Create_ImageView_Resource_VK(
                    dispatch.depth.view,
                    dispatch.depth.image,
                    color_range,
                    dispatch.depth.format,
                    static_cast<unsigned int>(dispatch.depth.valid_extent.x),
                    static_cast<unsigned int>(dispatch.depth.valid_extent.y),
                    false);
                auto motion = NVSDK_NGX_Create_ImageView_Resource_VK(
                    dispatch.motion.view,
                    dispatch.motion.image,
                    color_range,
                    dispatch.motion.format,
                    static_cast<unsigned int>(dispatch.motion.valid_extent.x),
                    static_cast<unsigned int>(dispatch.motion.valid_extent.y),
                    false);
                auto output = NVSDK_NGX_Create_ImageView_Resource_VK(
                    state.output_view,
                    state.output_image,
                    color_range,
                    OUTPUT_FORMAT,
                    static_cast<unsigned int>(dispatch.output_extent.x),
                    static_cast<unsigned int>(dispatch.output_extent.y),
                    true);

                NVSDK_NGX_VK_DLSS_Eval_Params evaluation{};
                evaluation.Feature.pInColor = &color;
                evaluation.Feature.pInOutput = &output;
                evaluation.pInDepth = &depth;
                evaluation.pInMotionVectors = &motion;
                evaluation.InJitterOffsetX = dispatch.jitter_pixels.x;
                evaluation.InJitterOffsetY = dispatch.jitter_pixels.y;
                evaluation.InRenderSubrectDimensions = {
                    static_cast<unsigned int>(dispatch.color.valid_extent.x),
                    static_cast<unsigned int>(dispatch.color.valid_extent.y)};
                evaluation.InReset = state.reset_pending ? 1 : 0;
                evaluation.InMVScaleX = 1.0f;
                evaluation.InMVScaleY = 1.0f;
                evaluation.InPreExposure = dispatch.exposure;
                evaluation.InExposureScale = 1.0f;
                evaluation.InFrameTimeDeltaInMsec = dispatch.frame_time_seconds * 1000.0f;
                const auto result = NGX_VULKAN_EVALUATE_DLSS_EXT(
                    command_buffer, state.feature, parameters_, &evaluation);
                if (NVSDK_NGX_FAILED(result)) {
                    LOG_WARN("NGX DLSS evaluate failed: {:#x}",
                             static_cast<unsigned int>(result));
                    return false;
                }
                state.reset_pending = false;
                ++state.generation;
                return true;
            }

            [[nodiscard]] VulkanSceneUpscalerOutput output(
                const TemporalViewId view) const noexcept override {
                const auto& state = views_[viewIndex(view)];
                if (state.output_image == VK_NULL_HANDLE || state.output_view == VK_NULL_HANDLE)
                    return {};
                return {.color = {
                            .image = state.output_image,
                            .view = state.output_view,
                            .format = OUTPUT_FORMAT,
                            .layout = VK_IMAGE_LAYOUT_GENERAL,
                            .valid_extent = state.output_extent,
                            .allocation_extent = state.output_extent,
                            .generation = state.generation,
                        }};
            }

            void reset(const TemporalViewId view,
                       const TemporalResetReason reasons) noexcept override {
                if (reasons != TemporalResetReason::None)
                    views_[viewIndex(view)].reset_pending = true;
            }

            void shutdown() noexcept override {
                if (!initialized_)
                    return;
                if (device_ != VK_NULL_HANDLE)
                    vkDeviceWaitIdle(device_);
                for (auto& state : views_)
                    destroyView(state);
                if (parameters_ != nullptr)
                    NVSDK_NGX_VULKAN_DestroyParameters(parameters_);
                NVSDK_NGX_VULKAN_Shutdown1(device_);
                parameters_ = nullptr;
                allocator_ = VK_NULL_HANDLE;
                device_ = VK_NULL_HANDLE;
                context_ = nullptr;
                initialized_ = false;
                LOG_DEBUG("NVIDIA DLSS Vulkan adapter shut down");
            }

        private:
            struct ViewState {
                NVSDK_NGX_Handle* feature = nullptr;
                VkImage output_image = VK_NULL_HANDLE;
                VkImageView output_view = VK_NULL_HANDLE;
                VmaAllocation output_allocation = VK_NULL_HANDLE;
                glm::ivec2 render_extent{0, 0};
                glm::ivec2 output_extent{0, 0};
                SceneTemporalQuality quality = SceneTemporalQuality::Balanced;
                bool motion_includes_jitter = false;
                bool reset_pending = true;
                std::uint64_t generation = 0;
            };

            void destroyView(ViewState& state) noexcept {
                if (state.feature != nullptr)
                    NVSDK_NGX_VULKAN_ReleaseFeature(state.feature);
                if (state.output_view != VK_NULL_HANDLE)
                    vkDestroyImageView(device_, state.output_view, nullptr);
                if (state.output_image != VK_NULL_HANDLE)
                    vmaDestroyImage(allocator_, state.output_image, state.output_allocation);
                state = {};
            }

            [[nodiscard]] bool ensureFeature(
                const VkCommandBuffer command_buffer,
                const VulkanSceneUpscalerDispatch& dispatch,
                ViewState& state) noexcept {
                if (state.feature != nullptr && state.render_extent == dispatch.color.valid_extent &&
                    state.output_extent == dispatch.output_extent &&
                    state.quality == dispatch.quality &&
                    state.motion_includes_jitter == dispatch.motion_includes_jitter) {
                    return true;
                }
                if (state.feature != nullptr || state.output_image != VK_NULL_HANDLE) {
                    vkDeviceWaitIdle(device_);
                    destroyView(state);
                }

                updateRecommendedScales(parameters_, dispatch.output_extent);

                VkImageCreateInfo image_info{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
                image_info.imageType = VK_IMAGE_TYPE_2D;
                image_info.format = OUTPUT_FORMAT;
                image_info.extent = {static_cast<std::uint32_t>(dispatch.output_extent.x),
                                     static_cast<std::uint32_t>(dispatch.output_extent.y),
                                     1};
                image_info.mipLevels = 1;
                image_info.arrayLayers = 1;
                image_info.samples = VK_SAMPLE_COUNT_1_BIT;
                image_info.tiling = VK_IMAGE_TILING_OPTIMAL;
                image_info.usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT |
                                   VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
                                   VK_IMAGE_USAGE_TRANSFER_DST_BIT;
                image_info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
                VmaAllocationCreateInfo allocation_info{};
                allocation_info.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
                if (vmaCreateImage(allocator_,
                                   &image_info,
                                   &allocation_info,
                                   &state.output_image,
                                   &state.output_allocation,
                                   nullptr) != VK_SUCCESS) {
                    destroyView(state);
                    return false;
                }
                VkImageViewCreateInfo view_info{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
                view_info.image = state.output_image;
                view_info.viewType = VK_IMAGE_VIEW_TYPE_2D;
                view_info.format = OUTPUT_FORMAT;
                view_info.subresourceRange = {
                    .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                    .baseMipLevel = 0,
                    .levelCount = 1,
                    .baseArrayLayer = 0,
                    .layerCount = 1,
                };
                if (vkCreateImageView(device_, &view_info, nullptr, &state.output_view) !=
                    VK_SUCCESS) {
                    destroyView(state);
                    return false;
                }
                VkImageMemoryBarrier barrier{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
                barrier.srcAccessMask = 0;
                barrier.dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT;
                barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
                barrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
                barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                barrier.image = state.output_image;
                barrier.subresourceRange = view_info.subresourceRange;
                vkCmdPipelineBarrier(command_buffer,
                                     VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                                     VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                     0,
                                     0,
                                     nullptr,
                                     0,
                                     nullptr,
                                     1,
                                     &barrier);

                NVSDK_NGX_DLSS_Create_Params create{};
                create.Feature.InWidth =
                    static_cast<unsigned int>(dispatch.color.valid_extent.x);
                create.Feature.InHeight =
                    static_cast<unsigned int>(dispatch.color.valid_extent.y);
                create.Feature.InTargetWidth =
                    static_cast<unsigned int>(dispatch.output_extent.x);
                create.Feature.InTargetHeight =
                    static_cast<unsigned int>(dispatch.output_extent.y);
                create.Feature.InPerfQualityValue = ngxQuality(dispatch.quality);
                create.InFeatureCreateFlags = NVSDK_NGX_DLSS_Feature_Flags_MVLowRes |
                                              NVSDK_NGX_DLSS_Feature_Flags_AutoExposure;
                if (dispatch.motion_includes_jitter)
                    create.InFeatureCreateFlags |= NVSDK_NGX_DLSS_Feature_Flags_MVJittered;
                const auto result = NGX_VULKAN_CREATE_DLSS_EXT(command_buffer,
                                                               1,
                                                               1,
                                                               &state.feature,
                                                               parameters_,
                                                               &create);
                if (NVSDK_NGX_FAILED(result) || state.feature == nullptr) {
                    LOG_WARN("NGX DLSS feature creation failed: {:#x}",
                             static_cast<unsigned int>(result));
                    destroyView(state);
                    return false;
                }
                state.render_extent = dispatch.color.valid_extent;
                state.output_extent = dispatch.output_extent;
                state.quality = dispatch.quality;
                state.motion_includes_jitter = dispatch.motion_includes_jitter;
                state.reset_pending = true;
                ++state.generation;
                LOG_DEBUG("NGX DLSS feature ready: view={} render={}x{} output={}x{} quality={}",
                          viewIndex(dispatch.view),
                          state.render_extent.x,
                          state.render_extent.y,
                          state.output_extent.x,
                          state.output_extent.y,
                          static_cast<int>(state.quality));
                return true;
            }

            VulkanContext* context_ = nullptr;
            VkDevice device_ = VK_NULL_HANDLE;
            VmaAllocator allocator_ = VK_NULL_HANDLE;
            NVSDK_NGX_Parameter* parameters_ = nullptr;
            std::array<ViewState, static_cast<std::size_t>(TemporalViewId::Count)> views_{};
            bool initialized_ = false;
        };

        SceneUpscalerAdapterFactoryResult makeNvidiaDlssAdapter() noexcept {
            try {
                return std::make_unique<NvidiaDlssVulkanAdapter>();
            } catch (...) {
                return std::unexpected(SceneUpscalerAvailabilityReason::ProbeFailed);
            }
        }
    } // namespace

    std::vector<std::string> nvidiaDlssRequiredInstanceExtensions() noexcept {
        try {
            const auto data_path = discoveryDataPath();
            if (!data_path) {
                bootstrap_ready.store(false, std::memory_order_relaxed);
                return {};
            }
            auto info = discoveryInfo(data_path->c_str());
            uint32_t count = 0;
            VkExtensionProperties* properties = nullptr;
            LOG_DEBUG("Querying NVIDIA DLSS Vulkan instance requirements");
            const auto result = NVSDK_NGX_VULKAN_GetFeatureInstanceExtensionRequirements(
                &info, &count, &properties);
            if (NVSDK_NGX_FAILED(result) || (count > 0 && properties == nullptr)) {
                bootstrap_ready.store(false, std::memory_order_relaxed);
                LOG_WARN("Cannot query NGX Vulkan instance extensions: {:#x}",
                         static_cast<unsigned int>(result));
                return {};
            }
            LOG_DEBUG("NVIDIA DLSS requires {} Vulkan instance extension(s)", count);
            return extensionNames(count, properties);
        } catch (const std::exception& exception) {
            bootstrap_ready.store(false, std::memory_order_relaxed);
            LOG_WARN("Cannot retain NGX Vulkan instance extensions: {}", exception.what());
            return {};
        }
    }

    std::vector<std::string> nvidiaDlssRequiredDeviceExtensions(
        const VkInstance instance,
        const VkPhysicalDevice physical_device) noexcept {
        try {
            if (!bootstrap_ready.load(std::memory_order_relaxed))
                return {};
            const auto data_path = discoveryDataPath();
            if (!data_path) {
                bootstrap_ready.store(false, std::memory_order_relaxed);
                return {};
            }
            auto info = discoveryInfo(data_path->c_str());
            uint32_t count = 0;
            VkExtensionProperties* properties = nullptr;
            LOG_DEBUG("Querying NVIDIA DLSS Vulkan device requirements");
            const auto result = NVSDK_NGX_VULKAN_GetFeatureDeviceExtensionRequirements(
                instance, physical_device, &info, &count, &properties);
            if (NVSDK_NGX_FAILED(result) || (count > 0 && properties == nullptr)) {
                bootstrap_ready.store(false, std::memory_order_relaxed);
                LOG_WARN("Cannot query NGX Vulkan device extensions: {:#x}",
                         static_cast<unsigned int>(result));
                return {};
            }
            LOG_DEBUG("NVIDIA DLSS requires {} Vulkan device extension(s)", count);
            return extensionNames(count, properties);
        } catch (const std::exception& exception) {
            bootstrap_ready.store(false, std::memory_order_relaxed);
            LOG_WARN("Cannot retain NGX Vulkan device extensions: {}", exception.what());
            return {};
        }
    }

    void disableNvidiaDlssVulkanBootstrap() noexcept {
        bootstrap_ready.store(false, std::memory_order_relaxed);
    }

    bool nvidiaDlssVulkanBootstrapReady() noexcept {
        return bootstrap_ready.load(std::memory_order_relaxed);
    }

    std::array<float, 3> nvidiaDlssRecommendedInputScales() noexcept {
        std::lock_guard lock(recommended_scales_mutex);
        return recommended_scales;
    }

    void registerNvidiaDlssVulkanAdapter() {
        static const bool registered = optionalSceneUpscalerRegistry().registerAdapter(
            {.id = std::string(ADAPTER_ID),
             .label_key = "preferences.scene_upscaler_nvidia_dlss",
             .requirements = {
                 .depth = true,
                 .motion_vectors = true,
                 .jitter = true,
                 .history = true,
             }},
            &makeNvidiaDlssAdapter);
        if (!registered)
            LOG_WARN("NVIDIA DLSS adapter registration was rejected");
    }
} // namespace lfs::vis
