/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: MIT */

#include "rendering/scene_upscaler_plugin_api.h"

#include <nvsdk_ngx_helpers.h>
#include <nvsdk_ngx_helpers_vk.h>
#include <nvsdk_ngx_vk.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <format>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>

namespace {
    constexpr std::string_view NVIDIA_DLSS_SDK_URL = "https://github.com/NVIDIA/DLSS";

    constexpr std::uint32_t MIN_DLSS_OUTPUT_EXTENT = 32;

    constexpr std::string_view PLUGIN_ID = "nvidia-dlss";
    constexpr std::string_view DISPLAY_NAME = "NVIDIA DLSS";
    constexpr std::size_t VIEW_COUNT = LFS_SCENE_UPSCALER_PLUGIN_VIEW_COUNT;

    [[nodiscard]] NVSDK_NGX_PerfQuality_Value ngxQuality(
        const std::uint32_t quality) noexcept {
        switch (quality) {
        case LFS_SCENE_UPSCALER_PLUGIN_QUALITY:
            return NVSDK_NGX_PerfQuality_Value_MaxQuality;
        case LFS_SCENE_UPSCALER_PLUGIN_BALANCED:
            return NVSDK_NGX_PerfQuality_Value_Balanced;
        case LFS_SCENE_UPSCALER_PLUGIN_PERFORMANCE:
            return NVSDK_NGX_PerfQuality_Value_MaxPerf;
        }
        return NVSDK_NGX_PerfQuality_Value_Balanced;
    }

    [[nodiscard]] bool validView(const std::uint32_t view) noexcept {
        return static_cast<std::size_t>(view) < VIEW_COUNT;
    }

    [[nodiscard]] bool validQuality(const std::uint32_t quality) noexcept {
        return quality == LFS_SCENE_UPSCALER_PLUGIN_QUALITY ||
               quality == LFS_SCENE_UPSCALER_PLUGIN_BALANCED ||
               quality == LFS_SCENE_UPSCALER_PLUGIN_PERFORMANCE;
    }

    [[nodiscard]] std::string ngxFailure(const std::string_view operation,
                                         const NVSDK_NGX_Result result) {
        return std::format("{} failed with NGX result {:#x}",
                           operation,
                           static_cast<unsigned int>(result));
    }

    struct ViewState {
        NVSDK_NGX_Handle* feature = nullptr;
        LfsSceneUpscalerFeatureConfigV1 config{};
        bool configured = false;
    };

    class NvidiaDlssPlugin final {
    public:
        explicit NvidiaDlssPlugin(const LfsSceneUpscalerBootstrapConfigV1& config)
            : project_id_(config.project_id),
              engine_version_(config.engine_version),
              application_data_path_(config.application_data_path),
              plugin_directory_(config.plugin_directory) {}

        ~NvidiaDlssPlugin() { shutdownRuntimeLocked(); }

        LfsSceneUpscalerPluginResult requiredInstanceExtensions(
            const LfsSceneUpscalerExtensionSink& sink) {
            std::scoped_lock lock(mutex_);
            if (!validSink(sink))
                return fail(LFS_SCENE_UPSCALER_PLUGIN_INVALID_ARGUMENT,
                            "invalid Vulkan instance-extension sink");
            auto discovery = discoveryInfo();
            std::uint32_t count = 0;
            VkExtensionProperties* properties = nullptr;
            const auto result = NVSDK_NGX_VULKAN_GetFeatureInstanceExtensionRequirements(
                &discovery, &count, &properties);
            if (NVSDK_NGX_FAILED(result) || (count != 0 && properties == nullptr))
                return fail(LFS_SCENE_UPSCALER_PLUGIN_RUNTIME_ERROR,
                            ngxFailure("querying Vulkan instance extensions", result));
            return appendExtensions(sink, count, properties);
        }

        LfsSceneUpscalerPluginResult requiredDeviceExtensions(
            const VkInstance instance,
            const VkPhysicalDevice physical_device,
            const LfsSceneUpscalerExtensionSink& sink) {
            std::scoped_lock lock(mutex_);
            if (instance == VK_NULL_HANDLE || physical_device == VK_NULL_HANDLE ||
                !validSink(sink)) {
                return fail(LFS_SCENE_UPSCALER_PLUGIN_INVALID_ARGUMENT,
                            "invalid Vulkan device-extension query");
            }
            auto discovery = discoveryInfo();
            std::uint32_t count = 0;
            VkExtensionProperties* properties = nullptr;
            const auto result = NVSDK_NGX_VULKAN_GetFeatureDeviceExtensionRequirements(
                instance, physical_device, &discovery, &count, &properties);
            if (NVSDK_NGX_FAILED(result) || (count != 0 && properties == nullptr))
                return fail(LFS_SCENE_UPSCALER_PLUGIN_RUNTIME_ERROR,
                            ngxFailure("querying Vulkan device extensions", result));
            return appendExtensions(sink, count, properties);
        }

        LfsSceneUpscalerPluginResult initializeRuntime(
            const LfsSceneUpscalerRuntimeConfigV1& config) {
            std::scoped_lock lock(mutex_);
            if (runtime_initialized_)
                return LFS_SCENE_UPSCALER_PLUGIN_OK;
            if (config.struct_size < sizeof(LfsSceneUpscalerRuntimeConfigV1) ||
                config.instance == VK_NULL_HANDLE ||
                config.physical_device == VK_NULL_HANDLE || config.device == VK_NULL_HANDLE ||
                config.get_instance_proc_addr == nullptr || config.get_device_proc_addr == nullptr) {
                return fail(LFS_SCENE_UPSCALER_PLUGIN_INVALID_ARGUMENT,
                            "invalid NGX Vulkan runtime configuration");
            }

            auto discovery = discoveryInfo();
            NVSDK_NGX_FeatureRequirement requirement{};
            const auto requirement_result = NVSDK_NGX_VULKAN_GetFeatureRequirements(
                config.instance, config.physical_device, &discovery, &requirement);
            if (NVSDK_NGX_FAILED(requirement_result))
                return fail(LFS_SCENE_UPSCALER_PLUGIN_RUNTIME_ERROR,
                            std::format(
                                "{}; verify that the NVIDIA NGX/DLSS runtime is staged "
                                "beside the LichtFeld DLSS plugin (official SDK: {})",
                                ngxFailure("querying DLSS feature support", requirement_result),
                                NVIDIA_DLSS_SDK_URL));
            if (requirement.FeatureSupported != 0)
                return fail(LFS_SCENE_UPSCALER_PLUGIN_UNSUPPORTED_DEVICE,
                            std::format(
                                "NGX reports DLSS support flags {:#x} "
                                "(minimum GPU architecture {}, minimum OS '{}')",
                                static_cast<unsigned int>(requirement.FeatureSupported),
                                requirement.MinHWArchitecture,
                                requirement.MinOSVersion));

            const auto init_result = NVSDK_NGX_VULKAN_Init_with_ProjectID(
                project_id_.c_str(),
                NVSDK_NGX_ENGINE_TYPE_CUSTOM,
                engine_version_.c_str(),
                application_data_path_.c_str(),
                config.instance,
                config.physical_device,
                config.device,
                config.get_instance_proc_addr,
                config.get_device_proc_addr,
                &feature_common_info_);
            if (NVSDK_NGX_FAILED(init_result))
                return fail(LFS_SCENE_UPSCALER_PLUGIN_UNAVAILABLE,
                            std::format(
                                "{}; verify that the NVIDIA NGX/DLSS runtime is staged "
                                "beside the LichtFeld DLSS plugin (official SDK: {})",
                                ngxFailure("initializing NGX Vulkan", init_result),
                                NVIDIA_DLSS_SDK_URL));

            device_ = config.device;
            const auto capability_result =
                NVSDK_NGX_VULKAN_GetCapabilityParameters(&parameters_);
            if (NVSDK_NGX_FAILED(capability_result) || parameters_ == nullptr) {
                const auto detail = ngxFailure("obtaining NGX capability parameters",
                                               capability_result);
                shutdownRuntimeLocked();
                return fail(LFS_SCENE_UPSCALER_PLUGIN_RUNTIME_ERROR, detail);
            }
            int available = 0;
            const auto availability_result = NVSDK_NGX_Parameter_GetI(
                parameters_, NVSDK_NGX_Parameter_SuperSampling_Available, &available);
            if (NVSDK_NGX_FAILED(availability_result)) {
                const auto detail = ngxFailure("reading the NGX DLSS availability flag",
                                               availability_result);
                shutdownRuntimeLocked();
                return fail(LFS_SCENE_UPSCALER_PLUGIN_RUNTIME_ERROR, detail);
            }
            if (available == 0) {
                std::string detail;
                int needs_updated_driver = 0;
                int minimum_driver_major = 0;
                int minimum_driver_minor = 0;
                const bool driver_update_required =
                    NVSDK_NGX_SUCCEED(NVSDK_NGX_Parameter_GetI(
                        parameters_,
                        NVSDK_NGX_Parameter_SuperSampling_NeedsUpdatedDriver,
                        &needs_updated_driver)) &&
                    needs_updated_driver != 0;
                const bool minimum_driver_known =
                    driver_update_required &&
                    NVSDK_NGX_SUCCEED(NVSDK_NGX_Parameter_GetI(
                        parameters_,
                        NVSDK_NGX_Parameter_SuperSampling_MinDriverVersionMajor,
                        &minimum_driver_major)) &&
                    NVSDK_NGX_SUCCEED(NVSDK_NGX_Parameter_GetI(
                        parameters_,
                        NVSDK_NGX_Parameter_SuperSampling_MinDriverVersionMinor,
                        &minimum_driver_minor));
                if (minimum_driver_known) {
                    detail = std::format(
                        "NGX requires NVIDIA driver {}.{} or newer for DLSS",
                        minimum_driver_major,
                        minimum_driver_minor);
                } else if (driver_update_required) {
                    detail = "NGX reports that the NVIDIA driver must be updated for DLSS";
                } else {
                    detail = "NGX reports DLSS as unavailable on this GPU/driver/OS";
                }
                shutdownRuntimeLocked();
                return fail(LFS_SCENE_UPSCALER_PLUGIN_UNSUPPORTED_DEVICE, detail);
            }

            runtime_initialized_ = true;
            last_error_.clear();
            return LFS_SCENE_UPSCALER_PLUGIN_OK;
        }

        LfsSceneUpscalerPluginResult optimalSettings(
            const std::uint32_t output_width,
            const std::uint32_t output_height,
            const std::uint32_t quality,
            LfsSceneUpscalerOptimalSettingsV1& settings) {
            std::scoped_lock lock(mutex_);
            if (!runtime_initialized_ || parameters_ == nullptr)
                return fail(LFS_SCENE_UPSCALER_PLUGIN_UNAVAILABLE,
                            "NGX runtime is not initialized");
            if (settings.struct_size < sizeof(LfsSceneUpscalerOptimalSettingsV1) ||
                output_width < MIN_DLSS_OUTPUT_EXTENT ||
                output_height < MIN_DLSS_OUTPUT_EXTENT || !validQuality(quality))
                return fail(LFS_SCENE_UPSCALER_PLUGIN_INVALID_ARGUMENT,
                            "invalid DLSS output extent");

            unsigned int render_width = 0;
            unsigned int render_height = 0;
            unsigned int maximum_width = 0;
            unsigned int maximum_height = 0;
            unsigned int minimum_width = 0;
            unsigned int minimum_height = 0;
            float sharpness = 0.0f;
            const auto result = NGX_DLSS_GET_OPTIMAL_SETTINGS(parameters_,
                                                              output_width,
                                                              output_height,
                                                              ngxQuality(quality),
                                                              &render_width,
                                                              &render_height,
                                                              &maximum_width,
                                                              &maximum_height,
                                                              &minimum_width,
                                                              &minimum_height,
                                                              &sharpness);
            if (NVSDK_NGX_FAILED(result) || render_width == 0 || render_height == 0)
                return fail(LFS_SCENE_UPSCALER_PLUGIN_RUNTIME_ERROR,
                            ngxFailure("querying DLSS optimal settings", result));
            settings.render_width = render_width;
            settings.render_height = render_height;
            settings.minimum_width = minimum_width;
            settings.minimum_height = minimum_height;
            settings.maximum_width = maximum_width;
            settings.maximum_height = maximum_height;
            settings.sharpness = sharpness;
            last_error_.clear();
            return LFS_SCENE_UPSCALER_PLUGIN_OK;
        }

        LfsSceneUpscalerPluginResult createFeature(
            const VkCommandBuffer command_buffer,
            const LfsSceneUpscalerFeatureConfigV1& config) {
            std::scoped_lock lock(mutex_);
            if (!runtime_initialized_ || parameters_ == nullptr)
                return fail(LFS_SCENE_UPSCALER_PLUGIN_UNAVAILABLE,
                            "NGX runtime is not initialized");
            if (command_buffer == VK_NULL_HANDLE ||
                config.struct_size < sizeof(LfsSceneUpscalerFeatureConfigV1) ||
                !validView(config.view) || !validQuality(config.quality) ||
                config.render_width == 0 ||
                config.render_height == 0 ||
                config.output_width < MIN_DLSS_OUTPUT_EXTENT ||
                config.output_height < MIN_DLSS_OUTPUT_EXTENT) {
                return fail(LFS_SCENE_UPSCALER_PLUGIN_INVALID_ARGUMENT,
                            "invalid DLSS feature configuration");
            }
            auto& view = views_[static_cast<std::size_t>(config.view)];
            if (view.feature != nullptr && sameFeatureConfig(view.config, config)) {
                last_error_.clear();
                return LFS_SCENE_UPSCALER_PLUGIN_OK;
            }
            releaseFeatureLocked(view);

            NVSDK_NGX_DLSS_Create_Params create{};
            create.Feature.InWidth = config.render_width;
            create.Feature.InHeight = config.render_height;
            create.Feature.InTargetWidth = config.output_width;
            create.Feature.InTargetHeight = config.output_height;
            create.Feature.InPerfQualityValue = ngxQuality(config.quality);
            create.InFeatureCreateFlags = NVSDK_NGX_DLSS_Feature_Flags_MVLowRes |
                                          NVSDK_NGX_DLSS_Feature_Flags_AutoExposure;
            if (config.motion_vectors_include_jitter != 0)
                create.InFeatureCreateFlags |= NVSDK_NGX_DLSS_Feature_Flags_MVJittered;

            const auto result = NGX_VULKAN_CREATE_DLSS_EXT(command_buffer,
                                                           1,
                                                           1,
                                                           &view.feature,
                                                           parameters_,
                                                           &create);
            if (NVSDK_NGX_FAILED(result) || view.feature == nullptr) {
                releaseFeatureLocked(view);
                return fail(LFS_SCENE_UPSCALER_PLUGIN_RUNTIME_ERROR,
                            ngxFailure("creating DLSS feature", result));
            }
            view.config = config;
            view.config.struct_size = sizeof(LfsSceneUpscalerFeatureConfigV1);
            view.configured = true;
            last_error_.clear();
            return LFS_SCENE_UPSCALER_PLUGIN_OK;
        }

        LfsSceneUpscalerPluginResult evaluate(
            const LfsSceneUpscalerEvaluateV1& evaluation) {
            std::scoped_lock lock(mutex_);
            if (!runtime_initialized_ || parameters_ == nullptr)
                return fail(LFS_SCENE_UPSCALER_PLUGIN_UNAVAILABLE,
                            "NGX runtime is not initialized");
            if (evaluation.struct_size < sizeof(LfsSceneUpscalerEvaluateV1) ||
                !validView(evaluation.view) || evaluation.command_buffer == VK_NULL_HANDLE ||
                !validImage(evaluation.color, false) ||
                !validImage(evaluation.depth, false) ||
                !validImage(evaluation.motion, false) ||
                !validImage(evaluation.output, true)) {
                return fail(LFS_SCENE_UPSCALER_PLUGIN_INVALID_ARGUMENT,
                            "invalid DLSS evaluation resources");
            }
            auto& view = views_[static_cast<std::size_t>(evaluation.view)];
            if (view.feature == nullptr || !view.configured)
                return fail(LFS_SCENE_UPSCALER_PLUGIN_UNAVAILABLE,
                            "DLSS feature is not configured for this view");
            if (evaluation.color.valid_width != view.config.render_width ||
                evaluation.color.valid_height != view.config.render_height ||
                evaluation.depth.valid_width != view.config.render_width ||
                evaluation.depth.valid_height != view.config.render_height ||
                evaluation.motion.valid_width != view.config.render_width ||
                evaluation.motion.valid_height != view.config.render_height ||
                evaluation.output.valid_width != view.config.output_width ||
                evaluation.output.valid_height != view.config.output_height ||
                evaluation.color.format != VK_FORMAT_R8G8B8A8_UNORM ||
                evaluation.depth.format != VK_FORMAT_R32_SFLOAT ||
                evaluation.motion.format != VK_FORMAT_R16G16_SFLOAT ||
                evaluation.output.format != VK_FORMAT_R8G8B8A8_UNORM ||
                !std::isfinite(evaluation.jitter_x_pixels) ||
                !std::isfinite(evaluation.jitter_y_pixels) ||
                !std::isfinite(evaluation.motion_scale_x) ||
                !std::isfinite(evaluation.motion_scale_y) ||
                !std::isfinite(evaluation.frame_time_milliseconds)) {
                return fail(LFS_SCENE_UPSCALER_PLUGIN_INVALID_ARGUMENT,
                            "DLSS evaluation resources do not match the active feature");
            }

            constexpr VkImageSubresourceRange COLOR_RANGE{
                .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                .baseMipLevel = 0,
                .levelCount = 1,
                .baseArrayLayer = 0,
                .layerCount = 1,
            };
            auto color = makeResource(evaluation.color, COLOR_RANGE);
            auto depth = makeResource(evaluation.depth, COLOR_RANGE);
            auto motion = makeResource(evaluation.motion, COLOR_RANGE);
            auto output = makeResource(evaluation.output, COLOR_RANGE);

            NVSDK_NGX_VK_DLSS_Eval_Params params{};
            params.Feature.pInColor = &color;
            params.Feature.pInOutput = &output;
            params.pInDepth = &depth;
            params.pInMotionVectors = &motion;
            params.InJitterOffsetX = evaluation.jitter_x_pixels;
            params.InJitterOffsetY = evaluation.jitter_y_pixels;
            params.InRenderSubrectDimensions = {
                evaluation.color.valid_width,
                evaluation.color.valid_height,
            };
            params.InReset = evaluation.reset_flags != 0 ? 1 : 0;
            params.InMVScaleX = evaluation.motion_scale_x;
            params.InMVScaleY = evaluation.motion_scale_y;
            params.InPreExposure = evaluation.pre_exposure > 0.0f
                                       ? evaluation.pre_exposure
                                       : 1.0f;
            params.InExposureScale = 1.0f;
            params.InFrameTimeDeltaInMsec = evaluation.frame_time_milliseconds;
            const auto result = NGX_VULKAN_EVALUATE_DLSS_EXT(
                evaluation.command_buffer, view.feature, parameters_, &params);
            if (NVSDK_NGX_FAILED(result))
                return fail(LFS_SCENE_UPSCALER_PLUGIN_RUNTIME_ERROR,
                            ngxFailure("evaluating DLSS", result));
            last_error_.clear();
            return LFS_SCENE_UPSCALER_PLUGIN_OK;
        }

        void releaseFeature(const std::uint32_t view) {
            std::scoped_lock lock(mutex_);
            if (validView(view))
                releaseFeatureLocked(views_[static_cast<std::size_t>(view)]);
        }

        void shutdownRuntime() {
            std::scoped_lock lock(mutex_);
            shutdownRuntimeLocked();
        }

        std::size_t lastError(char* const destination, const std::size_t capacity) const {
            std::scoped_lock lock(mutex_);
            const std::size_t required = last_error_.size() + 1;
            if (destination != nullptr && capacity != 0) {
                const std::size_t copy_size = std::min(last_error_.size(), capacity - 1);
                std::memcpy(destination, last_error_.data(), copy_size);
                destination[copy_size] = '\0';
            }
            return required;
        }

    private:
        [[nodiscard]] static bool validSink(
            const LfsSceneUpscalerExtensionSink& sink) noexcept {
            return sink.struct_size >= sizeof(LfsSceneUpscalerExtensionSink) &&
                   sink.append != nullptr;
        }

        [[nodiscard]] static bool validImage(const LfsSceneUpscalerImageV1& image,
                                             const bool writable) noexcept {
            const bool valid_layout = writable
                                          ? image.layout == VK_IMAGE_LAYOUT_GENERAL
                                          : image.layout ==
                                                VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            return image.struct_size >= sizeof(LfsSceneUpscalerImageV1) &&
                   image.image != VK_NULL_HANDLE && image.view != VK_NULL_HANDLE &&
                   valid_layout &&
                   image.format != VK_FORMAT_UNDEFINED && image.allocation_width != 0 &&
                   image.allocation_height != 0 && image.valid_width != 0 &&
                   image.valid_height != 0 &&
                   image.valid_width <= image.allocation_width &&
                   image.valid_height <= image.allocation_height &&
                   (!writable || image.writable != 0);
        }

        [[nodiscard]] static bool sameFeatureConfig(
            const LfsSceneUpscalerFeatureConfigV1& left,
            const LfsSceneUpscalerFeatureConfigV1& right) noexcept {
            return left.view == right.view && left.quality == right.quality &&
                   left.render_width == right.render_width &&
                   left.render_height == right.render_height &&
                   left.output_width == right.output_width &&
                   left.output_height == right.output_height &&
                   left.motion_vectors_include_jitter == right.motion_vectors_include_jitter;
        }

        [[nodiscard]] static NVSDK_NGX_Resource_VK makeResource(
            const LfsSceneUpscalerImageV1& image,
            const VkImageSubresourceRange& range) {
            return NVSDK_NGX_Create_ImageView_Resource_VK(image.view,
                                                          image.image,
                                                          range,
                                                          image.format,
                                                          image.allocation_width,
                                                          image.allocation_height,
                                                          image.writable != 0);
        }

        [[nodiscard]] NVSDK_NGX_FeatureDiscoveryInfo discoveryInfo() {
            feature_path_ = plugin_directory_.c_str();
            feature_path_list_.Path = &feature_path_;
            feature_path_list_.Length = 1;
            feature_common_info_.PathListInfo = feature_path_list_;

            NVSDK_NGX_FeatureDiscoveryInfo info{};
            info.SDKVersion = NVSDK_NGX_Version_API;
            info.FeatureID = NVSDK_NGX_Feature_SuperSampling;
            info.Identifier.IdentifierType =
                NVSDK_NGX_Application_Identifier_Type_Project_Id;
            info.Identifier.v.ProjectDesc.ProjectId = project_id_.c_str();
            info.Identifier.v.ProjectDesc.EngineType = NVSDK_NGX_ENGINE_TYPE_CUSTOM;
            info.Identifier.v.ProjectDesc.EngineVersion = engine_version_.c_str();
            info.ApplicationDataPath = application_data_path_.c_str();
            info.FeatureInfo = &feature_common_info_;
            return info;
        }

        LfsSceneUpscalerPluginResult appendExtensions(
            const LfsSceneUpscalerExtensionSink& sink,
            const std::uint32_t count,
            const VkExtensionProperties* const properties) {
            for (std::uint32_t index = 0; index < count; ++index) {
                if (sink.append(sink.user, properties[index].extensionName) == 0)
                    return fail(LFS_SCENE_UPSCALER_PLUGIN_RUNTIME_ERROR,
                                "host rejected a required Vulkan extension");
            }
            last_error_.clear();
            return LFS_SCENE_UPSCALER_PLUGIN_OK;
        }

        LfsSceneUpscalerPluginResult fail(const LfsSceneUpscalerPluginResult result,
                                          std::string error) {
            last_error_ = std::move(error);
            return result;
        }

        static void releaseFeatureLocked(ViewState& view) {
            if (view.feature != nullptr)
                NVSDK_NGX_VULKAN_ReleaseFeature(view.feature);
            view = {};
        }

        void shutdownRuntimeLocked() {
            for (auto& view : views_)
                releaseFeatureLocked(view);
            if (parameters_ != nullptr)
                NVSDK_NGX_VULKAN_DestroyParameters(parameters_);
            parameters_ = nullptr;
            if (device_ != VK_NULL_HANDLE)
                NVSDK_NGX_VULKAN_Shutdown1(device_);
            device_ = VK_NULL_HANDLE;
            runtime_initialized_ = false;
        }

        mutable std::mutex mutex_;
        std::string project_id_;
        std::string engine_version_;
        std::wstring application_data_path_;
        std::wstring plugin_directory_;
        const wchar_t* feature_path_ = nullptr;
        NVSDK_NGX_PathListInfo feature_path_list_{};
        NVSDK_NGX_FeatureCommonInfo feature_common_info_{};
        VkDevice device_ = VK_NULL_HANDLE;
        NVSDK_NGX_Parameter* parameters_ = nullptr;
        std::array<ViewState, VIEW_COUNT> views_{};
        std::string last_error_;
        bool runtime_initialized_ = false;
    };

    [[nodiscard]] NvidiaDlssPlugin* pluginFrom(void* const plugin) noexcept {
        return static_cast<NvidiaDlssPlugin*>(plugin);
    }

    void* createPlugin(const LfsSceneUpscalerBootstrapConfigV1* const config) noexcept {
        try {
            if (config == nullptr ||
                config->struct_size < sizeof(LfsSceneUpscalerBootstrapConfigV1) ||
                config->project_id == nullptr || config->engine_version == nullptr ||
                config->application_data_path == nullptr ||
                config->plugin_directory == nullptr) {
                return nullptr;
            }
            return new NvidiaDlssPlugin(*config);
        } catch (...) {
            // LFS-CENSUS-OK(empty-catch): construction is a C ABI boundary with no
            // plugin instance available for diagnostics; nullptr reports failure.
            return nullptr;
        }
    }

    void destroyPlugin(void* const plugin) noexcept {
        delete pluginFrom(plugin);
    }

    LfsSceneUpscalerPluginResult requiredInstanceExtensions(
        void* const plugin,
        const LfsSceneUpscalerExtensionSink* const sink) noexcept {
        try {
            return plugin != nullptr && sink != nullptr
                       ? pluginFrom(plugin)->requiredInstanceExtensions(*sink)
                       : LFS_SCENE_UPSCALER_PLUGIN_INVALID_ARGUMENT;
        } catch (...) {
            // LFS-CENSUS-OK(empty-catch): never unwind a vendor or allocation
            // exception through the C ABI; the typed result reports the failure.
            return LFS_SCENE_UPSCALER_PLUGIN_RUNTIME_ERROR;
        }
    }

    LfsSceneUpscalerPluginResult requiredDeviceExtensions(
        void* const plugin,
        const VkInstance instance,
        const VkPhysicalDevice physical_device,
        const LfsSceneUpscalerExtensionSink* const sink) noexcept {
        try {
            return plugin != nullptr && sink != nullptr
                       ? pluginFrom(plugin)->requiredDeviceExtensions(
                             instance, physical_device, *sink)
                       : LFS_SCENE_UPSCALER_PLUGIN_INVALID_ARGUMENT;
        } catch (...) {
            // LFS-CENSUS-OK(empty-catch): never unwind a vendor or allocation
            // exception through the C ABI; the typed result reports the failure.
            return LFS_SCENE_UPSCALER_PLUGIN_RUNTIME_ERROR;
        }
    }

    LfsSceneUpscalerPluginResult initializeRuntime(
        void* const plugin,
        const LfsSceneUpscalerRuntimeConfigV1* const config) noexcept {
        try {
            return plugin != nullptr && config != nullptr
                       ? pluginFrom(plugin)->initializeRuntime(*config)
                       : LFS_SCENE_UPSCALER_PLUGIN_INVALID_ARGUMENT;
        } catch (...) {
            // LFS-CENSUS-OK(empty-catch): never unwind a vendor or allocation
            // exception through the C ABI; the typed result reports the failure.
            return LFS_SCENE_UPSCALER_PLUGIN_RUNTIME_ERROR;
        }
    }

    LfsSceneUpscalerPluginResult optimalSettings(
        void* const plugin,
        const std::uint32_t output_width,
        const std::uint32_t output_height,
        const std::uint32_t quality,
        LfsSceneUpscalerOptimalSettingsV1* const settings) noexcept {
        try {
            return plugin != nullptr && settings != nullptr
                       ? pluginFrom(plugin)->optimalSettings(
                             output_width, output_height, quality, *settings)
                       : LFS_SCENE_UPSCALER_PLUGIN_INVALID_ARGUMENT;
        } catch (...) {
            // LFS-CENSUS-OK(empty-catch): never unwind a vendor or allocation
            // exception through the C ABI; the typed result reports the failure.
            return LFS_SCENE_UPSCALER_PLUGIN_RUNTIME_ERROR;
        }
    }

    LfsSceneUpscalerPluginResult createFeature(
        void* const plugin,
        const VkCommandBuffer command_buffer,
        const LfsSceneUpscalerFeatureConfigV1* const config) noexcept {
        try {
            return plugin != nullptr && config != nullptr
                       ? pluginFrom(plugin)->createFeature(command_buffer, *config)
                       : LFS_SCENE_UPSCALER_PLUGIN_INVALID_ARGUMENT;
        } catch (...) {
            // LFS-CENSUS-OK(empty-catch): never unwind a vendor or allocation
            // exception through the C ABI; the typed result reports the failure.
            return LFS_SCENE_UPSCALER_PLUGIN_RUNTIME_ERROR;
        }
    }

    LfsSceneUpscalerPluginResult evaluate(
        void* const plugin,
        const LfsSceneUpscalerEvaluateV1* const evaluation) noexcept {
        try {
            return plugin != nullptr && evaluation != nullptr
                       ? pluginFrom(plugin)->evaluate(*evaluation)
                       : LFS_SCENE_UPSCALER_PLUGIN_INVALID_ARGUMENT;
        } catch (...) {
            // LFS-CENSUS-OK(empty-catch): never unwind a vendor or allocation
            // exception through the C ABI; the typed result reports the failure.
            return LFS_SCENE_UPSCALER_PLUGIN_RUNTIME_ERROR;
        }
    }

    void releaseFeature(void* const plugin,
                        const std::uint32_t view) noexcept {
        if (plugin != nullptr) {
            try {
                pluginFrom(plugin)->releaseFeature(view);
            } catch (...) {
                // LFS-CENSUS-OK(empty-catch): release is a noexcept C ABI cleanup
                // boundary; a vendor exception cannot be propagated to the host.
            }
        }
    }

    void shutdownRuntime(void* const plugin) noexcept {
        if (plugin != nullptr) {
            try {
                pluginFrom(plugin)->shutdownRuntime();
            } catch (...) {
                // LFS-CENSUS-OK(empty-catch): shutdown is a noexcept C ABI cleanup
                // boundary; a vendor exception cannot be propagated to the host.
            }
        }
    }

    std::size_t lastError(void* const plugin,
                          char* const destination,
                          const std::size_t capacity) noexcept {
        try {
            return plugin != nullptr
                       ? pluginFrom(plugin)->lastError(destination, capacity)
                       : 0;
        } catch (...) {
            // LFS-CENSUS-OK(empty-catch): diagnostics are best-effort at the C ABI
            // boundary and must not create a second exception during recovery.
            return 0;
        }
    }

    const LfsSceneUpscalerPluginApiV1 API{
        .struct_size = sizeof(LfsSceneUpscalerPluginApiV1),
        .abi_version = LFS_SCENE_UPSCALER_PLUGIN_ABI_V1,
        .plugin_id = PLUGIN_ID.data(),
        .display_name = DISPLAY_NAME.data(),
        .create = &createPlugin,
        .destroy = &destroyPlugin,
        .required_instance_extensions = &requiredInstanceExtensions,
        .required_device_extensions = &requiredDeviceExtensions,
        .initialize_runtime = &initializeRuntime,
        .optimal_settings = &optimalSettings,
        .create_feature = &createFeature,
        .evaluate = &evaluate,
        .release_feature = &releaseFeature,
        .shutdown_runtime = &shutdownRuntime,
        .last_error = &lastError,
    };
} // namespace

extern "C" LFS_SCENE_UPSCALER_PLUGIN_EXPORT const LfsSceneUpscalerPluginApiV1*
lfs_scene_upscaler_plugin_get_api_v1(void) {
    return &API;
}
