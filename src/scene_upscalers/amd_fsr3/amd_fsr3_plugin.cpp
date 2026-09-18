/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: MIT */

#include "rendering/scene_upscaler_plugin_api.h"

#include <bit>

#include <FidelityFX/host/backends/vk/ffx_vk.h>
#include <FidelityFX/host/ffx_fsr3upscaler.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <format>
#include <mutex>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {
    static_assert(FFX_SDK_VERSION_MAJOR == 1 && FFX_SDK_VERSION_MINOR == 1 &&
                      FFX_SDK_VERSION_PATCH == 4,
                  "The AMD FSR plugin targets FidelityFX SDK v1.1.4");
    static_assert(FFX_FSR3UPSCALER_VERSION_MAJOR == 3 &&
                      FFX_FSR3UPSCALER_VERSION_MINOR == 1 &&
                      FFX_FSR3UPSCALER_VERSION_PATCH == 4,
                  "The AMD FSR plugin targets FSR 3.1.4");

    constexpr std::string_view PLUGIN_ID = "amd-fsr3";
    constexpr std::string_view DISPLAY_NAME = "AMD FSR 3.1";
    constexpr std::size_t VIEW_COUNT = LFS_SCENE_UPSCALER_PLUGIN_VIEW_COUNT;
    constexpr std::size_t SHARED_RESOURCE_COUNT = 3;
    constexpr std::uint32_t MIN_OUTPUT_EXTENT = 32;

    [[nodiscard]] bool validView(const std::uint32_t view) noexcept {
        return static_cast<std::size_t>(view) < VIEW_COUNT;
    }

    [[nodiscard]] bool validQuality(const std::uint32_t quality) noexcept {
        return quality == LFS_SCENE_UPSCALER_PLUGIN_QUALITY ||
               quality == LFS_SCENE_UPSCALER_PLUGIN_BALANCED ||
               quality == LFS_SCENE_UPSCALER_PLUGIN_PERFORMANCE;
    }

    [[nodiscard]] FfxFsr3UpscalerQualityMode fsrQuality(
        const std::uint32_t quality) noexcept {
        switch (quality) {
        case LFS_SCENE_UPSCALER_PLUGIN_QUALITY:
            return FFX_FSR3UPSCALER_QUALITY_MODE_QUALITY;
        case LFS_SCENE_UPSCALER_PLUGIN_BALANCED:
            return FFX_FSR3UPSCALER_QUALITY_MODE_BALANCED;
        case LFS_SCENE_UPSCALER_PLUGIN_PERFORMANCE:
            return FFX_FSR3UPSCALER_QUALITY_MODE_PERFORMANCE;
        }
        return FFX_FSR3UPSCALER_QUALITY_MODE_BALANCED;
    }

    [[nodiscard]] FfxDimensions2D dimensions(const std::uint32_t width,
                                             const std::uint32_t height) noexcept {
        return {.width = width, .height = height};
    }

    [[nodiscard]] FfxResourceStates resourceStateForLayout(
        const VkImageLayout layout) noexcept {
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

    [[nodiscard]] FfxResourceDescription imageDescription(
        const LfsSceneUpscalerImageV1& image,
        const FfxResourceUsage usage) {
        VkImageCreateInfo info{};
        info.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        info.imageType = VK_IMAGE_TYPE_2D;
        info.format = image.format;
        info.extent = {image.allocation_width, image.allocation_height, 1};
        info.mipLevels = 1;
        info.arrayLayers = 1;
        info.samples = VK_SAMPLE_COUNT_1_BIT;
        info.tiling = VK_IMAGE_TILING_OPTIMAL;
        info.usage = VK_IMAGE_USAGE_SAMPLED_BIT;
        if ((usage & FFX_RESOURCE_USAGE_UAV) != 0)
            info.usage |= VK_IMAGE_USAGE_STORAGE_BIT;
        return ffxGetImageResourceDescriptionVK(image.image, info, usage);
    }

    [[nodiscard]] FfxResource wrapImage(
        const LfsSceneUpscalerImageV1& image,
        const wchar_t* const name,
        const FfxResourceUsage usage = FFX_RESOURCE_USAGE_READ_ONLY) {
        return ffxGetResourceVK(image.image,
                                imageDescription(image, usage),
                                name,
                                resourceStateForLayout(image.layout));
    }

    [[nodiscard]] std::string fsrFailure(const std::string_view operation,
                                         const FfxErrorCode result) {
        return std::format("{} failed with FidelityFX result {}",
                           operation,
                           static_cast<int>(result));
    }

    struct ViewState {
        FfxFsr3UpscalerContext context{};
        std::array<FfxResourceInternal, SHARED_RESOURCE_COUNT> shared{};
        std::size_t shared_count = 0;
        FfxUInt32 shared_effect_context_id = 0;
        LfsSceneUpscalerFeatureConfigV1 config{};
        bool context_created = false;
        bool shared_context_created = false;
        bool configured = false;
    };

    class AmdFsr3Plugin final {
    public:
        explicit AmdFsr3Plugin(const LfsSceneUpscalerBootstrapConfigV1&) {}
        ~AmdFsr3Plugin() { shutdownRuntimeLocked(); }

        LfsSceneUpscalerPluginResult requiredInstanceExtensions(
            const LfsSceneUpscalerExtensionSink& sink) {
            std::scoped_lock lock(mutex_);
            if (!validSink(sink))
                return fail(LFS_SCENE_UPSCALER_PLUGIN_INVALID_ARGUMENT,
                            "invalid Vulkan instance-extension sink");
            last_error_.clear();
            return LFS_SCENE_UPSCALER_PLUGIN_OK;
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
            last_error_.clear();
            return LFS_SCENE_UPSCALER_PLUGIN_OK;
        }

        LfsSceneUpscalerPluginResult initializeRuntime(
            const LfsSceneUpscalerRuntimeConfigV1& config) {
            std::scoped_lock lock(mutex_);
            if (runtime_initialized_)
                return LFS_SCENE_UPSCALER_PLUGIN_OK;
            if (config.struct_size < sizeof(LfsSceneUpscalerRuntimeConfigV1) ||
                config.instance == VK_NULL_HANDLE ||
                config.physical_device == VK_NULL_HANDLE ||
                config.device == VK_NULL_HANDLE ||
                config.get_instance_proc_addr == nullptr ||
                config.get_device_proc_addr == nullptr) {
                return fail(LFS_SCENE_UPSCALER_PLUGIN_INVALID_ARGUMENT,
                            "invalid FidelityFX Vulkan runtime configuration");
            }

            constexpr std::size_t CONTEXTS_PER_VIEW =
                FFX_FSR3UPSCALER_CONTEXT_COUNT + 1;
            constexpr std::size_t MAX_CONTEXTS = VIEW_COUNT * CONTEXTS_PER_VIEW;
            const std::size_t scratch_size =
                ffxGetScratchMemorySizeVK(config.physical_device, MAX_CONTEXTS);
            if (scratch_size == 0)
                return fail(LFS_SCENE_UPSCALER_PLUGIN_UNSUPPORTED_DEVICE,
                            "FidelityFX Vulkan backend reported no scratch-memory requirement");

            scratch_.resize(scratch_size);
            VkDeviceContext device_context{
                config.device,
                config.physical_device,
                config.get_device_proc_addr,
            };
            const auto result = ffxGetInterfaceVK(&backend_,
                                                  ffxGetDeviceVK(&device_context),
                                                  scratch_.data(),
                                                  scratch_.size(),
                                                  MAX_CONTEXTS);
            if (result != FFX_OK) {
                clearRuntimeLocked();
                return fail(LFS_SCENE_UPSCALER_PLUGIN_UNSUPPORTED_DEVICE,
                            fsrFailure("initializing FidelityFX Vulkan backend", result));
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
            if (!runtime_initialized_)
                return fail(LFS_SCENE_UPSCALER_PLUGIN_UNAVAILABLE,
                            "FidelityFX runtime is not initialized");
            if (settings.struct_size < sizeof(LfsSceneUpscalerOptimalSettingsV1) ||
                output_width < MIN_OUTPUT_EXTENT ||
                output_height < MIN_OUTPUT_EXTENT || !validQuality(quality)) {
                return fail(LFS_SCENE_UPSCALER_PLUGIN_INVALID_ARGUMENT,
                            "invalid AMD FSR 3.1 output extent or quality");
            }

            std::uint32_t render_width = 0;
            std::uint32_t render_height = 0;
            const auto result = ffxFsr3UpscalerGetRenderResolutionFromQualityMode(
                &render_width,
                &render_height,
                output_width,
                output_height,
                fsrQuality(quality));
            if (result != FFX_OK || render_width == 0 || render_height == 0)
                return fail(LFS_SCENE_UPSCALER_PLUGIN_RUNTIME_ERROR,
                            fsrFailure("querying AMD FSR 3.1 optimal settings", result));

            settings.render_width = render_width;
            settings.render_height = render_height;
            settings.minimum_width = render_width;
            settings.minimum_height = render_height;
            settings.maximum_width = render_width;
            settings.maximum_height = render_height;
            // Presets select only the SDK's reconstruction ratio. LichtFeld
            // does not add RCAS sharpening on top of the reconstructed image.
            settings.sharpness = 0.0f;
            last_error_.clear();
            return LFS_SCENE_UPSCALER_PLUGIN_OK;
        }

        LfsSceneUpscalerPluginResult createFeature(
            const VkCommandBuffer command_buffer,
            const LfsSceneUpscalerFeatureConfigV1& config) {
            std::scoped_lock lock(mutex_);
            if (!runtime_initialized_)
                return fail(LFS_SCENE_UPSCALER_PLUGIN_UNAVAILABLE,
                            "FidelityFX runtime is not initialized");
            if (command_buffer == VK_NULL_HANDLE ||
                config.struct_size < sizeof(LfsSceneUpscalerFeatureConfigV1) ||
                !validView(config.view) || !validQuality(config.quality) ||
                config.render_width == 0 || config.render_height == 0 ||
                config.output_width < MIN_OUTPUT_EXTENT ||
                config.output_height < MIN_OUTPUT_EXTENT ||
                config.render_width > config.output_width ||
                config.render_height > config.output_height) {
                return fail(LFS_SCENE_UPSCALER_PLUGIN_INVALID_ARGUMENT,
                            "invalid AMD FSR 3.1 feature configuration");
            }

            auto& view = views_[static_cast<std::size_t>(config.view)];
            if (view.configured && sameFeatureConfig(view.config, config)) {
                last_error_.clear();
                return LFS_SCENE_UPSCALER_PLUGIN_OK;
            }
            releaseFeatureLocked(view);

            FfxFsr3UpscalerContextDescription description{};
            // LichtFeld supplies a fixed-size LDR viewport signal and an explicit
            // neutral pre-exposure. Contexts are recreated when render or output
            // size changes, so dynamic-size support is not advertised.
            description.flags = 0;
            if (config.motion_vectors_include_jitter != 0) {
                description.flags |=
                    FFX_FSR3UPSCALER_ENABLE_MOTION_VECTORS_JITTER_CANCELLATION;
            }
            description.maxRenderSize =
                dimensions(config.render_width, config.render_height);
            description.maxUpscaleSize =
                dimensions(config.output_width, config.output_height);
            description.backendInterface = backend_;

            auto result = ffxFsr3UpscalerContextCreate(&view.context, &description);
            if (result != FFX_OK) {
                releaseFeatureLocked(view);
                return fail(LFS_SCENE_UPSCALER_PLUGIN_RUNTIME_ERROR,
                            fsrFailure("creating AMD FSR 3.1 context", result));
            }
            view.context_created = true;

            result = backend_.fpCreateBackendContext(&backend_,
                                                     FFX_EFFECT_SHAREDRESOURCES,
                                                     nullptr,
                                                     &view.shared_effect_context_id);
            if (result != FFX_OK) {
                releaseFeatureLocked(view);
                return fail(LFS_SCENE_UPSCALER_PLUGIN_RUNTIME_ERROR,
                            fsrFailure("creating FSR shared-resource context", result));
            }
            view.shared_context_created = true;

            FfxFsr3UpscalerSharedResourceDescriptions shared{};
            result = ffxFsr3UpscalerGetSharedResourceDescriptions(&view.context, &shared);
            if (result != FFX_OK ||
                !createSharedResource(shared.dilatedDepth, view) ||
                !createSharedResource(shared.dilatedMotionVectors, view) ||
                !createSharedResource(shared.reconstructedPrevNearestDepth, view)) {
                const auto detail = result == FFX_OK
                                        ? std::string("creating FSR shared resources failed")
                                        : fsrFailure("querying FSR shared resources", result);
                releaseFeatureLocked(view);
                return fail(LFS_SCENE_UPSCALER_PLUGIN_RUNTIME_ERROR, detail);
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
            constexpr std::size_t REQUIRED_EVALUATION_SIZE =
                offsetof(LfsSceneUpscalerEvaluateV1, view_space_to_meters) +
                sizeof(float);
            if (!runtime_initialized_)
                return fail(LFS_SCENE_UPSCALER_PLUGIN_UNAVAILABLE,
                            "FidelityFX runtime is not initialized");
            if (evaluation.struct_size < REQUIRED_EVALUATION_SIZE ||
                !validView(evaluation.view) ||
                evaluation.command_buffer == VK_NULL_HANDLE ||
                !validImage(evaluation.color, false) ||
                !validImage(evaluation.depth, false) ||
                !validImage(evaluation.motion, false) ||
                !validImage(evaluation.output, true)) {
                return fail(LFS_SCENE_UPSCALER_PLUGIN_INVALID_ARGUMENT,
                            "invalid AMD FSR 3.1 evaluation resources");
            }

            auto& view = views_[static_cast<std::size_t>(evaluation.view)];
            if (!view.configured || !view.context_created ||
                !view.shared_context_created ||
                view.shared_count != SHARED_RESOURCE_COUNT) {
                return fail(LFS_SCENE_UPSCALER_PLUGIN_UNAVAILABLE,
                            "AMD FSR 3.1 feature is not configured for this view");
            }
            if (!evaluationMatchesFeature(evaluation, view.config))
                return fail(LFS_SCENE_UPSCALER_PLUGIN_INVALID_ARGUMENT,
                            "AMD FSR 3.1 resources do not match the active feature");

            FfxFsr3UpscalerDispatchDescription parameters{};
            parameters.commandList = ffxGetCommandListVK(evaluation.command_buffer);
            parameters.color = wrapImage(evaluation.color, L"LFS FSR input color");
            parameters.depth = wrapImage(evaluation.depth, L"LFS FSR input depth");
            parameters.motionVectors =
                wrapImage(evaluation.motion, L"LFS FSR input motion");
            parameters.output = wrapImage(evaluation.output,
                                          L"LFS FSR output",
                                          FFX_RESOURCE_USAGE_UAV);
            // Reactive and transparency/composition masks stay null until the
            // renderer can publish semantically correct material signals. The
            // default null exposure resource is exactly 1.0 and matches the
            // explicit neutral pre-exposure supplied below.
            parameters.dilatedDepth = sharedResource(view, 0);
            parameters.dilatedMotionVectors = sharedResource(view, 1);
            parameters.reconstructedPrevNearestDepth = sharedResource(view, 2);
            parameters.jitterOffset = {
                evaluation.jitter_x_pixels,
                evaluation.jitter_y_pixels,
            };
            parameters.motionVectorScale = {
                evaluation.motion_scale_x,
                evaluation.motion_scale_y,
            };
            parameters.renderSize =
                dimensions(evaluation.color.valid_width, evaluation.color.valid_height);
            parameters.upscaleSize =
                dimensions(evaluation.output.valid_width, evaluation.output.valid_height);
            // RCAS is an optional post-pass, not part of the quality-mode
            // contract. It exaggerates high-frequency splat edges and produces
            // a visibly hard, stippled result after temporal convergence.
            parameters.enableSharpening = false;
            parameters.sharpness = 0.0f;
            parameters.frameTimeDelta = evaluation.frame_time_milliseconds;
            parameters.preExposure = evaluation.pre_exposure;
            parameters.reset = evaluation.reset_flags != 0;
            parameters.cameraNear = evaluation.camera_near;
            parameters.cameraFar = evaluation.camera_far;
            parameters.cameraFovAngleVertical =
                evaluation.camera_vertical_fov_radians;
            parameters.viewSpaceToMetersFactor = evaluation.view_space_to_meters;

            const auto result =
                ffxFsr3UpscalerContextDispatch(&view.context, &parameters);
            if (result != FFX_OK)
                return fail(LFS_SCENE_UPSCALER_PLUGIN_RUNTIME_ERROR,
                            fsrFailure("evaluating AMD FSR 3.1", result));
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

        std::size_t lastError(char* const destination,
                              const std::size_t capacity) const {
            std::scoped_lock lock(mutex_);
            const std::size_t required = last_error_.size() + 1;
            if (destination != nullptr && capacity != 0) {
                const std::size_t copy_size =
                    std::min(last_error_.size(), capacity - 1);
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
            const bool valid_layout =
                writable ? image.layout == VK_IMAGE_LAYOUT_GENERAL
                         : image.layout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            return image.struct_size >= sizeof(LfsSceneUpscalerImageV1) &&
                   image.image != VK_NULL_HANDLE &&
                   image.view != VK_NULL_HANDLE && valid_layout &&
                   image.format != VK_FORMAT_UNDEFINED &&
                   image.allocation_width != 0 &&
                   image.allocation_height != 0 &&
                   image.valid_width != 0 && image.valid_height != 0 &&
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
                   left.motion_vectors_include_jitter ==
                       right.motion_vectors_include_jitter;
        }

        [[nodiscard]] static bool evaluationMatchesFeature(
            const LfsSceneUpscalerEvaluateV1& evaluation,
            const LfsSceneUpscalerFeatureConfigV1& config) noexcept {
            return evaluation.color.valid_width == config.render_width &&
                   evaluation.color.valid_height == config.render_height &&
                   evaluation.depth.valid_width == config.render_width &&
                   evaluation.depth.valid_height == config.render_height &&
                   evaluation.motion.valid_width == config.render_width &&
                   evaluation.motion.valid_height == config.render_height &&
                   evaluation.output.valid_width == config.output_width &&
                   evaluation.output.valid_height == config.output_height &&
                   evaluation.color.format == VK_FORMAT_R8G8B8A8_UNORM &&
                   evaluation.depth.format == VK_FORMAT_R32_SFLOAT &&
                   evaluation.motion.format == VK_FORMAT_R16G16_SFLOAT &&
                   evaluation.output.format == VK_FORMAT_R8G8B8A8_UNORM &&
                   std::isfinite(evaluation.jitter_x_pixels) &&
                   std::isfinite(evaluation.jitter_y_pixels) &&
                   std::isfinite(evaluation.motion_scale_x) &&
                   std::isfinite(evaluation.motion_scale_y) &&
                   evaluation.motion_scale_x != 0.0f &&
                   evaluation.motion_scale_y != 0.0f &&
                   std::isfinite(evaluation.pre_exposure) &&
                   evaluation.pre_exposure > 0.0f &&
                   std::isfinite(evaluation.frame_time_milliseconds) &&
                   evaluation.frame_time_milliseconds > 0.0f &&
                   std::isfinite(evaluation.camera_near) &&
                   std::isfinite(evaluation.camera_far) &&
                   evaluation.camera_near > 0.0f &&
                   evaluation.camera_far > evaluation.camera_near &&
                   std::isfinite(evaluation.camera_vertical_fov_radians) &&
                   evaluation.camera_vertical_fov_radians > 0.0f &&
                   evaluation.camera_vertical_fov_radians <=
                       3.14159265358979323846f &&
                   std::isfinite(evaluation.view_space_to_meters) &&
                   evaluation.view_space_to_meters > 0.0f;
        }

        [[nodiscard]] bool createSharedResource(
            const FfxCreateResourceDescription& description,
            ViewState& view) {
            if (view.shared_count >= view.shared.size())
                return false;
            auto& resource = view.shared[view.shared_count];
            const auto result =
                backend_.fpCreateResource(&backend_,
                                          &description,
                                          view.shared_effect_context_id,
                                          &resource);
            if (result != FFX_OK)
                return false;
            ++view.shared_count;
            return true;
        }

        [[nodiscard]] FfxResource sharedResource(ViewState& view,
                                                 const std::size_t index) {
            return backend_.fpGetResource(&backend_, view.shared[index]);
        }

        void releaseFeatureLocked(ViewState& view) noexcept {
            if (view.shared_context_created) {
                for (std::size_t index = 0; index < view.shared_count; ++index) {
                    backend_.fpDestroyResource(&backend_,
                                               view.shared[index],
                                               view.shared_effect_context_id);
                }
                backend_.fpDestroyBackendContext(&backend_,
                                                 view.shared_effect_context_id);
            }
            if (view.context_created)
                ffxFsr3UpscalerContextDestroy(&view.context);
            view = {};
        }

        void clearRuntimeLocked() noexcept {
            backend_ = {};
            scratch_.clear();
            runtime_initialized_ = false;
        }

        void shutdownRuntimeLocked() noexcept {
            for (auto& view : views_)
                releaseFeatureLocked(view);
            clearRuntimeLocked();
        }

        LfsSceneUpscalerPluginResult fail(
            const LfsSceneUpscalerPluginResult result,
            std::string error) {
            last_error_ = std::move(error);
            return result;
        }

        mutable std::mutex mutex_;
        FfxInterface backend_{};
        std::vector<std::byte> scratch_;
        std::array<ViewState, VIEW_COUNT> views_{};
        std::string last_error_;
        bool runtime_initialized_ = false;
    };

    [[nodiscard]] AmdFsr3Plugin* pluginFrom(void* const plugin) noexcept {
        return static_cast<AmdFsr3Plugin*>(plugin);
    }

    void* createPlugin(const LfsSceneUpscalerBootstrapConfigV1* const config) noexcept {
        try {
            if (config == nullptr ||
                config->struct_size < sizeof(LfsSceneUpscalerBootstrapConfigV1) ||
                config->project_id == nullptr ||
                config->engine_version == nullptr ||
                config->application_data_path == nullptr ||
                config->plugin_directory == nullptr) {
                return nullptr;
            }
            return new AmdFsr3Plugin(*config);
        } catch (...) {
            // LFS-CENSUS-OK(empty-catch): construction is a C ABI boundary with no
            // plugin instance available for diagnostics; nullptr reports failure.
            return nullptr;
        }
    }

    void destroyPlugin(void* const plugin) noexcept { delete pluginFrom(plugin); }

    LfsSceneUpscalerPluginResult requiredInstanceExtensions(
        void* const plugin,
        const LfsSceneUpscalerExtensionSink* const sink) noexcept {
        try {
            return plugin != nullptr && sink != nullptr
                       ? pluginFrom(plugin)->requiredInstanceExtensions(*sink)
                       : LFS_SCENE_UPSCALER_PLUGIN_INVALID_ARGUMENT;
        } catch (...) {
            // LFS-CENSUS-OK(empty-catch): never unwind through the C ABI.
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
            // LFS-CENSUS-OK(empty-catch): never unwind through the C ABI.
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
            // LFS-CENSUS-OK(empty-catch): never unwind through the C ABI.
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
            // LFS-CENSUS-OK(empty-catch): never unwind through the C ABI.
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
            // LFS-CENSUS-OK(empty-catch): never unwind through the C ABI.
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
            // LFS-CENSUS-OK(empty-catch): never unwind through the C ABI.
            return LFS_SCENE_UPSCALER_PLUGIN_RUNTIME_ERROR;
        }
    }

    void releaseFeature(void* const plugin, const std::uint32_t view) noexcept {
        if (plugin != nullptr) {
            try {
                pluginFrom(plugin)->releaseFeature(view);
            } catch (...) {
                // LFS-CENSUS-OK(empty-catch): noexcept cleanup boundary.
            }
        }
    }

    void shutdownRuntime(void* const plugin) noexcept {
        if (plugin != nullptr) {
            try {
                pluginFrom(plugin)->shutdownRuntime();
            } catch (...) {
                // LFS-CENSUS-OK(empty-catch): noexcept cleanup boundary.
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
            // LFS-CENSUS-OK(empty-catch): diagnostics are best effort.
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
