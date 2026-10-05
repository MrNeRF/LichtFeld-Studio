/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/export.hpp"
#include "rendering/scene_upscaler_registry.hpp"

#ifdef LFS_GRAPHICS_VULKAN
#include "rendering/scene_upscaler_plugin_api.h"
#endif

#include <cstdint>
#include <filesystem>
#include <mutex>
#include <optional>
#include <set>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace lfs::vis {

    enum class SceneUpscalerPluginState : std::uint8_t {
        Unprobed = 0,
        DisabledBySafeMode,
        NotInstalled,
        InvalidPlugin,
        BootstrapFailed,
        BootstrapReady,
        RuntimeReady,
        RuntimeMissing,
        UnsupportedEnvironment,
        RuntimeFailed,
    };

    // Backend-neutral identity for one optional vendor plugin. The runtime ABI
    // itself remains excluded from Metal-only builds below.
    struct SceneUpscalerPluginInfo {
        SceneUpscalerBackend backend;
        std::string_view id;
        std::string_view name;
        std::string_view directory;
        std::string_view library;
        std::string_view cache_dir;
    };

#ifdef LFS_GRAPHICS_VULKAN
    /*
     * Allocates identities for one host process. Dynamic-capability plugins
     * use IDs above the legacy three-view range; older plugins are limited to
     * those three IDs and therefore fail allocation instead of aliasing them.
     */
    class LFS_VIS_API SceneUpscalerPluginViewIdentityAllocator final {
    public:
        explicit SceneUpscalerPluginViewIdentityAllocator(bool dynamic_view_ids) noexcept
            : dynamic_view_ids_(dynamic_view_ids) {}

        [[nodiscard]] std::optional<std::uint32_t> acquire() {
            if (dynamic_view_ids_) {
                if (next_dynamic_ == LFS_SCENE_UPSCALER_PLUGIN_VIEW_INVALID)
                    return std::nullopt;
                const auto id = next_dynamic_++;
                allocated_.insert(id);
                return id;
            }
            for (std::uint32_t id = 0; id < LFS_SCENE_UPSCALER_PLUGIN_VIEW_COUNT; ++id) {
                if (allocated_.insert(id).second)
                    return id;
            }
            return std::nullopt;
        }

        void release(const std::uint32_t view) noexcept { allocated_.erase(view); }

        [[nodiscard]] bool owns(const std::uint32_t view) const noexcept {
            return allocated_.contains(view);
        }

    private:
        std::uint32_t next_dynamic_ = LFS_SCENE_UPSCALER_PLUGIN_VIEW_COUNT;
        bool dynamic_view_ids_ = false;
        std::set<std::uint32_t> allocated_;
    };

    class LFS_VIS_API SceneUpscalerPlugin final {
    public:
        explicit SceneUpscalerPlugin(const SceneUpscalerPluginInfo& info);
        ~SceneUpscalerPlugin();

        SceneUpscalerPlugin(const SceneUpscalerPlugin&) = delete;
        SceneUpscalerPlugin& operator=(const SceneUpscalerPlugin&) = delete;

        [[nodiscard]] const SceneUpscalerPluginInfo& info() const noexcept { return info_; }
        void configure(bool loading_enabled);
        [[nodiscard]] bool probe();
        [[nodiscard]] bool available();
        [[nodiscard]] std::string diagnostic() const;
        [[nodiscard]] bool hasCapability(LfsSceneUpscalerPluginCapability capability);

        [[nodiscard]] std::vector<std::string> requiredInstanceExtensions();
        [[nodiscard]] std::vector<std::string> requiredDeviceExtensions(
            VkInstance instance, VkPhysicalDevice physical_device);
        void markBootstrapFailed(std::string reason);
        [[nodiscard]] bool initializeRuntime(const LfsSceneUpscalerRuntimeConfigV1& config);
        [[nodiscard]] std::optional<LfsSceneUpscalerOptimalSettingsV1> optimalSettings(
            std::uint32_t output_width,
            std::uint32_t output_height,
            std::uint32_t quality);
        [[nodiscard]] bool createFeature(VkCommandBuffer command_buffer,
                                         const LfsSceneUpscalerFeatureConfigV1& config);
        [[nodiscard]] bool evaluate(const LfsSceneUpscalerEvaluateV1& evaluation);
        void releaseFeature(std::uint32_t view);
        [[nodiscard]] std::optional<std::uint32_t> acquireViewIdentity();
        void releaseViewIdentity(std::uint32_t view);
        void shutdownRuntime();
        void shutdown();

    private:
        struct Impl;
        SceneUpscalerPluginInfo info_;
        Impl* impl_;
    };
#else
    // Vendor reconstruction plugins expose Vulkan objects in their ABI.  Keep
    // that ABI entirely out of a Metal-only process while preserving the
    // backend-neutral availability queries used by preferences and scripting.
    class LFS_VIS_API SceneUpscalerPlugin final {
    public:
        explicit SceneUpscalerPlugin(const SceneUpscalerPluginInfo& info) : info_(info) {}
        [[nodiscard]] const SceneUpscalerPluginInfo& info() const noexcept { return info_; }
        [[nodiscard]] constexpr bool available() const noexcept { return false; }

    private:
        SceneUpscalerPluginInfo info_;
    };
#endif

    // Every optional plugin backend the host knows how to discover.
    [[nodiscard]] LFS_VIS_API std::span<SceneUpscalerPlugin* const> sceneUpscalerPlugins();
    // nullptr for built-in backends.
    [[nodiscard]] LFS_VIS_API SceneUpscalerPlugin* sceneUpscalerPlugin(SceneUpscalerBackend backend);
    LFS_VIS_API void configureSceneUpscalerPluginLoading(bool enabled);

} // namespace lfs::vis
