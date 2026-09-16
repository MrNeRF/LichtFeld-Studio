/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/export.hpp"
#include "rendering/scene_upscaler_plugin_api.h"

#include <cstdint>
#include <filesystem>
#include <mutex>
#include <optional>
#include <set>
#include <string>
#include <vector>

namespace lfs::vis {

    enum class NvidiaDlssPluginState : std::uint8_t {
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

    /*
     * Allocates identities for one host process. Dynamic-capability plugins
     * use IDs above the legacy three-view range; older plugins are limited to
     * those three IDs and therefore fail allocation instead of aliasing them.
     */
    class LFS_VIS_API NvidiaDlssViewIdentityAllocator final {
    public:
        explicit NvidiaDlssViewIdentityAllocator(bool dynamic_view_ids) noexcept
            : dynamic_view_ids_(dynamic_view_ids) {}

        [[nodiscard]] std::optional<std::uint32_t> acquire() {
            const auto first = dynamic_view_ids_
                                   ? static_cast<std::uint32_t>(
                                         LFS_SCENE_UPSCALER_PLUGIN_VIEW_COUNT)
                                   : 0u;
            const auto end = dynamic_view_ids_
                                 ? static_cast<std::uint64_t>(
                                       LFS_SCENE_UPSCALER_PLUGIN_VIEW_INVALID)
                                 : static_cast<std::uint64_t>(
                                       LFS_SCENE_UPSCALER_PLUGIN_VIEW_COUNT);
            for (std::uint64_t candidate = first; candidate < end; ++candidate) {
                const auto id = static_cast<std::uint32_t>(candidate);
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
        bool dynamic_view_ids_ = false;
        std::set<std::uint32_t> allocated_;
    };

    class LFS_VIS_API NvidiaDlssPlugin final {
    public:
        static NvidiaDlssPlugin& instance();

        NvidiaDlssPlugin(const NvidiaDlssPlugin&) = delete;
        NvidiaDlssPlugin& operator=(const NvidiaDlssPlugin&) = delete;

        void configure(bool loading_enabled);
        [[nodiscard]] bool probe();
        [[nodiscard]] bool available();
        [[nodiscard]] NvidiaDlssPluginState state() const;
        [[nodiscard]] std::string diagnostic() const;
        [[nodiscard]] std::filesystem::path libraryPath() const;

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
        NvidiaDlssPlugin();
        ~NvidiaDlssPlugin();

        struct Impl;
        Impl* impl_;
    };

    LFS_VIS_API void configureNvidiaDlssPluginLoading(bool enabled);
    [[nodiscard]] LFS_VIS_API bool nvidiaDlssPluginAvailable();

} // namespace lfs::vis
