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
