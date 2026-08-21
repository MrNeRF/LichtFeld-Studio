/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/error.hpp"
#include "core/export.hpp"

#include <filesystem>
#include <memory>
#include <string>

namespace lfs::vis {

    struct McpPreferenceState {
        bool enabled = true;
        bool expose_network = false;
        int port = 45677;
        bool request_logging = false;
    };

    /** Process-local, atomically persisted user preferences. */
    class LFS_VIS_API UserPreferences {
    public:
        static UserPreferences& instance();
        ~UserPreferences();

        UserPreferences(const UserPreferences&) = delete;
        UserPreferences& operator=(const UserPreferences&) = delete;

        void setThemeName(const std::string& value);
        [[nodiscard]] std::string themeName();
        void setUiScale(float value);
        [[nodiscard]] float uiScale();

        void setLanguage(const std::string& value);
        [[nodiscard]] std::string language();
        void clearLanguage();

        void setCameraNavigation(const std::string& value);
        [[nodiscard]] std::string cameraNavigation();
        void setRememberCameraNavigation(bool enabled);
        [[nodiscard]] bool rememberCameraNavigation();
        void setCameraViewSnap(bool enabled);
        [[nodiscard]] bool cameraViewSnap();
        void setRememberCameraViewSnap(bool enabled);
        [[nodiscard]] bool rememberCameraViewSnap();

        void setMcp(const McpPreferenceState& state);
        [[nodiscard]] McpPreferenceState mcp();

        void setSceneUpscaler(const std::string& backend_id, const std::string& preset_id);
        void clearSceneUpscaler();
        [[nodiscard]] std::string sceneUpscaler();
        [[nodiscard]] std::string sceneUpscalerPreset(const std::string& backend_id);

        [[nodiscard]] lfs::Status setWorkingDirectory(const std::filesystem::path& path);
        [[nodiscard]] std::filesystem::path workingDirectory();
        [[nodiscard]] std::filesystem::path workingDirectoryPreference();
        void clearWorkingDirectory();

    private:
        UserPreferences();
        struct Impl;
        std::unique_ptr<Impl> impl_;
    };

    LFS_VIS_API void saveLanguagePreference(const std::string& language_code);
    [[nodiscard]] LFS_VIS_API std::string loadLanguagePreference();
    LFS_VIS_API void clearLanguagePreference();

    LFS_VIS_API void saveCameraNavigationPreference(const std::string& mode);
    [[nodiscard]] LFS_VIS_API std::string loadCameraNavigationPreference();
    LFS_VIS_API void setRememberCameraNavigationPreference(bool enabled);
    [[nodiscard]] LFS_VIS_API bool rememberCameraNavigationPreference();
    LFS_VIS_API void saveCameraViewSnapPreference(bool enabled);
    [[nodiscard]] LFS_VIS_API bool loadCameraViewSnapPreference();
    LFS_VIS_API void setRememberCameraViewSnapPreference(bool enabled);
    [[nodiscard]] LFS_VIS_API bool rememberCameraViewSnapPreference();
    LFS_VIS_API void saveMcpPreferences(const McpPreferenceState& state);
    [[nodiscard]] LFS_VIS_API McpPreferenceState loadMcpPreferences();
    LFS_VIS_API void saveSceneUpscalerPreference(const std::string& backend_id,
                                                 const std::string& preset_id);
    LFS_VIS_API void clearSceneUpscalerPreference();
    [[nodiscard]] LFS_VIS_API std::string loadSceneUpscalerPreference();
    [[nodiscard]] LFS_VIS_API std::string loadSceneUpscalerPresetPreference(
        const std::string& backend_id);

    [[nodiscard]] LFS_VIS_API lfs::Status
    setWorkingDirectoryPreference(const std::filesystem::path& path);
    [[nodiscard]] LFS_VIS_API std::filesystem::path loadWorkingDirectoryPreference();
    [[nodiscard]] LFS_VIS_API std::filesystem::path workingDirectoryPreferenceRaw();
    LFS_VIS_API void clearWorkingDirectoryPreference();
    [[nodiscard]] LFS_VIS_API std::filesystem::path defaultWorkingDirectory();
    [[nodiscard]] LFS_VIS_API std::filesystem::path tempProjectDirectoryPreference();

} // namespace lfs::vis
