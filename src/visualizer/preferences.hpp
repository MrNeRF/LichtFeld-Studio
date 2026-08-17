/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/export.hpp"

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

} // namespace lfs::vis
