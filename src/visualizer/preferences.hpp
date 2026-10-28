/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/error.hpp"
#include "core/export.hpp"
#include "gui/utils/splat_drag_preferences.hpp"

#include <filesystem>
#include <memory>
#include <string>
#include <string_view>

namespace lfs::vis {

    struct RenderSettings;

    // Snapshot defaults seed new projects; restored VIEW state remains authoritative.
    [[nodiscard]] LFS_VIS_API std::string renderingDefaultsFromSettings(const RenderSettings& settings);
    [[nodiscard]] LFS_VIS_API lfs::Status applyRenderingDefaults(RenderSettings& settings, std::string_view serialized);

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

        [[nodiscard]] lfs::Status setRenderingDefaults(std::string_view serialized);
        [[nodiscard]] std::string renderingDefaults();
        [[nodiscard]] lfs::Status setGamepadNavigation(std::string_view serialized);
        [[nodiscard]] std::string gamepadNavigation();
        [[nodiscard]] std::string gamepadProfiles();
        [[nodiscard]] lfs::Status setGamepadProfiles(std::string_view serialized, bool enabled, std::string_view expected = {});
        [[nodiscard]] lfs::Status setViewportVignette(std::string_view serialized);
        [[nodiscard]] std::string viewportVignette();
        void setSidebarDefaults(float width, float scene_ratio, const std::string& active_tab);
        [[nodiscard]] std::string sidebarDefaults();
        void clearSidebarDefaults();

        void setThemeName(const std::string& value);
        [[nodiscard]] std::string themeName();
        void setUiScale(float value);
        [[nodiscard]] float uiScale();
        void setNativeFloatingWindows(bool enabled);
        [[nodiscard]] bool nativeFloatingWindows();
        void setVrEnabled(bool enabled);
        [[nodiscard]] bool vrEnabled();
        void setVrResolutionScale(float value);
        [[nodiscard]] float vrResolutionScale();
        void setVrSceneScale(float value);
        [[nodiscard]] float vrSceneScale();
        void setVrMoveSpeed(float value);
        [[nodiscard]] float vrMoveSpeed();
        void setFloatingWindowGeometry(const std::string& value);
        [[nodiscard]] std::string floatingWindowGeometry();
        void setZoomSpeed(float value, bool persist = true);
        [[nodiscard]] float zoomSpeed();
        void setNavigationSpeed(float value, bool persist = true);
        [[nodiscard]] float navigationSpeed();

        void setCameraBrowseRepeat(bool enabled);
        [[nodiscard]] bool cameraBrowseRepeat();
        void setCameraBrowseRepeatRate(float value, bool persist = true);
        [[nodiscard]] float cameraBrowseRepeatRate();

        void setLanguage(const std::string& value);
        [[nodiscard]] std::string language();
        void clearLanguage();

        void setNewProjectNavigation(std::string_view axis, bool align_grid);
        [[nodiscard]] std::string newProjectNavigationUpAxis();
        [[nodiscard]] bool alignNewProjectGrid();

        void setCameraNavigation(const std::string& value);
        [[nodiscard]] std::string cameraNavigation();
        void setRememberCameraNavigation(bool enabled);
        [[nodiscard]] bool rememberCameraNavigation();
        void setCameraViewSnap(bool enabled);
        [[nodiscard]] bool cameraViewSnap();
        void setRememberCameraViewSnap(bool enabled);
        [[nodiscard]] bool rememberCameraViewSnap();
        void setSceneGraphSelectionMarkers(bool enabled);
        [[nodiscard]] bool sceneGraphSelectionMarkers();
        void setProgressBarStyle(std::string_view value);
        [[nodiscard]] std::string progressBarStyle();
        void setViewportChromeStyle(std::string_view value);
        [[nodiscard]] std::string viewportChromeStyle();
        void setViewportToolbarPosition(std::string_view value);
        [[nodiscard]] std::string viewportToolbarPosition();
        void setViewportToolbarFreeY(float value);
        [[nodiscard]] float viewportToolbarFreeY();

        void setMarketplaceView(std::string_view value);
        [[nodiscard]] std::string marketplaceView();
        void setMarketplaceGitCheckout(bool enabled);
        [[nodiscard]] bool marketplaceGitCheckout();
        void resetMarketplacePreferences();

        void setProjectManagerDefaultView(std::string_view value);
        [[nodiscard]] std::string projectManagerDefaultView();
        void setOpenProjectManagerAtStartup(bool enabled);
        [[nodiscard]] bool openProjectManagerAtStartup();
        void setRememberProjectManagerState(bool enabled);
        [[nodiscard]] bool rememberProjectManagerState();
        void setProjectManagerState(std::string_view serialized_state);
        [[nodiscard]] std::string projectManagerState();
        void resetProjectManagerPreferences();

        void setSplatDrag(const SplatDragPreferenceState& state);
        [[nodiscard]] SplatDragPreferenceState splatDrag();

        void setMcp(const McpPreferenceState& state);
        [[nodiscard]] McpPreferenceState mcp();

        void setSceneUpscaler(const std::string& backend_id, const std::string& preset_id);
        void clearSceneUpscaler();
        [[nodiscard]] std::string sceneUpscaler();
        [[nodiscard]] std::string sceneUpscalerPreset(const std::string& backend_id);

        [[nodiscard]] lfs::Status setProjectLocation(
            const std::filesystem::path& path);
        [[nodiscard]] std::filesystem::path projectLocation();
        [[nodiscard]] std::filesystem::path projectLocationPreference();
        void clearProjectLocation();

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
    LFS_VIS_API void saveZoomSpeedPreference(float speed, bool persist = true);
    [[nodiscard]] LFS_VIS_API float loadZoomSpeedPreference();
    LFS_VIS_API void saveNavigationSpeedPreference(float speed, bool persist = true);
    [[nodiscard]] LFS_VIS_API float loadNavigationSpeedPreference();
    LFS_VIS_API void setRememberCameraNavigationPreference(bool enabled);
    [[nodiscard]] LFS_VIS_API bool rememberCameraNavigationPreference();
    LFS_VIS_API void saveCameraViewSnapPreference(bool enabled);
    [[nodiscard]] LFS_VIS_API bool loadCameraViewSnapPreference();
    LFS_VIS_API void setRememberCameraViewSnapPreference(bool enabled);
    [[nodiscard]] LFS_VIS_API bool rememberCameraViewSnapPreference();
    LFS_VIS_API void saveSceneGraphSelectionMarkersPreference(bool enabled);
    [[nodiscard]] LFS_VIS_API bool loadSceneGraphSelectionMarkersPreference();
    LFS_VIS_API void saveProgressBarStylePreference(std::string_view style);
    [[nodiscard]] LFS_VIS_API std::string loadProgressBarStylePreference();
    LFS_VIS_API void saveViewportChromeStylePreference(std::string_view style);
    [[nodiscard]] LFS_VIS_API std::string loadViewportChromeStylePreference();
    LFS_VIS_API void saveViewportToolbarPositionPreference(std::string_view position);
    [[nodiscard]] LFS_VIS_API std::string loadViewportToolbarPositionPreference();
    LFS_VIS_API void saveViewportToolbarFreeYPreference(float value);
    [[nodiscard]] LFS_VIS_API float loadViewportToolbarFreeYPreference();
    LFS_VIS_API void saveMcpPreferences(const McpPreferenceState& state);
    [[nodiscard]] LFS_VIS_API McpPreferenceState loadMcpPreferences();
    LFS_VIS_API void saveSceneUpscalerPreference(const std::string& backend_id,
                                                 const std::string& preset_id);
    LFS_VIS_API void clearSceneUpscalerPreference();
    [[nodiscard]] LFS_VIS_API std::string loadSceneUpscalerPreference();
    [[nodiscard]] LFS_VIS_API std::string loadSceneUpscalerPresetPreference(
        const std::string& backend_id);

    [[nodiscard]] LFS_VIS_API std::filesystem::path workingDirectoryPreferenceRaw();
    [[nodiscard]] LFS_VIS_API lfs::Status
    setProjectLocationPreference(const std::filesystem::path& path);
    [[nodiscard]] LFS_VIS_API std::filesystem::path
    loadProjectLocationPreference();
    [[nodiscard]] LFS_VIS_API std::filesystem::path
    projectLocationPreferenceRaw();
    // The configured project location as an absolute path, resolved without disk
    // access. An empty path when none is configured.
    [[nodiscard]] LFS_VIS_API std::filesystem::path
    resolvedProjectLocationPreference();
    // The create-and-write probe that setProjectLocationPreference applies. It touches
    // the disk, so keep it off the UI thread.
    [[nodiscard]] LFS_VIS_API lfs::Status
    probeProjectLocation(const std::filesystem::path& path);
    LFS_VIS_API void clearProjectLocationPreference();
    [[nodiscard]] LFS_VIS_API std::filesystem::path defaultProjectLocation();

} // namespace lfs::vis
