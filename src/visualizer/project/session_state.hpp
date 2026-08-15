/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#pragma once

#include "core/error.hpp"
#include "core/export.hpp"
#include "internal/viewport.hpp"
#include "io/project_chapters.hpp"
#include "io/session_chapters.hpp"
#include "rendering/rendering_types.hpp"

#include <array>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace lfs::vis {
    class VisualizerImpl;

    namespace project {

        using SessionJson = lfs::io::JsonChapterDom::Json;

        struct PanelCameraProjectState {
            std::array<float, 9> rotation{
                1.0f, 0.0f, 0.0f,
                0.0f, 1.0f, 0.0f,
                0.0f, 0.0f, 1.0f};
            std::array<float, 3> translation{
                -5.657f, 3.0f, -5.657f};
            std::array<float, 3> pivot{};
            std::array<float, 9> home_rotation{
                1.0f, 0.0f, 0.0f,
                0.0f, 1.0f, 0.0f,
                0.0f, 0.0f, 1.0f};
            std::array<float, 3> home_translation{
                -5.657f, 3.0f, -5.657f};
            std::array<float, 3> home_pivot{};
            bool home_saved = true;
            float zoom_speed = 11.0f;
            float max_zoom_speed = 100.0f;
            float rotate_speed = 0.001f;
            float centre_speed = 0.002f;
            float roll_speed = 0.01f;
            float translate_speed = 0.0005f;
            float wasd_speed = 8.0f;
            float max_wasd_speed = 100.0f;
            std::optional<float> ortho_scale;

            friend bool operator==(
                const PanelCameraProjectState&,
                const PanelCameraProjectState&) = default;
        };

        struct CameraBookmarkProjectState {
            std::string id;
            std::string name;
            PanelCameraProjectState camera;

            friend bool operator==(
                const CameraBookmarkProjectState&,
                const CameraBookmarkProjectState&) = default;
        };

        [[nodiscard]] LFS_VIS_API SessionJson
        renderSettingsToProjectJson(
            const RenderSettings& settings);
        [[nodiscard]] LFS_VIS_API lfs::Result<RenderSettings>
        renderSettingsFromProjectJson(
            const SessionJson& json,
            const RenderSettings& base = {});

        [[nodiscard]] LFS_VIS_API PanelCameraProjectState
        capturePanelCameraProjectState(
            const Viewport& viewport);
        LFS_VIS_API void applyPanelCameraProjectState(
            Viewport& viewport,
            const PanelCameraProjectState& state);
        [[nodiscard]] LFS_VIS_API SessionJson
        panelCameraProjectStateToJson(
            std::string_view panel,
            const PanelCameraProjectState& state);
        [[nodiscard]] LFS_VIS_API
            lfs::Result<PanelCameraProjectState>
            panelCameraProjectStateFromJson(
                const SessionJson& json);

        struct PreparedGuiSessionRestore {
            lfs::io::project::ProjectSessionChapters chapters;
            // REFS-resolved paths (open-time). Apply injects these into live
            // owners; VIEW/SEQR chapters store UUIDs, not raw machine paths.
            std::optional<std::filesystem::path> environment_map_path;
            std::optional<std::filesystem::path> ply_sequence_directory;
            std::uint64_t ticket = 0;
        };

        using GuiSessionRestoreTicket = std::uint64_t;

        [[nodiscard]] LFS_VIS_API
            lfs::Result<PreparedGuiSessionRestore>
            prepareGuiSessionRestore(
                lfs::io::project::ProjectSessionChapters chapters,
                const lfs::io::project::ReferencesChapter* references =
                    nullptr,
                const std::filesystem::path& project_root = {});

        // Once startup made its preload attempt, not_started is terminal for
        // native-only/autoload-off sessions. Enabled autoload publishes
        // discovering synchronously before the attempt returns.
        [[nodiscard]] LFS_VIS_API bool
        pluginPreloadTerminalForGuiPanels(
            bool start_attempted,
            std::string_view state) noexcept;

        // Hydration stages a complete five-chapter bundle first. The bundle
        // becomes observable to GUI owners only after both startup gates.
        class LFS_VIS_API GuiSessionRestoreCoordinator {
        public:
            [[nodiscard]] lfs::Result<void> stage(
                lfs::io::project::ProjectSessionChapters chapters);
            [[nodiscard]] GuiSessionRestoreTicket stagePrepared(
                PreparedGuiSessionRestore prepared);
            void onFirstGuiFrame();
            void onPanelsReady(std::uint64_t registration_revision);
            [[nodiscard]] bool ready() const noexcept;
            [[nodiscard]] bool hasPending() const noexcept {
                return pending_.has_value();
            }
            [[nodiscard]] std::uint64_t
            panelsRegistrationRevision() const noexcept {
                return panels_registration_revision_;
            }
            [[nodiscard]] std::optional<
                PreparedGuiSessionRestore>
            takeReady();
            void clear() noexcept;
            [[nodiscard]] bool isCurrent(
                GuiSessionRestoreTicket ticket) const noexcept {
                return pending_ticket_ == ticket && ticket != 0;
            }

        private:
            std::optional<PreparedGuiSessionRestore> pending_;
            GuiSessionRestoreTicket next_ticket_ = 0;
            GuiSessionRestoreTicket pending_ticket_ = 0;
            bool first_gui_frame_ready_ = false;
            bool panels_ready_ = false;
            std::uint64_t panels_registration_revision_ = 0;
        };

        [[nodiscard]] LFS_VIS_API
            lfs::Result<lfs::io::project::ProjectSessionChapters>
            captureGuiSession(
                const VisualizerImpl& viewer,
                const lfs::io::project::ProjectSessionChapters&
                    retained,
                const std::vector<
                    CameraBookmarkProjectState>& bookmarks,
                lfs::io::project::ReferencesChapter* references =
                    nullptr,
                const std::filesystem::path& project_root = {});

        LFS_VIS_API void applyGuiSession(
            VisualizerImpl& viewer,
            const PreparedGuiSessionRestore& prepared,
            std::vector<CameraBookmarkProjectState>& bookmarks);
        LFS_VIS_API void applyGuiSessionTools(
            VisualizerImpl& viewer,
            const PreparedGuiSessionRestore& prepared);
        LFS_VIS_API void applyDefaultGuiLayout(
            VisualizerImpl& viewer);

    } // namespace project
} // namespace lfs::vis
