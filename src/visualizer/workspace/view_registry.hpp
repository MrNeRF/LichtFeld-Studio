/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/error.hpp"
#include "core/export.hpp"
#include "internal/viewport.hpp"
#include "rendering/depth_window_state.hpp"
#include "workspace/view_id.hpp"
#include "workspace/view_types.hpp"

#include <glm/glm.hpp>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace lfs::vis {

    struct ViewRecord {
        ViewId id = kInvalidViewId;
        Viewport camera;
        // The camera is retained when an area changes editor. These fields
        // identify the current editor and preserve opaque per-editor chrome
        // state while the area is switched away from the viewport.
        std::string editor_id = "viewport";
        std::unordered_map<std::string, std::string> chrome;
        // Transient camera association, invalidated by navigation or scene reload.
        // It is not serialized.
        struct CameraBinding {
            int uid = -1;
            std::uint64_t camera_list_generation = 0;
            glm::mat3 rotation{1.0f};
            glm::vec3 translation{0.0f};
            glm::vec3 pivot{0.0f};
            ViewProjectionState projection{};

            [[nodiscard]] bool valid() const noexcept { return uid >= 0; }
        };
        std::optional<CameraBinding> camera_binding;
        ViewProjectionState projection{};
        DepthWindowState depth{};
        int grid_plane = 1;
        std::uint64_t state_generation = 1;
    };

    // Persistent camera/projection payload. No inertia, pointers, or IO types.
    struct ViewPersistentState {
        ViewId id = kInvalidViewId;
        glm::mat3 rotation{1.0f};
        glm::vec3 translation{-5.657f, 3.0f, -5.657f};
        glm::vec3 pivot{0.0f};
        glm::mat3 home_rotation{1.0f};
        glm::vec3 home_translation{-5.657f, 3.0f, -5.657f};
        glm::vec3 home_pivot{0.0f};
        bool home_saved = true;
        float zoom_speed = 11.0f;
        float max_zoom_speed = 100.0f;
        float rotate_speed = 0.001f;
        float rotate_center_speed = 0.002f;
        float rotate_roll_speed = 0.01f;
        float translate_speed = 0.0005f;
        float wasd_speed = 8.0f;
        float max_wasd_speed = 100.0f;
        std::optional<float> ortho_scale_override;
        std::string editor_id = "viewport";
        std::unordered_map<std::string, std::string> chrome;
        ViewProjectionState projection{};
        DepthWindowState depth{};
        int grid_plane = 1;
        glm::ivec2 window_size{1280, 720};
        glm::ivec2 framebuffer_size{1280, 720};
        std::uint64_t state_generation = 1;
    };

    // Copy pose/pivot/home/speeds only. Destination transients stay cleared.
    LFS_VIS_API void clonePersistentViewport(const Viewport& source, Viewport& destination);
    LFS_VIS_API void resetTransientNavigation(Viewport& viewport);
    LFS_VIS_API void initializeViewportSizes(Viewport& viewport, glm::ivec2 logical,
                                             glm::ivec2 framebuffer = {});
    [[nodiscard]] LFS_VIS_API ViewPersistentState captureViewPersistentState(const ViewRecord& record);
    LFS_VIS_API void applyViewPersistentState(ViewRecord& record, const ViewPersistentState& state);
    [[nodiscard]] LFS_VIS_API lfs::Status validateViewPersistentState(const ViewPersistentState& state);

    class LFS_VIS_API ViewRegistry {
    public:
        ViewRegistry() = default;
        ViewRegistry(const ViewRegistry&) = delete;
        ViewRegistry& operator=(const ViewRegistry&) = delete;
        ViewRegistry(ViewRegistry&&) noexcept = default;
        ViewRegistry& operator=(ViewRegistry&&) noexcept = default;
        ~ViewRegistry() = default;

        [[nodiscard]] lfs::Result<ViewId> create(glm::ivec2 logical_size = {1280, 720},
                                                 std::optional<ViewId> clone_from = {});
        [[nodiscard]] bool contains(ViewId id) const noexcept;
        [[nodiscard]] ViewRecord* find(ViewId id) noexcept;
        [[nodiscard]] const ViewRecord* find(ViewId id) const noexcept;
        [[nodiscard]] bool erase(ViewId id);
        [[nodiscard]] std::vector<ViewId> ids() const;
        [[nodiscard]] std::size_t size() const noexcept { return records_.size(); }
        [[nodiscard]] ViewId nextId() const noexcept { return next_id_; }
        [[nodiscard]] std::uint64_t generation() const noexcept { return generation_; }

        // Keep the generation monotonic when replacing the registry from a
        // persisted workspace. This does not fabricate a view mutation; it
        // only prevents restored state from looking older to frame consumers.
        void bumpGenerationAtLeast(std::uint64_t at_least) noexcept;

        [[nodiscard]] lfs::Status setLogicalSize(ViewId id, glm::ivec2 size);
        [[nodiscard]] lfs::Status setFramebufferSize(ViewId id, glm::ivec2 size);
        [[nodiscard]] lfs::Status setProjection(ViewId id, const ViewProjectionState& projection);
        [[nodiscard]] lfs::Status setDepthWindow(ViewId id, const DepthWindowState& depth);
        [[nodiscard]] lfs::Status setGridPlane(ViewId id, int grid_plane);
        [[nodiscard]] lfs::Status setEditorId(ViewId id, std::string editor_id);
        [[nodiscard]] lfs::Status bumpState(ViewId id);

        [[nodiscard]] std::vector<ViewPersistentState> exportStates() const;
        [[nodiscard]] lfs::Status adopt(ViewPersistentState state);

        // Never decreases. Used so restore cannot reissue retired IDs.
        void bumpNextId(ViewId at_least) noexcept;

        // Front/top/right orthographic seeds for newly created quad panes.
        [[nodiscard]] lfs::Status seedAxisAlignedOrtho(ViewId id, int axis, bool negative);

    private:
        ViewId allocateId();
        void bumpGeneration();
        [[nodiscard]] static glm::ivec2 clampSize(glm::ivec2 size) noexcept;

        ViewId next_id_ = 1;
        bool id_space_exhausted_ = false;
        std::uint64_t generation_ = 0;
        std::unordered_map<ViewId, std::unique_ptr<ViewRecord>> records_;
    };

} // namespace lfs::vis
