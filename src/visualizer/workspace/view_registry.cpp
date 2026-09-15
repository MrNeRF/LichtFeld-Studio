/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "workspace/view_registry.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <string>
#include <utility>

namespace lfs::vis {
    namespace {

        lfs::Error registryError(const lfs::ErrorCode code, std::string message,
                                 const lfs::core::SourceSite site) {
            return lfs::make_error(lfs::ErrorInit{
                .code = code,
                .domain = lfs::ErrorDomain::App,
                .severity = lfs::Severity::Error,
                .retryability = lfs::Retryability::NotRetryable,
                .user_message = message,
                .detail = message,
                .detection = site,
            });
        }

        [[nodiscard]] bool finiteVec(const glm::vec3& v) noexcept {
            return std::isfinite(v.x) && std::isfinite(v.y) && std::isfinite(v.z);
        }

        [[nodiscard]] bool finiteMat(const glm::mat3& m) noexcept {
            for (int col = 0; col < 3; ++col) {
                if (!finiteVec(m[col]))
                    return false;
            }
            return true;
        }

        [[nodiscard]] bool validRotation(const glm::mat3& rotation) noexcept {
            if (!finiteMat(rotation))
                return false;
            constexpr float tolerance = 1e-3f;
            const glm::vec3& x = rotation[0];
            const glm::vec3& y = rotation[1];
            const glm::vec3& z = rotation[2];
            const auto unit = [tolerance](const glm::vec3& axis) {
                return std::abs(glm::dot(axis, axis) - 1.0f) <= tolerance;
            };
            const auto orthogonal = [tolerance](const glm::vec3& a, const glm::vec3& b) {
                return std::abs(glm::dot(a, b)) <= tolerance;
            };
            const float determinant = glm::dot(glm::cross(x, y), z);
            return unit(x) && unit(y) && unit(z) && orthogonal(x, y) &&
                   orthogonal(x, z) && orthogonal(y, z) &&
                   determinant > 1.0f - tolerance && determinant < 1.0f + tolerance;
        }

        [[nodiscard]] bool finitePositive(const float value) noexcept {
            return std::isfinite(value) && value > 0.0f;
        }

        [[nodiscard]] bool reasonableEditorId(const std::string& id) noexcept {
            constexpr std::size_t kMaxEditorIdBytes = 64;
            return !id.empty() && id.size() <= kMaxEditorIdBytes &&
                   std::all_of(id.begin(), id.end(), [](const unsigned char value) {
                       return value >= 0x21 && value <= 0x7e;
                   });
        }

        [[nodiscard]] bool reasonableChrome(const std::unordered_map<std::string, std::string>& chrome) noexcept {
            constexpr std::size_t kMaxEntries = 64;
            constexpr std::size_t kMaxKeyBytes = 128;
            constexpr std::size_t kMaxValueBytes = 64 * 1024;
            if (chrome.size() > kMaxEntries)
                return false;
            return std::all_of(chrome.begin(), chrome.end(), [](const auto& entry) {
                return !entry.first.empty() && entry.first.size() <= kMaxKeyBytes &&
                       entry.second.size() <= kMaxValueBytes;
            });
        }

        [[nodiscard]] lfs::Status validateProjection(const ViewProjectionState& projection,
                                                     const lfs::core::SourceSite site) {
            if (!finitePositive(projection.focal_length_mm) ||
                !finitePositive(projection.ortho_scale) ||
                !std::isfinite(projection.near_plane) || !std::isfinite(projection.far_plane) ||
                !(projection.near_plane < projection.far_plane)) {
                return lfs::Status::failure(registryError(
                    lfs::ErrorCode::InvalidArgument, "View projection values are invalid", site));
            }
            return {};
        }

        [[nodiscard]] lfs::Status validateDepth(const DepthWindowState& depth,
                                                const lfs::core::SourceSite site) {
            if (!std::isfinite(depth.near_plane) || !std::isfinite(depth.far_plane) ||
                !std::isfinite(depth.scale_x) || !std::isfinite(depth.scale_y) ||
                !std::isfinite(depth.offset_x) || !std::isfinite(depth.offset_y) ||
                !(depth.near_plane < depth.far_plane)) {
                return lfs::Status::failure(registryError(
                    lfs::ErrorCode::InvalidArgument, "View depth window values are invalid", site));
            }
            return {};
        }

    } // namespace

    void resetTransientNavigation(Viewport& viewport) {
        viewport.camera.prePos = glm::vec2(0.0f);
        viewport.camera.isOrbiting = false;
        viewport.camera.resetRollTarget();
        viewport.camera.clearTransientMotion();
    }

    void initializeViewportSizes(Viewport& viewport, const glm::ivec2 logical,
                                 glm::ivec2 framebuffer) {
        const glm::ivec2 window{std::max(logical.x, 0), std::max(logical.y, 0)};
        if (framebuffer.x <= 0 && framebuffer.y <= 0)
            framebuffer = window;
        else {
            framebuffer.x = std::max(framebuffer.x, 0);
            framebuffer.y = std::max(framebuffer.y, 0);
        }
        viewport.windowSize = window;
        viewport.frameBufferSize = framebuffer;
    }

    void clonePersistentViewport(const Viewport& source, Viewport& destination) {
        destination.camera.R = source.camera.R;
        destination.camera.t = source.camera.t;
        destination.camera.pivot = source.camera.pivot;
        destination.camera.home_R = source.camera.home_R;
        destination.camera.home_t = source.camera.home_t;
        destination.camera.home_pivot = source.camera.home_pivot;
        destination.camera.home_saved = source.camera.home_saved;
        destination.camera.zoomSpeed = source.camera.zoomSpeed;
        destination.camera.maxZoomSpeed = source.camera.maxZoomSpeed;
        destination.camera.rotateSpeed = source.camera.rotateSpeed;
        destination.camera.rotateCenterSpeed = source.camera.rotateCenterSpeed;
        destination.camera.rotateRollSpeed = source.camera.rotateRollSpeed;
        destination.camera.translateSpeed = source.camera.translateSpeed;
        destination.camera.wasdSpeed = source.camera.wasdSpeed;
        destination.camera.maxWasdSpeed = source.camera.maxWasdSpeed;
        destination.ortho_scale_override = source.ortho_scale_override;
        resetTransientNavigation(destination);
    }

    ViewPersistentState captureViewPersistentState(const ViewRecord& record) {
        ViewPersistentState state;
        state.id = record.id;
        const auto& camera = record.camera.camera;
        state.rotation = camera.R;
        state.translation = camera.t;
        state.pivot = camera.pivot;
        state.home_rotation = camera.home_R;
        state.home_translation = camera.home_t;
        state.home_pivot = camera.home_pivot;
        state.home_saved = camera.home_saved;
        state.zoom_speed = camera.zoomSpeed;
        state.max_zoom_speed = camera.maxZoomSpeed;
        state.rotate_speed = camera.rotateSpeed;
        state.rotate_center_speed = camera.rotateCenterSpeed;
        state.rotate_roll_speed = camera.rotateRollSpeed;
        state.translate_speed = camera.translateSpeed;
        state.wasd_speed = camera.wasdSpeed;
        state.max_wasd_speed = camera.maxWasdSpeed;
        state.ortho_scale_override = record.camera.ortho_scale_override;
        state.editor_id = record.editor_id;
        state.chrome = record.chrome;
        state.projection = record.projection;
        state.depth = record.depth;
        state.grid_plane = record.grid_plane;
        state.window_size = record.camera.windowSize;
        state.framebuffer_size = record.camera.frameBufferSize;
        state.state_generation = record.state_generation;
        return state;
    }

    void applyViewPersistentState(ViewRecord& record, const ViewPersistentState& state) {
        // Dataset-camera bindings are transient and must never survive a
        // project/scene restore with a possibly different camera registry.
        record.camera_binding.reset();
        record.id = state.id;
        initializeViewportSizes(record.camera, state.window_size, state.framebuffer_size);
        record.camera.camera.R = state.rotation;
        record.camera.camera.t = state.translation;
        record.camera.camera.pivot = state.pivot;
        record.camera.camera.home_R = state.home_rotation;
        record.camera.camera.home_t = state.home_translation;
        record.camera.camera.home_pivot = state.home_pivot;
        record.camera.camera.home_saved = state.home_saved;
        record.camera.camera.zoomSpeed = state.zoom_speed;
        record.camera.camera.maxZoomSpeed = state.max_zoom_speed;
        record.camera.camera.rotateSpeed = state.rotate_speed;
        record.camera.camera.rotateCenterSpeed = state.rotate_center_speed;
        record.camera.camera.rotateRollSpeed = state.rotate_roll_speed;
        record.camera.camera.translateSpeed = state.translate_speed;
        record.camera.camera.wasdSpeed = state.wasd_speed;
        record.camera.camera.maxWasdSpeed = state.max_wasd_speed;
        record.camera.ortho_scale_override = state.ortho_scale_override;
        record.editor_id = state.editor_id;
        record.chrome = state.chrome;
        record.projection = state.projection;
        record.depth = state.depth;
        record.grid_plane = state.grid_plane;
        record.state_generation = state.state_generation == 0 ? 1 : state.state_generation;
        resetTransientNavigation(record.camera);
    }

    lfs::Status validateViewPersistentState(const ViewPersistentState& state) {
        if (!isValidViewId(state.id)) {
            return lfs::Status::failure(registryError(
                lfs::ErrorCode::InvalidArgument, "View id is invalid", LFS_SOURCE_SITE_CURRENT()));
        }
        if (!reasonableEditorId(state.editor_id) || !reasonableChrome(state.chrome)) {
            return lfs::Status::failure(registryError(
                lfs::ErrorCode::InvalidArgument, "View editor metadata is invalid",
                LFS_SOURCE_SITE_CURRENT()));
        }
        if (!validRotation(state.rotation) || !validRotation(state.home_rotation) ||
            !finiteVec(state.translation) || !finiteVec(state.pivot) ||
            !finiteVec(state.home_translation) || !finiteVec(state.home_pivot)) {
            return lfs::Status::failure(registryError(
                lfs::ErrorCode::InvalidArgument, "View camera values are not finite",
                LFS_SOURCE_SITE_CURRENT()));
        }
        if (!std::isfinite(state.zoom_speed) || !std::isfinite(state.max_zoom_speed) ||
            !std::isfinite(state.rotate_speed) || !std::isfinite(state.rotate_center_speed) ||
            !std::isfinite(state.rotate_roll_speed) || !std::isfinite(state.translate_speed) ||
            !std::isfinite(state.wasd_speed) || !std::isfinite(state.max_wasd_speed)) {
            return lfs::Status::failure(registryError(
                lfs::ErrorCode::InvalidArgument, "View navigation speeds are not finite",
                LFS_SOURCE_SITE_CURRENT()));
        }
        if (state.ortho_scale_override &&
            (!std::isfinite(*state.ortho_scale_override) || *state.ortho_scale_override <= 0.0f)) {
            return lfs::Status::failure(registryError(
                lfs::ErrorCode::InvalidArgument, "View ortho scale override is invalid",
                LFS_SOURCE_SITE_CURRENT()));
        }
        if (auto status = validateProjection(state.projection, LFS_SOURCE_SITE_CURRENT()); !status)
            return status;
        if (auto status = validateDepth(state.depth, LFS_SOURCE_SITE_CURRENT()); !status)
            return status;
        if (state.window_size.x < 0 || state.window_size.y < 0 ||
            state.framebuffer_size.x < 0 || state.framebuffer_size.y < 0) {
            return lfs::Status::failure(registryError(
                lfs::ErrorCode::InvalidArgument, "View sizes must be non-negative",
                LFS_SOURCE_SITE_CURRENT()));
        }
        return {};
    }

    lfs::Result<ViewId> ViewRegistry::create(const glm::ivec2 logical_size,
                                             const std::optional<ViewId> clone_from) {
        const ViewRecord* source = nullptr;
        if (clone_from) {
            source = find(*clone_from);
            if (source == nullptr) {
                return registryError(lfs::ErrorCode::NotFound, "Clone source view does not exist",
                                     LFS_SOURCE_SITE_CURRENT());
            }
        }

        const ViewId id = allocateId();
        if (!isValidViewId(id)) {
            return registryError(lfs::ErrorCode::ResourceExhausted,
                                 "View id space is exhausted", LFS_SOURCE_SITE_CURRENT());
        }
        auto record = std::make_unique<ViewRecord>();
        record->id = id;
        const glm::ivec2 size = clampSize(logical_size);
        initializeViewportSizes(record->camera, size, size);
        resetTransientNavigation(record->camera);

        if (source != nullptr) {
            clonePersistentViewport(source->camera, record->camera);
            record->editor_id = source->editor_id;
            record->chrome = source->chrome;
            record->projection = source->projection;
            record->depth = source->depth;
            record->grid_plane = source->grid_plane;
            record->camera_binding = source->camera_binding;
            initializeViewportSizes(record->camera, size, size);
            resetTransientNavigation(record->camera);
        }

        record->state_generation = 1;
        records_.emplace(id, std::move(record));
        bumpGeneration();
        return id;
    }

    bool ViewRegistry::contains(const ViewId id) const noexcept {
        return isValidViewId(id) && records_.contains(id);
    }

    ViewRecord* ViewRegistry::find(const ViewId id) noexcept {
        const auto it = records_.find(id);
        return it == records_.end() ? nullptr : it->second.get();
    }

    const ViewRecord* ViewRegistry::find(const ViewId id) const noexcept {
        const auto it = records_.find(id);
        return it == records_.end() ? nullptr : it->second.get();
    }

    bool ViewRegistry::erase(const ViewId id) {
        if (records_.erase(id) == 0)
            return false;
        bumpGeneration();
        return true;
    }

    std::vector<ViewId> ViewRegistry::ids() const {
        std::vector<ViewId> out;
        out.reserve(records_.size());
        for (const auto& [id, _] : records_)
            out.push_back(id);
        std::sort(out.begin(), out.end());
        return out;
    }

    lfs::Status ViewRegistry::setEditorId(const ViewId id, std::string editor_id) {
        if (!contains(id))
            return lfs::Status::failure(registryError(
                lfs::ErrorCode::NotFound, "Editor target view does not exist",
                LFS_SOURCE_SITE_CURRENT()));
        if (!reasonableEditorId(editor_id))
            return lfs::Status::failure(registryError(
                lfs::ErrorCode::InvalidArgument, "Editor id is invalid",
                LFS_SOURCE_SITE_CURRENT()));
        auto* record = find(id);
        if (record->editor_id == editor_id)
            return {};
        record->editor_id = std::move(editor_id);
        resetTransientNavigation(record->camera);
        ++record->state_generation;
        bumpGeneration();
        return {};
    }

    lfs::Status ViewRegistry::setLogicalSize(const ViewId id, const glm::ivec2 size) {
        ViewRecord* record = find(id);
        if (record == nullptr) {
            return lfs::Status::failure(registryError(
                lfs::ErrorCode::NotFound, "View does not exist", LFS_SOURCE_SITE_CURRENT()));
        }
        if (record->camera.windowSize == clampSize(size))
            return {};
        record->camera.windowSize = clampSize(size);
        ++record->state_generation;
        bumpGeneration();
        return {};
    }

    lfs::Status ViewRegistry::setFramebufferSize(const ViewId id, const glm::ivec2 size) {
        ViewRecord* record = find(id);
        if (record == nullptr) {
            return lfs::Status::failure(registryError(
                lfs::ErrorCode::NotFound, "View does not exist", LFS_SOURCE_SITE_CURRENT()));
        }
        if (record->camera.frameBufferSize == clampSize(size))
            return {};
        record->camera.frameBufferSize = clampSize(size);
        ++record->state_generation;
        bumpGeneration();
        return {};
    }

    lfs::Status ViewRegistry::setProjection(const ViewId id, const ViewProjectionState& projection) {
        if (auto status = validateProjection(projection, LFS_SOURCE_SITE_CURRENT()); !status)
            return status;
        ViewRecord* record = find(id);
        if (record == nullptr) {
            return lfs::Status::failure(registryError(
                lfs::ErrorCode::NotFound, "View does not exist", LFS_SOURCE_SITE_CURRENT()));
        }
        if (record->projection == projection)
            return {};
        record->projection = projection;
        ++record->state_generation;
        bumpGeneration();
        return {};
    }

    lfs::Status ViewRegistry::setDepthWindow(const ViewId id, const DepthWindowState& depth) {
        if (auto status = validateDepth(depth, LFS_SOURCE_SITE_CURRENT()); !status)
            return status;
        ViewRecord* record = find(id);
        if (record == nullptr) {
            return lfs::Status::failure(registryError(
                lfs::ErrorCode::NotFound, "View does not exist", LFS_SOURCE_SITE_CURRENT()));
        }
        if (record->depth == depth)
            return {};
        record->depth = depth;
        ++record->state_generation;
        bumpGeneration();
        return {};
    }

    lfs::Status ViewRegistry::setGridPlane(const ViewId id, const int grid_plane) {
        ViewRecord* record = find(id);
        if (record == nullptr) {
            return lfs::Status::failure(registryError(
                lfs::ErrorCode::NotFound, "View does not exist", LFS_SOURCE_SITE_CURRENT()));
        }
        if (record->grid_plane == grid_plane)
            return {};
        record->grid_plane = grid_plane;
        ++record->state_generation;
        bumpGeneration();
        return {};
    }

    lfs::Status ViewRegistry::bumpState(const ViewId id) {
        ViewRecord* record = find(id);
        if (record == nullptr) {
            return lfs::Status::failure(registryError(
                lfs::ErrorCode::NotFound, "View does not exist", LFS_SOURCE_SITE_CURRENT()));
        }
        ++record->state_generation;
        bumpGeneration();
        return {};
    }

    std::vector<ViewPersistentState> ViewRegistry::exportStates() const {
        std::vector<ViewPersistentState> out;
        out.reserve(records_.size());
        for (const ViewId id : ids())
            out.push_back(captureViewPersistentState(*records_.at(id)));
        return out;
    }

    lfs::Status ViewRegistry::adopt(ViewPersistentState state) {
        if (auto status = validateViewPersistentState(state); !status)
            return status;
        if (records_.contains(state.id)) {
            return lfs::Status::failure(registryError(
                lfs::ErrorCode::AlreadyExists, "View id is already live", LFS_SOURCE_SITE_CURRENT()));
        }
        auto record = std::make_unique<ViewRecord>();
        applyViewPersistentState(*record, state);
        if (state.id == std::numeric_limits<ViewId>::max()) {
            bumpNextId(state.id);
            id_space_exhausted_ = true;
        } else {
            bumpNextId(state.id + 1);
        }
        records_.emplace(state.id, std::move(record));
        bumpGeneration();
        return {};
    }

    void ViewRegistry::bumpNextId(const ViewId at_least) noexcept {
        if (at_least > next_id_)
            next_id_ = at_least;
        if (next_id_ == kInvalidViewId)
            next_id_ = 1;
        if (next_id_ == std::numeric_limits<ViewId>::max() &&
            records_.contains(next_id_)) {
            id_space_exhausted_ = true;
        }
    }

    lfs::Status ViewRegistry::seedAxisAlignedOrtho(const ViewId id, const int axis,
                                                   const bool negative) {
        ViewRecord* record = find(id);
        if (record == nullptr) {
            return lfs::Status::failure(registryError(
                lfs::ErrorCode::NotFound, "View does not exist", LFS_SOURCE_SITE_CURRENT()));
        }
        record->camera.camera.setAxisAlignedView(axis, negative);
        record->projection.orthographic = true;
        record->projection.equirectangular = false;
        record->grid_plane = axis;
        resetTransientNavigation(record->camera);
        ++record->state_generation;
        bumpGeneration();
        return {};
    }

    ViewId ViewRegistry::allocateId() {
        if (id_space_exhausted_)
            return kInvalidViewId;
        if (next_id_ == kInvalidViewId)
            next_id_ = 1;
        while (records_.contains(next_id_)) {
            if (next_id_ == std::numeric_limits<ViewId>::max()) {
                id_space_exhausted_ = true;
                return kInvalidViewId;
            }
            ++next_id_;
        }
        const ViewId id = next_id_;
        if (next_id_ != std::numeric_limits<ViewId>::max())
            ++next_id_;
        else
            id_space_exhausted_ = true;
        return id;
    }

    void ViewRegistry::bumpGeneration() { ++generation_; }

    void ViewRegistry::bumpGenerationAtLeast(const std::uint64_t at_least) noexcept {
        if (generation_ < at_least)
            generation_ = at_least;
    }

    glm::ivec2 ViewRegistry::clampSize(const glm::ivec2 size) noexcept {
        return {std::max(size.x, 0), std::max(size.y, 0)};
    }

} // namespace lfs::vis
