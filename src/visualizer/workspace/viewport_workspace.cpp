/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "workspace/viewport_workspace.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <string>
#include <unordered_set>
#include <utility>

namespace lfs::vis {
    namespace {

        lfs::Error workspaceError(const lfs::ErrorCode code, std::string message,
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

        [[nodiscard]] bool containsId(const std::vector<ViewId>& ids, const ViewId id) {
            return std::find(ids.begin(), ids.end(), id) != ids.end();
        }

        [[nodiscard]] glm::vec2 validFramebufferScale(glm::vec2 scale) noexcept {
            if (!std::isfinite(scale.x) || scale.x < 0.0f)
                scale.x = 1.0f;
            if (!std::isfinite(scale.y) || scale.y < 0.0f)
                scale.y = 1.0f;
            return scale;
        }

        [[nodiscard]] glm::ivec2 scaledExtent(const ViewRect rect, const glm::vec2 scale) noexcept {
            const glm::vec2 safe = validFramebufferScale(scale);
            return {std::max(0, static_cast<int>(std::lround(
                                    static_cast<float>(std::max(rect.width, 0)) * safe.x))),
                    std::max(0, static_cast<int>(std::lround(
                                    static_cast<float>(std::max(rect.height, 0)) * safe.y)))};
        }

    } // namespace

    lfs::Status validateWorkspaceState(const WorkspacePersistentState& state) {
        if (state.format_version != 1) {
            return lfs::Status::failure(workspaceError(
                lfs::ErrorCode::Unsupported, "Unsupported workspace state version",
                LFS_SOURCE_SITE_CURRENT()));
        }

        auto layout = WorkspaceLayout::fromState(state.layout);
        if (!layout) {
            return lfs::Status::failure(std::move(layout).error());
        }

        std::unordered_set<ViewId> view_ids;
        view_ids.reserve(state.views.size());
        ViewId max_view = 0;
        for (const ViewPersistentState& view : state.views) {
            if (auto status = validateViewPersistentState(view); !status)
                return status;
            if (!view_ids.insert(view.id).second) {
                return lfs::Status::failure(workspaceError(
                    lfs::ErrorCode::AlreadyExists, "Workspace state has duplicate view ids",
                    LFS_SOURCE_SITE_CURRENT()));
            }
            max_view = std::max(max_view, view.id);
        }

        const auto leaves = layout->leafIds();
        if (leaves.size() != view_ids.size()) {
            return lfs::Status::failure(workspaceError(
                lfs::ErrorCode::DataLoss, "Workspace layout leaves and view records are different sizes",
                LFS_SOURCE_SITE_CURRENT()));
        }
        for (const ViewId id : leaves) {
            if (!view_ids.contains(id)) {
                return lfs::Status::failure(workspaceError(
                    lfs::ErrorCode::DataLoss, "Workspace layout leaf has no view record",
                    LFS_SOURCE_SITE_CURRENT()));
            }
        }
        for (const ViewId id : view_ids) {
            if (!layout->contains(id)) {
                return lfs::Status::failure(workspaceError(
                    lfs::ErrorCode::DataLoss, "Workspace view record is not a layout leaf",
                    LFS_SOURCE_SITE_CURRENT()));
            }
        }
        if (state.active_viewport) {
            if (!view_ids.contains(*state.active_viewport)) {
                return lfs::Status::failure(workspaceError(
                    lfs::ErrorCode::DataLoss,
                    "Workspace active viewport is not a live view",
                    LFS_SOURCE_SITE_CURRENT()));
            }
            const auto active = std::find_if(
                state.views.begin(), state.views.end(),
                [&state](const ViewPersistentState& view) {
                    return view.id == *state.active_viewport;
                });
            if (active == state.views.end() || active->editor_id != "viewport") {
                return lfs::Status::failure(workspaceError(
                    lfs::ErrorCode::DataLoss,
                    "Workspace active view is not a viewport editor",
                    LFS_SOURCE_SITE_CURRENT()));
            }
        }
        if (leaves.empty()) {
            return lfs::Status::failure(workspaceError(
                lfs::ErrorCode::DataLoss, "Workspace state has no live views",
                LFS_SOURCE_SITE_CURRENT()));
        }
        if (state.next_view_id != kInvalidViewId && state.next_view_id <= max_view) {
            return lfs::Status::failure(workspaceError(
                lfs::ErrorCode::DataLoss, "Workspace next_view_id aliases a live view",
                LFS_SOURCE_SITE_CURRENT()));
        }
        return {};
    }

    ViewportWorkspace::ViewportWorkspace(const glm::ivec2 initial_size)
        : default_size_{std::max(initial_size.x, 0), std::max(initial_size.y, 0)} {
        auto created = views_.create(default_size_);
        const ViewId id = created ? *created : kInvalidViewId;
        layout_ = WorkspaceLayout::single(id);
        if (isValidViewId(id))
            active_viewport_ = id;
    }

    ViewId ViewportWorkspace::primaryView() const noexcept {
        const auto ids = layout_.leafIds();
        for (const ViewId id : ids) {
            if (const auto* record = views_.find(id);
                record != nullptr && record->editor_id == "viewport") {
                return id;
            }
        }
        return ids.empty() ? kInvalidViewId : ids.front();
    }

    std::optional<ViewId> ViewportWorkspace::firstViewport() const noexcept {
        for (const ViewId id : layout_.leafIds()) {
            const auto* record = views_.find(id);
            if (record != nullptr && record->editor_id == "viewport")
                return id;
        }
        return std::nullopt;
    }

    void ViewportWorkspace::repairActiveViewport() noexcept {
        if (active_viewport_) {
            const auto* record = views_.find(*active_viewport_);
            if (record != nullptr && layout_.contains(*active_viewport_) &&
                record->editor_id == "viewport")
                return;
        }
        // A missing persisted field is the legacy format. Prefer its focused
        // viewport when possible, then use layout order as the stable fallback.
        if (const auto focused = layout_.focused()) {
            const auto* record = views_.find(*focused);
            if (record != nullptr && record->editor_id == "viewport") {
                active_viewport_ = *focused;
                return;
            }
        }
        active_viewport_ = firstViewport();
    }

    std::optional<ViewId> ViewportWorkspace::activeViewport() const noexcept {
        if (!active_viewport_)
            return std::nullopt;
        const auto* record = views_.find(*active_viewport_);
        if (record == nullptr || !layout_.contains(*active_viewport_) ||
            record->editor_id != "viewport")
            return std::nullopt;
        return active_viewport_;
    }

    ViewRecord* ViewportWorkspace::findView(const ViewId id) noexcept {
        return views_.find(id);
    }

    const ViewRecord* ViewportWorkspace::findView(const ViewId id) const noexcept {
        return views_.find(id);
    }

    Viewport* ViewportWorkspace::findCamera(const ViewId id) noexcept {
        ViewRecord* record = findView(id);
        return record == nullptr ? nullptr : &record->camera;
    }

    const Viewport* ViewportWorkspace::findCamera(const ViewId id) const noexcept {
        const ViewRecord* record = findView(id);
        return record == nullptr ? nullptr : &record->camera;
    }

    lfs::Status ViewportWorkspace::setViewProjection(const ViewId id,
                                                     const ViewProjectionState& projection) {
        return views_.setProjection(id, projection);
    }

    lfs::Status ViewportWorkspace::setViewDepthWindow(const ViewId id,
                                                      const DepthWindowState& depth) {
        return views_.setDepthWindow(id, depth);
    }

    lfs::Status ViewportWorkspace::setViewGridPlane(const ViewId id, const int grid_plane) {
        return views_.setGridPlane(id, grid_plane);
    }

    lfs::Status ViewportWorkspace::setAreaEditor(const ViewId id, std::string editor_id) {
        const auto* const record = views_.find(id);
        if (record == nullptr || !layout_.contains(id))
            return lfs::Status::failure(workspaceError(
                lfs::ErrorCode::NotFound, "Editor target view does not exist",
                LFS_SOURCE_SITE_CURRENT()));
        if (record->editor_id == editor_id)
            return {};
        const bool becomes_viewport = editor_id == "viewport";
        const bool was_viewport = record->editor_id == "viewport";
        if (auto status = views_.setEditorId(id, std::move(editor_id)); !status)
            return status;
        if (becomes_viewport)
            active_viewport_ = id;
        else if (was_viewport && active_viewport_ == id)
            active_viewport_ = firstViewport();
        else
            repairActiveViewport();
        return {};
    }

    lfs::Status ViewportWorkspace::bumpViewState(const ViewId id) {
        return views_.bumpState(id);
    }

    lfs::Result<ViewId> ViewportWorkspace::split(const ViewId source, const SplitAxis axis,
                                                 const float ratio) {
        if (!layout_.contains(source) || !views_.contains(source)) {
            return workspaceError(lfs::ErrorCode::NotFound, "Split source view does not exist",
                                  LFS_SOURCE_SITE_CURRENT());
        }
        if (axis != SplitAxis::Horizontal && axis != SplitAxis::Vertical) {
            return workspaceError(lfs::ErrorCode::InvalidArgument,
                                  "Split axis is invalid", LFS_SOURCE_SITE_CURRENT());
        }
        if (!std::isfinite(ratio)) {
            return workspaceError(lfs::ErrorCode::InvalidArgument, "Split ratio must be finite",
                                  LFS_SOURCE_SITE_CURRENT());
        }

        auto created = views_.create(default_size_, source);
        if (!created)
            return std::move(created).error();
        const ViewId id = *created;
        if (!layout_.split(source, id, axis, ratio)) {
            static_cast<void>(views_.erase(id));
            return workspaceError(lfs::ErrorCode::InvalidArgument,
                                  "Split failed; layout and registry were left unchanged",
                                  LFS_SOURCE_SITE_CURRENT());
        }
        repairActiveViewport();

        return id;
    }

    lfs::Status ViewportWorkspace::close(const ViewId id) {
        if (!layout_.contains(id)) {
            return lfs::Status::failure(workspaceError(
                lfs::ErrorCode::NotFound, "Close target view does not exist",
                LFS_SOURCE_SITE_CURRENT()));
        }
        if (layout_.leafCount() <= 1) {
            return lfs::Status::failure(workspaceError(
                lfs::ErrorCode::FailedPrecondition, "Closing the last pane is not allowed",
                LFS_SOURCE_SITE_CURRENT()));
        }

        const WorkspaceLayout previous = layout_;
        if (!layout_.close(id)) {
            layout_ = previous;
            return lfs::Status::failure(workspaceError(
                lfs::ErrorCode::FailedPrecondition, "Layout close failed",
                LFS_SOURCE_SITE_CURRENT()));
        }
        if (!views_.erase(id)) {
            layout_ = previous;
            return lfs::Status::failure(workspaceError(
                lfs::ErrorCode::ContractViolation,
                "Registry erase failed after layout close; layout was restored",
                LFS_SOURCE_SITE_CURRENT()));
        }
        if (active_viewport_ == id)
            active_viewport_ = firstViewport();
        else
            repairActiveViewport();
        return {};
    }

    lfs::Status ViewportWorkspace::focus(const ViewId id) {
        if (!layout_.focus(id)) {
            return lfs::Status::failure(workspaceError(
                lfs::ErrorCode::NotFound, "Focus target view does not exist",
                LFS_SOURCE_SITE_CURRENT()));
        }
        if (const auto* record = views_.find(id);
            record != nullptr && record->editor_id == "viewport")
            active_viewport_ = id;
        else
            repairActiveViewport();
        return {};
    }

    lfs::Status ViewportWorkspace::maximize(const ViewId id) {
        if (!layout_.maximize(id)) {
            return lfs::Status::failure(workspaceError(
                lfs::ErrorCode::NotFound, "Maximize target view does not exist",
                LFS_SOURCE_SITE_CURRENT()));
        }
        return {};
    }

    lfs::Status ViewportWorkspace::restoreMaximized() {
        static_cast<void>(layout_.restoreMaximized());
        return {};
    }

    lfs::Status ViewportWorkspace::resize(const LayoutNodeId split, const float ratio) {
        if (!layout_.resize(split, ratio)) {
            return lfs::Status::failure(workspaceError(
                lfs::ErrorCode::InvalidArgument, "Splitter resize failed",
                LFS_SOURCE_SITE_CURRENT()));
        }
        return {};
    }

    lfs::Status ViewportWorkspace::setPreset(const LayoutPreset preset) {
        if (layout_.matchesPreset(preset))
            return {};
        const std::size_t target = presetLeafCount(preset);
        auto keep = layout_.requiredPresetIds(preset);
        if (keep.empty()) {
            return lfs::Status::failure(workspaceError(
                lfs::ErrorCode::FailedPrecondition, "Workspace has no views to arrange",
                LFS_SOURCE_SITE_CURRENT()));
        }
        if (keep.size() > target)
            keep.resize(target);

        const std::size_t additional = target > keep.size() ? target - keep.size() : 0;

        const WorkspaceLayout previous_layout = layout_;
        const auto previous_ids = previous_layout.leafIds();
        std::vector<ViewId> ids = keep;
        std::vector<ViewId> created;
        created.reserve(additional);
        const ViewId clone_source = layout_.focused().value_or(keep.front());
        while (ids.size() < target) {
            auto made = views_.create(default_size_, views_.contains(clone_source)
                                                         ? std::optional<ViewId>{clone_source}
                                                         : std::optional<ViewId>{});
            if (!made) {
                for (const ViewId id : created)
                    static_cast<void>(views_.erase(id));
                return lfs::Status::failure(std::move(made).error());
            }
            created.push_back(*made);
            ids.push_back(*made);
        }

        if (!layout_.applyPreset(preset, ids)) {
            layout_ = previous_layout;
            for (const ViewId id : created)
                static_cast<void>(views_.erase(id));
            return lfs::Status::failure(workspaceError(
                lfs::ErrorCode::Internal, "Preset apply failed; workspace was restored",
                LFS_SOURCE_SITE_CURRENT()));
        }

        for (const ViewId id : previous_ids) {
            if (!containsId(ids, id))
                static_cast<void>(views_.erase(id));
        }

        if (preset == LayoutPreset::Quad)
            seedNewQuadPanes(ids, created);

        if (previous_layout.focused() && containsId(ids, *previous_layout.focused()))
            static_cast<void>(layout_.focus(*previous_layout.focused()));
        repairActiveViewport();
        return {};
    }

    lfs::Status ViewportWorkspace::resetToDefaultThreeAreas() {
        // Keep the active 3D area as the focused survivor.  setPreset() uses
        // the focused leaf when shrinking, which avoids dropping its cached
        // camera even when a panel currently owns UI focus.
        if (const auto active = activeViewport()) {
            if (auto status = focus(*active); !status)
                return status;
        } else if (const auto viewport = firstViewport()) {
            if (auto status = focus(*viewport); !status)
                return status;
        }

        if (auto status = setPreset(LayoutPreset::Single); !status)
            return status;

        const ViewId root = primaryView();
        if (!isValidViewId(root)) {
            return lfs::Status::failure(workspaceError(
                lfs::ErrorCode::FailedPrecondition,
                "Default workspace reset has no surviving root view",
                LFS_SOURCE_SITE_CURRENT()));
        }
        if (auto status = setAreaEditor(root, "viewport"); !status)
            return status;

        const auto scene = split(root, SplitAxis::Horizontal, 0.72f);
        if (!scene)
            return lfs::Status::failure(std::move(scene).error());
        if (auto status = setAreaEditor(*scene, "lfs.scene"); !status)
            return status;

        const auto rendering = split(*scene, SplitAxis::Vertical, 0.40f);
        if (!rendering)
            return lfs::Status::failure(std::move(rendering).error());
        if (auto status = setAreaEditor(*rendering, "lfs.rendering"); !status)
            return status;
        return focus(root);
    }

    WorkspaceFrameSnapshot ViewportWorkspace::snapshot(const ViewRect outer,
                                                       const int min_pane_pixels,
                                                       const int divider_pixels,
                                                       const glm::vec2 framebuffer_scale,
                                                       const int header_pixels) const {
        const auto layout_snap = layout_.solve(outer, min_pane_pixels, divider_pixels);
        WorkspaceFrameSnapshot frame;
        frame.outer = layout_snap.outer;
        frame.splitters = layout_snap.splitters;
        frame.focused = layout_snap.focused;
        frame.active_viewport = activeViewport();
        frame.maximized = layout_snap.maximized;
        frame.primary = primaryView();
        frame.layout_generation = layout_snap.generation;
        frame.registry_generation = views_.generation();
        frame.live_ids = views_.ids();
        frame.live_viewport_ids.reserve(frame.live_ids.size());
        for (const ViewId id : frame.live_ids) {
            if (const auto* record = views_.find(id);
                record != nullptr && record->editor_id == "viewport") {
                frame.live_viewport_ids.push_back(id);
            }
        }
        const int header = std::max(header_pixels, 0);
        frame.areas.reserve(layout_snap.panes.size());
        frame.panes.reserve(layout_snap.panes.size());
        for (const LayoutPane& pane : layout_snap.panes) {
            const ViewRecord* record = views_.find(pane.id);
            const ViewRect content{
                pane.rect.x,
                pane.rect.y + std::min(header, std::max(pane.rect.height, 0)),
                pane.rect.width,
                std::max(pane.rect.height - header, 0)};
            frame.areas.push_back(AreaSnapshot{
                .id = pane.id,
                .leaf = pane.leaf,
                .rect = pane.rect,
                .content_rect = content,
                .editor_id = record ? record->editor_id : "viewport",
                .focused = frame.focused && *frame.focused == pane.id,
                .maximized = frame.maximized && *frame.maximized == pane.id,
            });
            if (record && record->editor_id != "viewport")
                continue;
            LayoutPane content_pane = pane;
            content_pane.rect = content;
            const bool focused = frame.focused && *frame.focused == pane.id;
            const bool maximized = frame.maximized && *frame.maximized == pane.id;
            frame.panes.push_back(copyPane(content_pane, focused, maximized, framebuffer_scale));
        }
        return frame;
    }

    void ViewportWorkspace::syncExtents(const ViewRect outer, const int min_pane_pixels,
                                        const int divider_pixels,
                                        const glm::vec2 framebuffer_scale,
                                        const int header_pixels) {
        const auto layout_snap = layout_.solve(outer, min_pane_pixels, divider_pixels);
        const int header = std::max(header_pixels, 0);
        for (const LayoutPane& pane : layout_snap.panes) {
            const glm::ivec2 logical_size{std::max(pane.rect.width, 0),
                                          std::max(pane.rect.height - header, 0)};
            const ViewRect content{
                pane.rect.x,
                pane.rect.y + std::min(header, std::max(pane.rect.height, 0)),
                pane.rect.width,
                std::max(pane.rect.height - header, 0)};
            static_cast<void>(views_.setLogicalSize(pane.id, logical_size));
            static_cast<void>(views_.setFramebufferSize(pane.id,
                                                        scaledExtent(content, framebuffer_scale)));
        }
    }

    WorkspacePersistentState ViewportWorkspace::exportState() const {
        WorkspacePersistentState state;
        state.format_version = 1;
        state.layout = layout_.exportState();
        state.views = views_.exportStates();
        state.active_viewport = activeViewport();
        state.next_view_id = views_.nextId();
        state.registry_generation = views_.generation();
        return state;
    }

    lfs::Status ViewportWorkspace::importState(const WorkspacePersistentState& state) {
        if (auto status = validateWorkspaceState(state); !status)
            return status;

        const auto next_registry_generation =
            views_.generation() == std::numeric_limits<std::uint64_t>::max()
                ? views_.generation()
                : views_.generation() + 1;
        const auto next_layout_generation =
            layout_.generation() == std::numeric_limits<std::uint64_t>::max()
                ? layout_.generation()
                : layout_.generation() + 1;

        auto layout = WorkspaceLayout::fromState(state.layout);
        if (!layout)
            return lfs::Status::failure(std::move(layout).error());

        ViewRegistry registry;
        registry.bumpNextId(std::max(views_.nextId(), state.next_view_id));
        layout->bumpNextNodeId(std::max(layout_.nextNodeId(), state.layout.next_node_id));
        for (const ViewPersistentState& view : state.views) {
            if (auto status = registry.adopt(view); !status)
                return status;
        }
        registry.bumpGenerationAtLeast(std::max(next_registry_generation,
                                                state.registry_generation));
        layout->bumpGenerationAtLeast(std::max(next_layout_generation,
                                               state.layout.generation));
        registry.bumpNextId(std::max(views_.nextId(), std::max(state.next_view_id, registry.nextId())));

        layout_ = std::move(*layout);
        views_ = std::move(registry);
        active_viewport_ = state.active_viewport;
        repairActiveViewport();
        return {};
    }

    void ViewportWorkspace::seedNewQuadPanes(const std::vector<ViewId>& ids,
                                             const std::vector<ViewId>& created) {
        if (ids.size() != 4)
            return;
        const std::unordered_set<ViewId> is_new(created.begin(), created.end());
        // Slot 0 keeps the cloned/existing camera. Newly created slots seed
        // front (Z), top (Y), and right (X) orthographic views.
        constexpr int kFrontAxis = 2;
        constexpr int kTopAxis = 1;
        constexpr int kRightAxis = 0;
        if (is_new.contains(ids[1]))
            static_cast<void>(views_.seedAxisAlignedOrtho(ids[1], kFrontAxis, false));
        if (is_new.contains(ids[2]))
            static_cast<void>(views_.seedAxisAlignedOrtho(ids[2], kTopAxis, false));
        if (is_new.contains(ids[3]))
            static_cast<void>(views_.seedAxisAlignedOrtho(ids[3], kRightAxis, false));
    }

    PaneSnapshot ViewportWorkspace::copyPane(const LayoutPane& pane, const bool focused,
                                             const bool maximized,
                                             const glm::vec2 framebuffer_scale) const {
        PaneSnapshot snap;
        snap.id = pane.id;
        snap.leaf = pane.leaf;
        snap.rect = pane.rect;
        snap.window_size = {std::max(pane.rect.width, 0), std::max(pane.rect.height, 0)};
        // Logical dimensions always come from the current solve. The
        // framebuffer dimensions are an explicit caller-provided scale, so a
        // snapshot cannot inherit stale extents from a prior syncExtents call.
        snap.framebuffer_size = scaledExtent(pane.rect, framebuffer_scale);
        snap.focused = focused;
        snap.maximized = maximized;
        const ViewRecord* record = views_.find(pane.id);
        if (record == nullptr)
            return snap;
        snap.projection = record->projection;
        snap.depth = record->depth;
        snap.grid_plane = record->grid_plane;
        snap.rotation = record->camera.camera.R;
        snap.translation = record->camera.camera.t;
        snap.pivot = record->camera.camera.pivot;
        snap.state_generation = record->state_generation;
        if (const auto& binding = record->camera_binding;
            binding && binding->valid() && binding->rotation == snap.rotation &&
            binding->translation == snap.translation && binding->pivot == snap.pivot &&
            binding->projection == snap.projection) {
            snap.camera_uid = binding->uid;
            snap.camera_binding_generation = binding->camera_list_generation;
        }
        return snap;
    }

} // namespace lfs::vis
