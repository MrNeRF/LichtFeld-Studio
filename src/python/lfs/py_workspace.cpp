/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "py_workspace.hpp"
#include "gui/gui_manager.hpp"
#include "gui/panel_registry.hpp"
#include "python/python_runtime.hpp"
#include "rendering/rendering_manager.hpp"
#include "visualizer/post_work_utils.hpp"
#include "workspace/viewport_workspace.hpp"

#include <array>
#include <cmath>
#include <functional>
#include <nanobind/stl/array.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace nb = nanobind;

namespace lfs::python {
    namespace {
        template <typename F>
        auto onWorkspaceThread(F&& fn) {
            using Result = std::invoke_result_t<F>;
            auto* viewer = get_visualizer();
            if (!viewer || !viewer->acceptsPostedWork())
                throw std::runtime_error("Viewport workspace is unavailable");
            if (viewer->isOnViewerThread())
                return std::invoke(std::forward<F>(fn));
            const nb::gil_scoped_release release;
            return vis::post_work_and_wait(
                [viewer](vis::Visualizer::WorkItem work) { return viewer->postWork(std::move(work)); },
                std::forward<F>(fn),
                []() -> Result { throw std::runtime_error("Viewport workspace operation was cancelled"); });
        }

        vis::ViewportWorkspace& workspace() {
            auto* viewer = get_visualizer();
            auto* value = viewer ? viewer->getViewportWorkspace() : nullptr;
            if (!value)
                throw std::runtime_error("Viewport workspace is unavailable");
            return *value;
        }

        template <typename T>
        void requireSuccess(const T& result) {
            if (!result)
                throw std::invalid_argument(std::string(result.error().user_message()));
        }

        void requestFrame() {
            if (auto* renderer = get_rendering_manager())
                renderer->markDirty(vis::DirtyFlag::CAMERA | vis::DirtyFlag::VIEWPORT);
        }

        nb::tuple vec3(const glm::vec3& value) {
            return nb::make_tuple(value.x, value.y, value.z);
        }

        vis::ViewRecord& viewRecord(const vis::ViewId id) {
            auto* record = workspace().findView(id);
            if (!record)
                throw std::invalid_argument("Unknown viewport view ID");
            return *record;
        }
    } // namespace

    void register_workspace(nb::module_& m) {
        m.def("workspace_state", []() {
            // Copy native values while on the viewer thread. Python objects are
            // created afterward on the calling thread with the GIL held.
            const auto state = onWorkspaceThread([] {
                if (const auto* gui = get_gui_manager())
                    return std::pair(gui->workspaceSnapshot(), workspace().exportState());
                glm::vec2 pos(0.0f), size(1280.0f, 720.0f);
                const vis::ViewRect outer{static_cast<int>(std::lround(pos.x)),
                                          static_cast<int>(std::lround(pos.y)),
                                          static_cast<int>(std::lround(size.x)),
                                          static_cast<int>(std::lround(size.y))};
                return std::pair(workspace().snapshot(outer), workspace().exportState());
            });
            const auto& [snapshot, persistent] = state;
            nb::dict result;
            result["focused"] = nb::cast(snapshot.focused);
            result["active_viewport"] = nb::cast(snapshot.active_viewport);
            result["maximized"] = nb::cast(snapshot.maximized);
            result["primary"] = snapshot.primary;
            result["layout_generation"] = snapshot.layout_generation;
            nb::list views;
            for (const auto& record : persistent.views) {
                nb::dict value;
                value["id"] = record.id;
                value["editor_id"] = record.editor_id;
                value["eye"] = vec3(record.translation);
                value["target"] = vec3(record.pivot);
                value["up"] = vec3(record.rotation[1]);
                value["orthographic"] = record.projection.orthographic;
                value["focal_length_mm"] = record.projection.focal_length_mm;
                value["ortho_scale"] = record.projection.ortho_scale;
                value["near_plane"] = record.projection.near_plane;
                value["far_plane"] = record.projection.far_plane;
                value["grid_plane"] = record.grid_plane;
                value["visible"] = false;
                for (const auto& area : snapshot.areas) {
                    if (area.id != record.id)
                        continue;
                    value["visible"] = true;
                    value["area_rect"] = nb::make_tuple(area.rect.x, area.rect.y, area.rect.width, area.rect.height);
                    value["rect"] = nb::make_tuple(area.content_rect.x, area.content_rect.y,
                                                   area.content_rect.width, area.content_rect.height);
                    break;
                }
                for (const auto& pane : snapshot.panes) {
                    if (pane.id != record.id)
                        continue;
                    value["visible"] = true;
                    value["rect"] = nb::make_tuple(pane.rect.x, pane.rect.y, pane.rect.width, pane.rect.height);
                    break;
                }
                views.append(value);
            }
            result["views"] = views;
            nb::list splits;
            for (const auto& node : persistent.layout.nodes) {
                if (node.kind != vis::LayoutNodeKind::Split)
                    continue;
                nb::dict value;
                value["id"] = node.id;
                value["axis"] = node.axis == vis::SplitAxis::Horizontal ? "horizontal" : "vertical";
                value["ratio"] = node.ratio;
                splits.append(value);
            }
            result["splitters"] = splits;
            return result; }, "Inspect stable view IDs, cameras and workspace layout.");

        m.def("workspace_set_editor", [](vis::ViewId id, const std::string& editor_id) { onWorkspaceThread([id, editor_id] {
                                                                                             if (editor_id != "viewport") {
                                                                                                 const auto prototype = vis::gui::PanelRegistry::instance().get_panel_instance(editor_id);
                                                                                                 if (!prototype || !prototype->supportsAreaInstances())
                                                                                                     throw std::invalid_argument("Unknown area editor: " + editor_id);
                                                                                             }
                                                                                             requireSuccess(workspace().setAreaEditor(id, editor_id));
                                                                                             requestFrame();
                                                                                         }); }, nb::arg("view_id"), nb::arg("editor_id"), "Choose an area's editor while retaining its identity and camera state.");

        m.def("workspace_set_layout", [](const std::string& layout) { onWorkspaceThread([layout] {
                                                                          vis::LayoutPreset preset;
                                                                          if (layout == "single")
                                                                              preset = vis::LayoutPreset::Single;
                                                                          else if (layout == "horizontal")
                                                                              preset = vis::LayoutPreset::DualHorizontal;
                                                                          else if (layout == "vertical")
                                                                              preset = vis::LayoutPreset::DualVertical;
                                                                          else if (layout == "quad")
                                                                              preset = vis::LayoutPreset::Quad;
                                                                          else
                                                                              throw std::invalid_argument("Layout must be single, horizontal, vertical or quad");
                                                                          requireSuccess(workspace().setPreset(preset));
                                                                          requestFrame();
                                                                      }); }, nb::arg("layout"), "Set the workspace layout while retaining surviving view identities.");

        m.def("workspace_split", [](const vis::ViewId id, const std::string& axis, const float ratio) { return onWorkspaceThread([id, axis, ratio] {
                                                                                                            if (axis != "horizontal" && axis != "vertical")
                                                                                                                throw std::invalid_argument("Split axis must be horizontal or vertical");
                                                                                                            const auto result = workspace().split(id, axis == "horizontal" ? vis::SplitAxis::Horizontal : vis::SplitAxis::Vertical, ratio);
                                                                                                            requireSuccess(result);
                                                                                                            requestFrame();
                                                                                                            return *result;
                                                                                                        }); }, nb::arg("view_id"), nb::arg("axis") = "horizontal", nb::arg("ratio") = 0.5f);

        m.def("workspace_close", [](const vis::ViewId id) { onWorkspaceThread([id] { requireSuccess(workspace().close(id)); requestFrame(); }); }, nb::arg("view_id"));
        m.def("workspace_focus", [](const vis::ViewId id) { onWorkspaceThread([id] { requireSuccess(workspace().focus(id)); requestFrame(); }); }, nb::arg("view_id"));
        m.def("workspace_maximize", [](const std::optional<vis::ViewId> id) { onWorkspaceThread([id] {
                                                                                  requireSuccess(id ? workspace().maximize(*id) : workspace().restoreMaximized());
                                                                                  requestFrame();
                                                                              }); }, nb::arg("view_id") = nb::none(), "Maximize a view, or restore the layout with None.");
        m.def("workspace_resize", [](const vis::LayoutNodeId split, const float ratio) { onWorkspaceThread([split, ratio] { requireSuccess(workspace().resize(split, ratio)); requestFrame(); }); }, nb::arg("split_id"), nb::arg("ratio"));

        m.def("workspace_set_camera", [](const vis::ViewId id, const std::array<float, 3>& eye, const std::array<float, 3>& target, const std::array<float, 3>& up) { onWorkspaceThread([id, eye, target, up] {
                                                                                                                                                                          const glm::vec3 position(eye[0], eye[1], eye[2]);
                                                                                                                                                                          const glm::vec3 pivot(target[0], target[1], target[2]);
                                                                                                                                                                          const auto rotation = lfs::rendering::tryMakeVisualizerLookAtRotation(
                                                                                                                                                                              position, pivot, {up[0], up[1], up[2]});
                                                                                                                                                                          if (!rotation)
                                                                                                                                                                              throw std::invalid_argument("Camera eye, target and up must define a finite nondegenerate view");
                                                                                                                                                                          auto& record = viewRecord(id);
                                                                                                                                                                          vis::resetTransientNavigation(record.camera);
                                                                                                                                                                          record.camera.setViewMatrix(*rotation, position);
                                                                                                                                                                          record.camera.camera.setPivot(pivot);
                                                                                                                                                                          requireSuccess(workspace().bumpViewState(id));
                                                                                                                                                                          requestFrame();
                                                                                                                                                                      }); }, nb::arg("view_id"), nb::arg("eye"), nb::arg("target"), nb::arg("up") = std::array<float, 3>{0.0f, 1.0f, 0.0f});

        m.def("workspace_set_projection", [](const vis::ViewId id, const bool orthographic, const std::optional<float> focal_length_mm, const std::optional<float> ortho_scale) { onWorkspaceThread([id, orthographic, focal_length_mm, ortho_scale] {
                                                                                                                                                                                      auto projection = viewRecord(id).projection;
                                                                                                                                                                                      projection.orthographic = orthographic;
                                                                                                                                                                                      if (focal_length_mm)
                                                                                                                                                                                          projection.focal_length_mm = *focal_length_mm;
                                                                                                                                                                                      if (ortho_scale)
                                                                                                                                                                                          projection.ortho_scale = *ortho_scale;
                                                                                                                                                                                      requireSuccess(workspace().setViewProjection(id, projection));
                                                                                                                                                                                      requestFrame();
                                                                                                                                                                                  }); }, nb::arg("view_id"), nb::arg("orthographic"), nb::arg("focal_length_mm") = nb::none(), nb::arg("ortho_scale") = nb::none());
    }
} // namespace lfs::python
