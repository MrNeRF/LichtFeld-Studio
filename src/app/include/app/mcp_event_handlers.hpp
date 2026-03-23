/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/event_bridge/scoped_handler.hpp"
#include "core/events.hpp"
#include "core/path_utils.hpp"

#include <algorithm>
#include <array>
#include <functional>
#include <memory>
#include <string>
#include <string_view>
#include <utility>

#include <nlohmann/json.hpp>

namespace lfs::app {

    enum class McpEventStreamKind {
        RuntimeJournal,
        SubscriptionQueue,
    };

    inline constexpr std::array<std::string_view, 22> kMcpRuntimeEventTypes = {
        "editor.started",
        "editor.completed",
        "scene.loaded",
        "scene.cleared",
        "selection.changed",
        "training.started",
        "training.progress",
        "training.paused",
        "training.resumed",
        "training.completed",
        "training.stopped",
        "dataset.load_started",
        "dataset.load_progress",
        "dataset.load_completed",
        "checkpoint.saved",
        "export.completed",
        "export.failed",
        "video_export.completed",
        "video_export.failed",
        "mesh2splat.completed",
        "mesh2splat.failed",
        "disk_space.save_failed",
    };

    inline constexpr std::array<std::string_view, 28> kMcpSubscriptionEventTypes = {
        "editor.started",
        "editor.completed",
        "scene.loaded",
        "scene.cleared",
        "scene.changed",
        "scene.node_added",
        "scene.node_removed",
        "scene.node_reparented",
        "selection.changed",
        "training.started",
        "training.progress",
        "training.paused",
        "training.resumed",
        "training.completed",
        "training.stopped",
        "dataset.load_started",
        "dataset.load_progress",
        "dataset.load_completed",
        "render.frame",
        "checkpoint.saved",
        "keyframes.changed",
        "export.completed",
        "export.failed",
        "video_export.completed",
        "video_export.failed",
        "mesh2splat.completed",
        "mesh2splat.failed",
        "disk_space.save_failed",
    };

    inline const char* mcp_export_format_name(const core::ExportFormat format) {
        switch (format) {
        case core::ExportFormat::PLY:
            return "ply";
        case core::ExportFormat::SOG:
            return "sog";
        case core::ExportFormat::SPZ:
            return "spz";
        case core::ExportFormat::HTML_VIEWER:
            return "html";
        }
        return "unknown";
    }

    inline nlohmann::json mcp_runtime_event_types_json() {
        nlohmann::json event_types = nlohmann::json::array();
        for (const auto type : kMcpRuntimeEventTypes) {
            event_types.push_back(type);
        }
        return event_types;
    }

    inline nlohmann::json mcp_subscription_event_types_json() {
        nlohmann::json event_types = nlohmann::json::array();
        for (const auto type : kMcpSubscriptionEventTypes) {
            event_types.push_back(type);
        }
        return event_types;
    }

    inline bool is_mcp_runtime_event_type(const std::string_view type) {
        return std::find(kMcpRuntimeEventTypes.begin(), kMcpRuntimeEventTypes.end(), type) !=
               kMcpRuntimeEventTypes.end();
    }

    template <typename PublishFn>
    void register_mcp_event_handlers(event::ScopedHandler& handlers,
                                     const McpEventStreamKind kind,
                                     PublishFn&& publish) {
        auto publisher = std::make_shared<std::decay_t<PublishFn>>(std::forward<PublishFn>(publish));
        auto register_handler = [&handlers, publisher]<typename Event, typename Builder>(
                                    std::string_view type,
                                    Builder&& builder) {
            handlers.subscribe<Event>(
                [publisher, type = std::string(type), builder = std::forward<Builder>(builder)](const Event& event) mutable {
                    std::invoke(*publisher, type, builder(event));
                });
        };

        register_handler.template operator()<core::events::state::EditorScriptStarted>("editor.started", [](const auto& event) {
            return nlohmann::json{
                {"path", core::path_to_utf8(event.path)},
                {"code_chars", static_cast<int64_t>(event.code_chars)},
            };
        });
        register_handler.template operator()<core::events::state::EditorScriptCompleted>("editor.completed", [](const auto& event) {
            return nlohmann::json{
                {"path", core::path_to_utf8(event.path)},
                {"code_chars", static_cast<int64_t>(event.code_chars)},
                {"output_chars", static_cast<int64_t>(event.output_chars)},
                {"success", event.success},
                {"interrupted", event.interrupted},
            };
        });
        register_handler.template operator()<core::events::state::SceneLoaded>("scene.loaded", [](const auto& event) {
            return nlohmann::json{
                {"path", core::path_to_utf8(event.path)},
                {"type", static_cast<int>(event.type)},
                {"num_gaussians", static_cast<int64_t>(event.num_gaussians)},
                {"checkpoint_iteration", event.checkpoint_iteration},
            };
        });
        register_handler.template operator()<core::events::state::SceneCleared>("scene.cleared", [](const auto&) {
            return nlohmann::json::object();
        });
        register_handler.template operator()<core::events::state::SelectionChanged>("selection.changed", [](const auto& event) {
            return nlohmann::json{
                {"has_selection", event.has_selection},
                {"count", event.count},
            };
        });
        register_handler.template operator()<core::events::state::TrainingStarted>("training.started", [](const auto& event) {
            return nlohmann::json{{"total_iterations", event.total_iterations}};
        });
        register_handler.template operator()<core::events::state::TrainingProgress>("training.progress", [](const auto& event) {
            return nlohmann::json{
                {"iteration", event.iteration},
                {"loss", event.loss},
                {"num_gaussians", static_cast<int64_t>(event.num_gaussians)},
                {"is_refining", event.is_refining},
            };
        });
        register_handler.template operator()<core::events::state::TrainingPaused>("training.paused", [](const auto& event) {
            return nlohmann::json{{"iteration", event.iteration}};
        });
        register_handler.template operator()<core::events::state::TrainingResumed>("training.resumed", [](const auto& event) {
            return nlohmann::json{{"iteration", event.iteration}};
        });
        register_handler.template operator()<core::events::state::TrainingCompleted>("training.completed", [](const auto& event) {
            nlohmann::json payload{
                {"iteration", event.iteration},
                {"final_loss", event.final_loss},
                {"elapsed_seconds", event.elapsed_seconds},
                {"success", event.success},
                {"user_stopped", event.user_stopped},
            };
            if (event.error) {
                payload["error"] = *event.error;
            }
            return payload;
        });
        register_handler.template operator()<core::events::state::TrainingStopped>("training.stopped", [](const auto& event) {
            return nlohmann::json{
                {"iteration", event.iteration},
                {"user_requested", event.user_requested},
            };
        });
        register_handler.template operator()<core::events::state::DatasetLoadStarted>("dataset.load_started", [](const auto& event) {
            return nlohmann::json{{"path", core::path_to_utf8(event.path)}};
        });
        register_handler.template operator()<core::events::state::DatasetLoadProgress>("dataset.load_progress", [](const auto& event) {
            return nlohmann::json{
                {"path", core::path_to_utf8(event.path)},
                {"progress", event.progress},
                {"step", event.step},
            };
        });
        register_handler.template operator()<core::events::state::DatasetLoadCompleted>("dataset.load_completed", [](const auto& event) {
            nlohmann::json payload{
                {"path", core::path_to_utf8(event.path)},
                {"success", event.success},
                {"num_images", static_cast<int64_t>(event.num_images)},
                {"num_points", static_cast<int64_t>(event.num_points)},
            };
            if (event.error) {
                payload["error"] = *event.error;
            }
            return payload;
        });
        register_handler.template operator()<core::events::state::CheckpointSaved>("checkpoint.saved", [](const auto& event) {
            return nlohmann::json{
                {"iteration", event.iteration},
                {"path", core::path_to_utf8(event.path)},
            };
        });
        register_handler.template operator()<core::events::state::ExportCompleted>("export.completed", [](const auto& event) {
            return nlohmann::json{
                {"path", core::path_to_utf8(event.path)},
                {"format", mcp_export_format_name(event.format)},
            };
        });
        register_handler.template operator()<core::events::state::ExportFailed>("export.failed", [](const auto& event) {
            return nlohmann::json{{"error", event.error}};
        });
        register_handler.template operator()<core::events::state::VideoExportCompleted>("video_export.completed", [](const auto& event) {
            return nlohmann::json{
                {"path", core::path_to_utf8(event.path)},
                {"total_frames", event.total_frames},
            };
        });
        register_handler.template operator()<core::events::state::VideoExportFailed>("video_export.failed", [](const auto& event) {
            return nlohmann::json{{"error", event.error}};
        });
        register_handler.template operator()<core::events::state::Mesh2SplatCompleted>("mesh2splat.completed", [](const auto& event) {
            return nlohmann::json{
                {"source_name", event.source_name},
                {"node_name", event.node_name},
                {"num_gaussians", static_cast<int64_t>(event.num_gaussians)},
            };
        });
        register_handler.template operator()<core::events::state::Mesh2SplatFailed>("mesh2splat.failed", [](const auto& event) {
            return nlohmann::json{{"error", event.error}};
        });
        register_handler.template operator()<core::events::state::DiskSpaceSaveFailed>("disk_space.save_failed", [](const auto& event) {
            return nlohmann::json{
                {"iteration", event.iteration},
                {"path", core::path_to_utf8(event.path)},
                {"error", event.error},
                {"required_bytes", static_cast<int64_t>(event.required_bytes)},
                {"available_bytes", static_cast<int64_t>(event.available_bytes)},
                {"is_disk_space_error", event.is_disk_space_error},
                {"is_checkpoint", event.is_checkpoint},
            };
        });

        if (kind != McpEventStreamKind::SubscriptionQueue) {
            return;
        }

        register_handler.template operator()<core::events::state::SceneChanged>("scene.changed", [](const auto& event) {
            return nlohmann::json{{"mutation_flags", event.mutation_flags}};
        });
        register_handler.template operator()<core::events::state::PLYAdded>("scene.node_added", [](const auto& event) {
            return nlohmann::json{
                {"name", event.name},
                {"node_gaussians", static_cast<int64_t>(event.node_gaussians)},
                {"total_gaussians", static_cast<int64_t>(event.total_gaussians)},
                {"is_visible", event.is_visible},
                {"parent_name", event.parent_name},
                {"is_group", event.is_group},
                {"node_type", event.node_type},
            };
        });
        register_handler.template operator()<core::events::state::PLYRemoved>("scene.node_removed", [](const auto& event) {
            return nlohmann::json{
                {"name", event.name},
                {"children_kept", event.children_kept},
                {"parent_of_removed", event.parent_of_removed},
            };
        });
        register_handler.template operator()<core::events::state::NodeReparented>("scene.node_reparented", [](const auto& event) {
            return nlohmann::json{
                {"name", event.name},
                {"old_parent", event.old_parent},
                {"new_parent", event.new_parent},
            };
        });
        register_handler.template operator()<core::events::state::FrameRendered>("render.frame", [](const auto& event) {
            return nlohmann::json{
                {"render_ms", event.render_ms},
                {"fps", event.fps},
                {"num_gaussians", event.num_gaussians},
            };
        });
        register_handler.template operator()<core::events::state::KeyframeListChanged>("keyframes.changed", [](const auto& event) {
            return nlohmann::json{{"count", static_cast<int64_t>(event.count)}};
        });
    }

} // namespace lfs::app
