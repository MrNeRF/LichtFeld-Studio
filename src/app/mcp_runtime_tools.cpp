/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "app/mcp_runtime_tools.hpp"
#include "app/mcp_app_utils.hpp"
#include "app/mcp_event_handlers.hpp"

#include "core/event_bridge/scoped_handler.hpp"
#include "core/events.hpp"
#include "core/path_utils.hpp"
#include "visualizer/gui/async_task_manager.hpp"
#include "visualizer/gui/gui_manager.hpp"
#include "visualizer/gui/panels/python_console_panel.hpp"
#include "visualizer/operator/operator_registry.hpp"
#include "visualizer/training/training_manager.hpp"
#include "visualizer/training/training_state.hpp"
#include "visualizer/visualizer.hpp"
#include "visualizer/visualizer_impl.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <deque>
#include <expected>
#include <future>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <thread>
#include <type_traits>
#include <unordered_set>
#include <vector>

namespace lfs::app {

    namespace {

        using json = nlohmann::json;
        using mcp::McpResourceContent;

        constexpr std::array<std::string_view, 7> kRuntimeJobIds = {
            "editor.python",
            "training.main",
            "export.scene",
            "import.dataset",
            "export.video",
            "mesh2splat",
            "operator.modal",
        };

        json supported_runtime_job_ids_json() {
            json job_ids = json::array();
            for (const auto job_id : kRuntimeJobIds) {
                job_ids.push_back(job_id);
            }
            return job_ids;
        }

        json supported_runtime_event_types_json() {
            return mcp_runtime_event_types_json();
        }

        bool is_supported_runtime_event_type(const std::string_view type) {
            return is_mcp_runtime_event_type(type);
        }

        std::string runtime_job_resource_uri(const std::string_view job_id) {
            return "lichtfeld://runtime/jobs/" + std::string(job_id);
        }

        std::string runtime_event_resource_uri(const std::string_view event_type) {
            return "lichtfeld://runtime/events/" + std::string(event_type);
        }

        std::string_view runtime_job_label(const std::string_view job_id) {
            if (job_id == "editor.python") {
                return "Python Editor";
            }
            if (job_id == "training.main") {
                return "Training";
            }
            if (job_id == "export.scene") {
                return "Scene Export";
            }
            if (job_id == "import.dataset") {
                return "Dataset Import";
            }
            if (job_id == "export.video") {
                return "Video Export";
            }
            if (job_id == "mesh2splat") {
                return "Mesh to Splat";
            }
            if (job_id == "operator.modal") {
                return "Modal Operator";
            }
            return "Runtime Job";
        }

        json runtime_job_event_types_json(const std::string_view job_id) {
            if (job_id == "editor.python") {
                return json::array({"editor.started", "editor.completed"});
            }
            if (job_id == "training.main") {
                return json::array({
                    "training.started",
                    "training.progress",
                    "training.paused",
                    "training.resumed",
                    "training.completed",
                    "training.stopped",
                    "checkpoint.saved",
                    "disk_space.save_failed",
                });
            }
            if (job_id == "export.scene") {
                return json::array({"export.completed", "export.failed"});
            }
            if (job_id == "import.dataset") {
                return json::array({
                    "dataset.load_started",
                    "dataset.load_progress",
                    "dataset.load_completed",
                });
            }
            if (job_id == "export.video") {
                return json::array({"video_export.completed", "video_export.failed"});
            }
            if (job_id == "mesh2splat") {
                return json::array({"mesh2splat.completed", "mesh2splat.failed"});
            }
            if (job_id == "operator.modal") {
                return json::array();
            }
            return json::array();
        }

        json runtime_job_event_resource_uris_json(const std::string_view job_id) {
            json uris = json::array();
            for (const auto& event_type : runtime_job_event_types_json(job_id)) {
                uris.push_back(runtime_event_resource_uri(event_type.get<std::string>()));
            }
            return uris;
        }

        void add_runtime_job_links(json& payload) {
            const std::string job_id = payload.value("id", "");
            payload["resource_uri"] = runtime_job_resource_uri(job_id);
            payload["event_types"] = runtime_job_event_types_json(job_id);
            payload["event_resource_uris"] = runtime_job_event_resource_uris_json(job_id);
        }

        json runtime_catalog_json() {
            json jobs = json::array();
            for (const auto job_id : kRuntimeJobIds) {
                jobs.push_back(json{
                    {"id", job_id},
                    {"label", runtime_job_label(job_id)},
                    {"resource_uri", runtime_job_resource_uri(job_id)},
                    {"event_types", runtime_job_event_types_json(job_id)},
                    {"event_resource_uris", runtime_job_event_resource_uris_json(job_id)},
                });
            }

            json events = json::array();
            for (const auto event_type : kMcpRuntimeEventTypes) {
                events.push_back(json{
                    {"type", event_type},
                    {"resource_uri", runtime_event_resource_uri(event_type)},
                });
            }

            return json{
                {"catalog_uri", "lichtfeld://runtime/catalog"},
                {"state_uri", "lichtfeld://runtime/state"},
                {"jobs_uri", "lichtfeld://runtime/jobs"},
                {"events_uri", "lichtfeld://runtime/events"},
                {"supported_job_ids", supported_runtime_job_ids_json()},
                {"supported_event_types", supported_runtime_event_types_json()},
                {"jobs", std::move(jobs)},
                {"events", std::move(events)},
            };
        }

        vis::VisualizerImpl* as_visualizer_impl(vis::Visualizer* viewer) {
            return dynamic_cast<vis::VisualizerImpl*>(viewer);
        }

        const char* training_state_to_string(const vis::TrainingState state) {
            switch (state) {
            case vis::TrainingState::Idle:
                return "idle";
            case vis::TrainingState::Ready:
                return "ready";
            case vis::TrainingState::Running:
                return "running";
            case vis::TrainingState::Paused:
                return "paused";
            case vis::TrainingState::Stopping:
                return "stopping";
            case vis::TrainingState::Finished:
                return "finished";
            }
            return "unknown";
        }

        const char* export_format_to_string(const core::ExportFormat format) {
            switch (format) {
            case core::ExportFormat::PLY:
                return "ply";
            case core::ExportFormat::SOG:
                return "sog";
            case core::ExportFormat::SPZ:
                return "spz";
            case core::ExportFormat::HTML_VIEWER:
                return "html_viewer";
            }
            return "unknown";
        }

        const char* modal_state_to_string(const vis::op::ModalState state) {
            switch (state) {
            case vis::op::ModalState::IDLE:
                return "idle";
            case vis::op::ModalState::ACTIVE_CPP:
                return "active_cpp";
            case vis::op::ModalState::ACTIVE_PYTHON:
                return "active_python";
            }
            return "unknown";
        }

        json text_snapshot_json(const std::string& text,
                                const size_t max_chars,
                                const bool tail) {
            const size_t total_chars = text.size();
            if (max_chars == 0 || total_chars == 0) {
                return json{
                    {"mime_type", "text/plain"},
                    {"text", ""},
                    {"returned_chars", 0},
                    {"total_chars", static_cast<int64_t>(total_chars)},
                    {"truncated", total_chars > 0},
                    {"tail", tail},
                };
            }

            const size_t returned_chars = std::min(max_chars, total_chars);
            const size_t start = tail && total_chars > returned_chars
                                     ? total_chars - returned_chars
                                     : 0;
            return json{
                {"mime_type", "text/plain"},
                {"text", text.substr(start, returned_chars)},
                {"returned_chars", static_cast<int64_t>(returned_chars)},
                {"total_chars", static_cast<int64_t>(total_chars)},
                {"truncated", returned_chars < total_chars},
                {"tail", tail},
            };
        }

        json unavailable_job_json(const std::string_view id,
                                  const std::string_view label,
                                  const std::string_view kind,
                                  const std::string_view reason) {
            json payload{
                {"id", id},
                {"label", label},
                {"kind", kind},
                {"active", false},
                {"status", "unavailable"},
                {"stage", ""},
                {"progress", nullptr},
                {"cancel_supported", false},
                {"dismiss_supported", false},
                {"actions",
                 json{
                     {"start", false},
                     {"pause", false},
                     {"resume", false},
                     {"cancel", false},
                     {"dismiss", false},
                 }},
                {"details", json{{"reason", reason}}},
            };
            add_runtime_job_links(payload);
            return payload;
        }

        json editor_job_json(const bool include_output,
                             const size_t output_max_chars,
                             const bool output_tail) {
            auto& console = vis::gui::panels::PythonConsoleState::getInstance();
            auto* const editor = console.getEditor();
            auto* const output_terminal = console.getOutputTerminal();

            if (!editor || !output_terminal) {
                return unavailable_job_json(
                    "editor.python",
                    "Python Editor",
                    "editor",
                    "Python editor or output terminal is not initialized");
            }

            const std::string output = console.getOutputText();
            json payload{
                {"id", "editor.python"},
                {"label", "Python Editor"},
                {"kind", "editor"},
                {"active", console.isScriptRunning()},
                {"status", console.isScriptRunning() ? "running" : "idle"},
                {"stage", console.isScriptRunning() ? "Executing Python script" : ""},
                {"progress", nullptr},
                {"cancel_supported", console.isScriptRunning()},
                {"dismiss_supported", false},
                {"actions",
                 json{
                     {"start", false},
                     {"pause", false},
                     {"resume", false},
                     {"cancel", console.isScriptRunning()},
                     {"dismiss", false},
                 }},
                {"details",
                 json{
                     {"active_tab", console.getActiveTab()},
                     {"modified", console.isModified()},
                     {"terminal_focused", console.isTerminalFocused()},
                     {"script_path", core::path_to_utf8(console.getScriptPath())},
                     {"code_chars", static_cast<int64_t>(console.getEditorText().size())},
                     {"output_total_chars", static_cast<int64_t>(output.size())},
                     {"has_editor", editor != nullptr},
                     {"has_output_terminal", output_terminal != nullptr},
                 }},
            };

            if (include_output) {
                payload["output"] = text_snapshot_json(output, output_max_chars, output_tail);
            }

            add_runtime_job_links(payload);
            return payload;
        }

        json training_job_json(vis::VisualizerImpl& viewer) {
            auto* const trainer = viewer.getTrainerManager();
            if (!trainer) {
                return unavailable_job_json(
                    "training.main",
                    "Training",
                    "training",
                    "Trainer manager is not initialized");
            }

            const auto state = trainer->getState();
            const int total_iterations = trainer->getTotalIterations();
            const int current_iteration = trainer->getCurrentIteration();
            json payload{
                {"id", "training.main"},
                {"label", "Training"},
                {"kind", "training"},
                {"active", trainer->isTrainingActive()},
                {"status",
                 state == vis::TrainingState::Finished && !trainer->getLastError().empty()
                     ? "failed"
                     : training_state_to_string(state)},
                {"stage", vis::TrainingStateMachine::stateName(state)},
                {"progress", total_iterations > 0
                                 ? json(std::clamp(
                                       static_cast<double>(current_iteration) /
                                           static_cast<double>(total_iterations),
                                       0.0,
                                       1.0))
                                 : json(nullptr)},
                {"error", trainer->getLastError()},
                {"cancel_supported", trainer->canStop()},
                {"dismiss_supported", false},
                {"actions",
                 json{
                     {"start", trainer->canStart()},
                     {"pause", trainer->canPause()},
                     {"resume", trainer->canResume()},
                     {"cancel", trainer->canStop()},
                     {"dismiss", false},
                 }},
                {"details",
                 json{
                     {"has_trainer", trainer->hasTrainer()},
                     {"current_iteration", current_iteration},
                     {"total_iterations", total_iterations},
                     {"current_loss", trainer->getCurrentLoss()},
                     {"num_gaussians", trainer->getNumSplats()},
                     {"max_gaussians", trainer->getMaxGaussians()},
                     {"strategy_type", trainer->getStrategyType()},
                     {"gut_enabled", trainer->isGutEnabled()},
                     {"elapsed_seconds", trainer->getElapsedSeconds()},
                     {"eta_seconds", trainer->getEstimatedRemainingSeconds()},
                 }},
            };

            if (trainer->getLastError().empty()) {
                payload.erase("error");
            }

            add_runtime_job_links(payload);
            return payload;
        }

        json export_job_json(vis::gui::GuiManager& gui) {
            const auto& tasks = gui.asyncTasks();
            const bool active = tasks.isExporting();
            const std::string error = tasks.getExportError();
            const std::string stage = tasks.getExportStage();
            std::string path = core::path_to_utf8(tasks.getExportPath());
            json payload{
                {"id", "export.scene"},
                {"label", "Scene Export"},
                {"kind", "export"},
                {"active", active},
                {"status",
                 active                                ? "running"
                 : stage == "Cancelled"               ? "cancelled"
                 : !error.empty() || stage == "Failed" ? "failed"
                 : stage == "Complete"                 ? "finished"
                                                       : "idle"},
                {"stage", stage},
                {"progress", tasks.getExportProgress()},
                {"cancel_supported", active},
                {"dismiss_supported", false},
                {"actions",
                 json{
                     {"start", false},
                     {"pause", false},
                     {"resume", false},
                     {"cancel", active},
                     {"dismiss", false},
                 }},
                {"details",
                 json{
                     {"format", export_format_to_string(tasks.getExportFormat())},
                     {"path", path},
                 }},
            };

            if (!error.empty()) {
                payload["error"] = error;
            }

            add_runtime_job_links(payload);
            return payload;
        }

        json import_job_json(vis::gui::GuiManager& gui) {
            const auto& tasks = gui.asyncTasks();
            const bool active = tasks.isImporting();
            const bool show_completion = tasks.isImportCompletionShowing();
            const std::string error = tasks.getImportError();
            const std::string stage = tasks.getImportStage();
            const bool success = tasks.getImportSuccess();

            std::string status = "idle";
            if (active) {
                status = "running";
            } else if (!error.empty() || stage == "Failed") {
                status = "failed";
            } else if (stage == "Complete") {
                status = "finished";
            }

            json payload{
                {"id", "import.dataset"},
                {"label", "Dataset Import"},
                {"kind", "import"},
                {"active", active},
                {"status", status},
                {"stage", stage},
                {"progress", tasks.getImportProgress()},
                {"cancel_supported", false},
                {"dismiss_supported", show_completion},
                {"actions",
                 json{
                     {"start", false},
                     {"pause", false},
                     {"resume", false},
                     {"cancel", false},
                     {"dismiss", show_completion},
                 }},
                {"details",
                 json{
                     {"path", tasks.getImportPath()},
                     {"dataset_type", tasks.getImportDatasetType()},
                     {"show_completion", show_completion},
                     {"success", success},
                     {"num_images", static_cast<int64_t>(tasks.getImportNumImages())},
                     {"num_points", static_cast<int64_t>(tasks.getImportNumPoints())},
                     {"seconds_since_completion", tasks.getImportSecondsSinceCompletion()},
                 }},
            };

            if (!error.empty()) {
                payload["error"] = error;
            }

            add_runtime_job_links(payload);
            return payload;
        }

        json video_export_job_json(vis::gui::GuiManager& gui) {
            const auto& tasks = gui.asyncTasks();
            const bool active = tasks.isExportingVideo();
            const std::string error = tasks.getVideoExportError();
            const std::string stage = tasks.getVideoExportStage();

            json payload{
                {"id", "export.video"},
                {"label", "Video Export"},
                {"kind", "video_export"},
                {"active", active},
                {"status",
                 active                                ? "running"
                 : !error.empty() || stage == "Failed" ? "failed"
                 : stage == "Complete"                 ? "finished"
                                                       : "idle"},
                {"stage", stage},
                {"progress", tasks.getVideoExportProgress()},
                {"cancel_supported", active},
                {"dismiss_supported", false},
                {"actions",
                 json{
                     {"start", false},
                     {"pause", false},
                     {"resume", false},
                     {"cancel", active},
                     {"dismiss", false},
                 }},
                {"details",
                 json{
                     {"path", core::path_to_utf8(tasks.getVideoExportPath())},
                     {"current_frame", tasks.getVideoExportCurrentFrame()},
                     {"total_frames", tasks.getVideoExportTotalFrames()},
                 }},
            };

            if (!error.empty()) {
                payload["error"] = error;
            }

            add_runtime_job_links(payload);
            return payload;
        }

        json mesh2splat_job_json(vis::gui::GuiManager& gui) {
            const auto& tasks = gui.asyncTasks();
            const bool active = tasks.isMesh2SplatActive();
            const std::string error = tasks.getMesh2SplatError();
            const std::string stage = tasks.getMesh2SplatStage();

            json payload{
                {"id", "mesh2splat"},
                {"label", "Mesh to Splat"},
                {"kind", "conversion"},
                {"active", active},
                {"status",
                 active                                ? "running"
                 : !error.empty() || stage == "Failed" ? "failed"
                 : stage == "Complete"                 ? "finished"
                                                       : "idle"},
                {"stage", stage},
                {"progress", tasks.getMesh2SplatProgress()},
                {"cancel_supported", false},
                {"dismiss_supported", false},
                {"actions",
                 json{
                     {"start", false},
                     {"pause", false},
                     {"resume", false},
                     {"cancel", false},
                     {"dismiss", false},
                 }},
                {"details",
                 json{
                     {"source_name", tasks.getMesh2SplatSourceName()},
                 }},
            };

            if (!error.empty()) {
                payload["error"] = error;
            }

            add_runtime_job_links(payload);
            return payload;
        }

        json operator_modal_job_json() {
            auto& operators = vis::op::operators();
            const auto modal_state = operators.modalState();
            const bool active = modal_state != vis::op::ModalState::IDLE;
            json payload{
                {"id", "operator.modal"},
                {"label", "Modal Operator"},
                {"kind", "operator"},
                {"active", active},
                {"status", modal_state_to_string(modal_state)},
                {"stage", modal_state_to_string(modal_state)},
                {"progress", nullptr},
                {"cancel_supported", active},
                {"dismiss_supported", false},
                {"actions",
                 json{
                     {"start", false},
                     {"pause", false},
                     {"resume", false},
                     {"cancel", active},
                     {"dismiss", false},
                 }},
                {"details",
                 json{
                     {"operator_id", operators.activeModalId()},
                 }},
            };
            add_runtime_job_links(payload);
            return payload;
        }

        std::expected<json, std::string> describe_job_payload_on_gui(
            vis::Visualizer* viewer,
            const std::string& job_id,
            const bool include_output,
            const size_t output_max_chars,
            const bool output_tail) {
            auto* const viewer_impl = as_visualizer_impl(viewer);
            if (!viewer_impl) {
                return std::unexpected("Visualizer implementation is unavailable");
            }

            auto* const gui = viewer_impl->getGuiManager();

            if (job_id == "editor.python") {
                return editor_job_json(include_output, output_max_chars, output_tail);
            }
            if (job_id == "training.main") {
                return training_job_json(*viewer_impl);
            }
            if (job_id == "export.scene") {
                return gui ? std::expected<json, std::string>(export_job_json(*gui))
                           : std::expected<json, std::string>(unavailable_job_json(
                                 "export.scene",
                                 "Scene Export",
                                 "export",
                                 "GUI manager is not initialized"));
            }
            if (job_id == "import.dataset") {
                return gui ? std::expected<json, std::string>(import_job_json(*gui))
                           : std::expected<json, std::string>(unavailable_job_json(
                                 "import.dataset",
                                 "Dataset Import",
                                 "import",
                                 "GUI manager is not initialized"));
            }
            if (job_id == "export.video") {
                return gui ? std::expected<json, std::string>(video_export_job_json(*gui))
                           : std::expected<json, std::string>(unavailable_job_json(
                                 "export.video",
                                 "Video Export",
                                 "video_export",
                                 "GUI manager is not initialized"));
            }
            if (job_id == "mesh2splat") {
                return gui ? std::expected<json, std::string>(mesh2splat_job_json(*gui))
                           : std::expected<json, std::string>(unavailable_job_json(
                                 "mesh2splat",
                                 "Mesh to Splat",
                                 "conversion",
                                 "GUI manager is not initialized"));
            }
            if (job_id == "operator.modal") {
                return operator_modal_job_json();
            }

            return std::unexpected("Unsupported runtime job id: " + job_id);
        }

        std::expected<json, std::string> list_jobs_payload_on_gui(
            vis::Visualizer* viewer,
            const bool include_output,
            const size_t output_max_chars,
            const bool output_tail) {
            json jobs = json::array();
            for (const auto job_id : kRuntimeJobIds) {
                auto payload = describe_job_payload_on_gui(
                    viewer,
                    std::string(job_id),
                    include_output,
                    output_max_chars,
                    output_tail);
                if (!payload) {
                    return std::unexpected(payload.error());
                }
                jobs.push_back(std::move(*payload));
            }

            int64_t active_jobs = 0;
            int64_t cancellable_jobs = 0;
            for (const auto& job : jobs) {
                if (job.value("active", false)) {
                    ++active_jobs;
                }
                if (job.value("cancel_supported", false)) {
                    ++cancellable_jobs;
                }
            }

            return json{
                {"catalog_uri", "lichtfeld://runtime/catalog"},
                {"state_uri", "lichtfeld://runtime/state"},
                {"jobs_uri", "lichtfeld://runtime/jobs"},
                {"events_uri", "lichtfeld://runtime/events"},
                {"supported_job_ids", supported_runtime_job_ids_json()},
                {"supported_event_types", supported_runtime_event_types_json()},
                {"count", static_cast<int64_t>(jobs.size())},
                {"active_job_count", active_jobs},
                {"cancellable_job_count", cancellable_jobs},
                {"jobs", std::move(jobs)},
            };
        }

        class RuntimeEventJournal {
        public:
            static RuntimeEventJournal& instance() {
                static RuntimeEventJournal journal;
                return journal;
            }

            json tail_payload(const std::vector<std::string>& types,
                              const size_t max_events) {
                std::unordered_set<std::string> filters;
                for (const auto& type : types) {
                    filters.insert(type);
                }

                std::lock_guard lock(mutex_);
                json events = json::array();
                size_t matched = 0;
                for (auto it = events_.rbegin(); it != events_.rend(); ++it) {
                    if (!filters.empty() && !filters.contains("*") && !filters.contains(it->type)) {
                        continue;
                    }
                    ++matched;
                    events.push_back(json{
                        {"sequence", it->sequence},
                        {"timestamp_ms", it->timestamp_ms},
                        {"type", it->type},
                        {"data", it->data},
                    });
                    if (events.size() >= max_events) {
                        break;
                    }
                }
                std::reverse(events.begin(), events.end());

                return json{
                    {"success", true},
                    {"catalog_uri", "lichtfeld://runtime/catalog"},
                    {"events_uri", "lichtfeld://runtime/events"},
                    {"supported_types", supported_runtime_event_types_json()},
                    {"filters", types},
                    {"retained", static_cast<int64_t>(events_.size())},
                    {"matched", static_cast<int64_t>(matched)},
                    {"returned", static_cast<int64_t>(events.size())},
                    {"events", std::move(events)},
                };
            }

            json clear_payload(const std::vector<std::string>& types) {
                std::unordered_set<std::string> filters;
                for (const auto& type : types) {
                    filters.insert(type);
                }

                std::lock_guard lock(mutex_);
                const size_t before = events_.size();
                if (filters.empty() || filters.contains("*")) {
                    events_.clear();
                } else {
                    std::erase_if(events_, [&filters](const auto& entry) {
                        return filters.contains(entry.type);
                    });
                }

                return json{
                    {"success", true},
                    {"filters", types},
                    {"removed", static_cast<int64_t>(before - events_.size())},
                    {"retained", static_cast<int64_t>(events_.size())},
                };
            }

            size_t size() const {
                std::lock_guard lock(mutex_);
                return events_.size();
            }

        private:
            struct Entry {
                int64_t sequence = 0;
                int64_t timestamp_ms = 0;
                std::string type;
                json data;
            };

            static constexpr size_t kMaxRetainedEvents = 1024;

            mutable std::mutex mutex_;
            std::deque<Entry> events_;
            int64_t next_sequence_ = 1;
            event::ScopedHandler handlers_;

            RuntimeEventJournal() {
                register_mcp_event_handlers(
                    handlers_,
                    McpEventStreamKind::RuntimeJournal,
                    [this](const std::string& type, json payload) {
                        publish(type, std::move(payload));
                    });
            }

            void publish(const std::string& type, json payload) {
                const auto timestamp_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                                              std::chrono::system_clock::now().time_since_epoch())
                                              .count();

                std::lock_guard lock(mutex_);
                if (events_.size() >= kMaxRetainedEvents) {
                    events_.pop_front();
                }
                events_.push_back(Entry{
                    .sequence = next_sequence_++,
                    .timestamp_ms = timestamp_ms,
                    .type = type,
                    .data = std::move(payload),
                });
            }
        };

        std::expected<std::vector<std::string>, std::string> optional_string_array_arg(
            const json& args,
            const std::string_view field_name) {
            if (!args.contains(field_name) || args[field_name].is_null()) {
                return std::vector<std::string>{};
            }
            if (!args[field_name].is_array()) {
                return std::unexpected("Field '" + std::string(field_name) + "' must be an array of strings");
            }

            std::vector<std::string> result;
            result.reserve(args[field_name].size());
            for (const auto& item : args[field_name]) {
                if (!item.is_string()) {
                    return std::unexpected("Field '" + std::string(field_name) + "' must contain only strings");
                }
                result.push_back(item.get<std::string>());
            }
            return result;
        }

        json job_wait_fingerprint(const json& job) {
            json fingerprint{
                {"active", job.value("active", false)},
                {"status", job.value("status", "")},
                {"stage", job.value("stage", "")},
            };
            if (job.contains("progress")) {
                fingerprint["progress"] = job["progress"];
            }
            if (job.contains("error")) {
                fingerprint["error"] = job["error"];
            }
            if (job.contains("details")) {
                const auto& details = job["details"];
                if (details.contains("output_total_chars")) {
                    fingerprint["output_total_chars"] = details["output_total_chars"];
                }
                if (details.contains("current_iteration")) {
                    fingerprint["current_iteration"] = details["current_iteration"];
                }
                if (details.contains("current_frame")) {
                    fingerprint["current_frame"] = details["current_frame"];
                }
            }
            return fingerprint;
        }

        bool wait_condition_met(const json& baseline,
                                const json& current,
                                const std::string_view until) {
            if (until == "inactive") {
                return !current.value("active", false);
            }
            if (until == "active") {
                return current.value("active", false);
            }
            if (until == "changed") {
                return job_wait_fingerprint(baseline) != job_wait_fingerprint(current);
            }
            return false;
        }

        std::expected<void, std::string> control_job_on_gui(
            vis::Visualizer* viewer,
            const std::string& job_id,
            const std::string& action) {
            auto* const viewer_impl = as_visualizer_impl(viewer);
            if (!viewer_impl) {
                return std::unexpected("Visualizer implementation is unavailable");
            }

            auto* const gui = viewer_impl->getGuiManager();

            if (job_id == "editor.python") {
                auto& console = vis::gui::panels::PythonConsoleState::getInstance();
                if (action != "cancel") {
                    return std::unexpected("Action '" + action + "' is not supported for editor.python");
                }
                if (!console.isScriptRunning()) {
                    return std::unexpected("Editor script is not running");
                }
                console.interruptScript();
                return {};
            }

            if (job_id == "training.main") {
                auto* const trainer = viewer_impl->getTrainerManager();
                if (!trainer) {
                    return std::unexpected("Trainer manager is not initialized");
                }
                if (action == "start") {
                    if (auto result = viewer_impl->startTraining(); !result) {
                        return std::unexpected(result.error());
                    }
                    return {};
                }
                if (action == "pause") {
                    if (!trainer->canPause()) {
                        return std::unexpected("Training cannot be paused in the current state");
                    }
                    trainer->pauseTraining();
                    return {};
                }
                if (action == "resume") {
                    if (!trainer->canResume()) {
                        return std::unexpected("Training cannot be resumed in the current state");
                    }
                    trainer->resumeTraining();
                    return {};
                }
                if (action == "cancel") {
                    if (!trainer->canStop()) {
                        return std::unexpected("Training cannot be stopped in the current state");
                    }
                    trainer->stopTraining();
                    return {};
                }
                return std::unexpected("Action '" + action + "' is not supported for training.main");
            }

            if (job_id == "export.scene") {
                if (action != "cancel") {
                    return std::unexpected("Action '" + action + "' is not supported for export.scene");
                }
                if (!gui || !gui->asyncTasks().isExporting()) {
                    return std::unexpected("Scene export is not running");
                }
                gui->asyncTasks().cancelExport();
                return {};
            }

            if (job_id == "import.dataset") {
                if (action != "dismiss") {
                    return std::unexpected("Action '" + action + "' is not supported for import.dataset");
                }
                if (!gui || !gui->asyncTasks().isImportCompletionShowing()) {
                    return std::unexpected("Import completion overlay is not showing");
                }
                gui->asyncTasks().dismissImport();
                return {};
            }

            if (job_id == "export.video") {
                if (action != "cancel") {
                    return std::unexpected("Action '" + action + "' is not supported for export.video");
                }
                if (!gui || !gui->asyncTasks().isExportingVideo()) {
                    return std::unexpected("Video export is not running");
                }
                gui->asyncTasks().cancelVideoExport();
                return {};
            }

            if (job_id == "mesh2splat") {
                return std::unexpected("mesh2splat does not currently support runtime control");
            }

            if (job_id == "operator.modal") {
                if (action != "cancel") {
                    return std::unexpected("Action '" + action + "' is not supported for operator.modal");
                }
                if (!vis::op::operators().hasModalOperator()) {
                    return std::unexpected("No modal operator is active");
                }
                vis::op::operators().cancelModalOperator();
                return {};
            }

            return std::unexpected("Unsupported runtime job id: " + job_id);
        }

    } // namespace

    void register_generic_gui_runtime_tools(mcp::ToolRegistry& registry,
                                            vis::Visualizer* viewer) {
        RuntimeEventJournal::instance();

        registry.register_tool(
            mcp::McpTool{
                .name = "runtime.job.list",
                .description = "List normalized long-running runtime jobs and their current status",
                .input_schema = {
                    .type = "object",
                    .properties = json{
                        {"include_output", json{{"type", "boolean"}, {"description", "Include job output snapshots when available (default: false)"}}},
                        {"output_max_chars", json{{"type", "integer"}, {"description", "Maximum output characters to include for text-producing jobs (default: 20000)"}}},
                        {"output_tail", json{{"type", "boolean"}, {"description", "Return the newest output when truncating text output (default: true)"}}}},
                    .required = {}},
                .metadata = mcp::McpToolMetadata{
                    .category = "runtime",
                    .kind = "query",
                    .runtime = "gui",
                    .thread_affinity = "gui_thread",
                }},
            [viewer](const json& args) -> json {
                const bool include_output = args.value("include_output", false);
                const size_t output_max_chars =
                    static_cast<size_t>(std::max(0, args.value("output_max_chars", 20000)));
                const bool output_tail = args.value("output_tail", true);

                return post_and_wait(viewer, [viewer, include_output, output_max_chars, output_tail]() -> json {
                    auto payload = list_jobs_payload_on_gui(viewer, include_output, output_max_chars, output_tail);
                    if (!payload) {
                        return json{{"error", payload.error()}};
                    }
                    (*payload)["success"] = true;
                    (*payload)["event_count"] = static_cast<int64_t>(RuntimeEventJournal::instance().size());
                    return *payload;
                });
            });

        registry.register_tool(
            mcp::McpTool{
                .name = "runtime.job.describe",
                .description = "Describe one normalized runtime job, including output snapshots when available",
                .input_schema = {
                    .type = "object",
                    .properties = json{
                        {"job_id", json{{"type", "string"}, {"description", "Runtime job id"}}},
                        {"include_output", json{{"type", "boolean"}, {"description", "Include output snapshots when available (default: true)"}}},
                        {"output_max_chars", json{{"type", "integer"}, {"description", "Maximum output characters to include for text-producing jobs (default: 20000)"}}},
                        {"output_tail", json{{"type", "boolean"}, {"description", "Return the newest output when truncating text output (default: true)"}}}},
                    .required = {"job_id"}},
                .metadata = mcp::McpToolMetadata{
                    .category = "runtime",
                    .kind = "query",
                    .runtime = "gui",
                    .thread_affinity = "gui_thread",
                }},
            [viewer](const json& args) -> json {
                const std::string job_id = args["job_id"].get<std::string>();
                const bool include_output = args.value("include_output", true);
                const size_t output_max_chars =
                    static_cast<size_t>(std::max(0, args.value("output_max_chars", 20000)));
                const bool output_tail = args.value("output_tail", true);

                return post_and_wait(viewer, [viewer, job_id, include_output, output_max_chars, output_tail]() -> json {
                    auto payload = describe_job_payload_on_gui(
                        viewer,
                        job_id,
                        include_output,
                        output_max_chars,
                        output_tail);
                    if (!payload) {
                        return json{{"error", payload.error()}};
                    }
                    (*payload)["success"] = true;
                    return *payload;
                });
            });

        registry.register_tool(
            mcp::McpTool{
                .name = "runtime.job.wait",
                .description = "Poll a runtime job until it becomes active, inactive, or changes state/output",
                .input_schema = {
                    .type = "object",
                    .properties = json{
                        {"job_id", json{{"type", "string"}, {"description", "Runtime job id"}}},
                        {"until", json{{"type", "string"}, {"enum", json::array({"inactive", "active", "changed"})}, {"description", "Wait condition (default: inactive)"}}},
                        {"timeout_ms", json{{"type", "integer"}, {"description", "Maximum wait time before returning (default: 2000)"}}},
                        {"poll_interval_ms", json{{"type", "integer"}, {"description", "Polling interval between checks (default: 100)"}}},
                        {"include_output", json{{"type", "boolean"}, {"description", "Include output snapshots when available (default: true)"}}},
                        {"output_max_chars", json{{"type", "integer"}, {"description", "Maximum output characters to include for text-producing jobs (default: 20000)"}}},
                        {"output_tail", json{{"type", "boolean"}, {"description", "Return the newest output when truncating text output (default: true)"}}}},
                    .required = {"job_id"}},
                .metadata = mcp::McpToolMetadata{
                    .category = "runtime",
                    .kind = "query",
                    .runtime = "gui",
                    .thread_affinity = "gui_thread",
                    .long_running = true,
                }},
            [viewer](const json& args) -> json {
                const std::string job_id = args["job_id"].get<std::string>();
                const std::string until = args.value("until", "inactive");
                if (until != "inactive" && until != "active" && until != "changed") {
                    return json{{"error", "Unsupported wait condition: " + until}};
                }

                const int timeout_ms = std::max(0, args.value("timeout_ms", 2000));
                const int poll_interval_ms = std::max(10, args.value("poll_interval_ms", 100));
                const bool include_output = args.value("include_output", true);
                const size_t output_max_chars =
                    static_cast<size_t>(std::max(0, args.value("output_max_chars", 20000)));
                const bool output_tail = args.value("output_tail", true);

                auto describe = [&]() -> std::expected<json, std::string> {
                    return post_and_wait(viewer, [viewer, &job_id, include_output, output_max_chars, output_tail]() {
                        return describe_job_payload_on_gui(
                            viewer,
                            job_id,
                            include_output,
                            output_max_chars,
                            output_tail);
                    });
                };

                auto baseline = describe();
                if (!baseline) {
                    return json{{"error", baseline.error()}};
                }

                const auto start = std::chrono::steady_clock::now();
                while (true) {
                    auto current = describe();
                    if (!current) {
                        return json{{"error", current.error()}};
                    }
                    if (wait_condition_met(*baseline, *current, until)) {
                        (*current)["success"] = true;
                        (*current)["until"] = until;
                        (*current)["timed_out"] = false;
                        (*current)["wait_ms"] = std::chrono::duration_cast<std::chrono::milliseconds>(
                                                    std::chrono::steady_clock::now() - start)
                                                    .count();
                        return *current;
                    }

                    const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                        std::chrono::steady_clock::now() - start);
                    if (elapsed.count() >= timeout_ms) {
                        (*current)["success"] = true;
                        (*current)["until"] = until;
                        (*current)["timed_out"] = true;
                        (*current)["wait_ms"] = elapsed.count();
                        return *current;
                    }

                    std::this_thread::sleep_for(std::chrono::milliseconds(poll_interval_ms));
                }
            });

        registry.register_tool(
            mcp::McpTool{
                .name = "runtime.job.control",
                .description = "Apply a normalized control action to a runtime job",
                .input_schema = {
                    .type = "object",
                    .properties = json{
                        {"job_id", json{{"type", "string"}, {"description", "Runtime job id"}}},
                        {"action", json{{"type", "string"}, {"enum", json::array({"start", "pause", "resume", "cancel", "dismiss"})}, {"description", "Control action"}}},
                        {"include_output", json{{"type", "boolean"}, {"description", "Include output snapshots when available in the returned job description (default: true)"}}},
                        {"output_max_chars", json{{"type", "integer"}, {"description", "Maximum output characters to include for text-producing jobs (default: 20000)"}}},
                        {"output_tail", json{{"type", "boolean"}, {"description", "Return the newest output when truncating text output (default: true)"}}}},
                    .required = {"job_id", "action"}},
                .metadata = mcp::McpToolMetadata{
                    .category = "runtime",
                    .kind = "command",
                    .runtime = "gui",
                    .thread_affinity = "gui_thread",
                    .destructive = true,
                    .long_running = true,
                }},
            [viewer](const json& args) -> json {
                const std::string job_id = args["job_id"].get<std::string>();
                const std::string action = args["action"].get<std::string>();
                const bool include_output = args.value("include_output", true);
                const size_t output_max_chars =
                    static_cast<size_t>(std::max(0, args.value("output_max_chars", 20000)));
                const bool output_tail = args.value("output_tail", true);

                return post_and_wait(viewer, [viewer, job_id, action, include_output, output_max_chars, output_tail]() -> json {
                    if (auto result = control_job_on_gui(viewer, job_id, action); !result) {
                        return json{{"error", result.error()}};
                    }

                    auto payload = describe_job_payload_on_gui(
                        viewer,
                        job_id,
                        include_output,
                        output_max_chars,
                        output_tail);
                    if (!payload) {
                        return json{{"error", payload.error()}};
                    }
                    (*payload)["success"] = true;
                    (*payload)["action"] = action;
                    return *payload;
                });
            });

        registry.register_tool(
            mcp::McpTool{
                .name = "runtime.events.tail",
                .description = "Read recent retained runtime events without creating an explicit subscription",
                .input_schema = {
                    .type = "object",
                    .properties = json{
                        {"types", json{{"type", "array"}, {"items", json{{"type", "string"}}}, {"description", "Optional event type filters"}}},
                        {"max_events", json{{"type", "integer"}, {"description", "Maximum retained events to return (default: 100)"}}}},
                    .required = {}},
                .metadata = mcp::McpToolMetadata{
                    .category = "runtime",
                    .kind = "query",
                    .runtime = "process",
                }},
            [](const json& args) -> json {
                auto types = optional_string_array_arg(args, "types");
                if (!types) {
                    return json{{"error", types.error()}};
                }
                const size_t max_events =
                    static_cast<size_t>(std::max(1, args.value("max_events", 100)));
                return RuntimeEventJournal::instance().tail_payload(*types, max_events);
            });

        registry.register_tool(
            mcp::McpTool{
                .name = "runtime.events.clear",
                .description = "Clear retained runtime journal events, optionally limited to specific event types",
                .input_schema = {
                    .type = "object",
                    .properties = json{
                        {"types", json{{"type", "array"}, {"items", json{{"type", "string"}}}, {"description", "Optional event type filters; omit to clear all retained events"}}}},
                    .required = {}},
                .metadata = mcp::McpToolMetadata{
                    .category = "runtime",
                    .kind = "command",
                    .runtime = "process",
                    .destructive = true,
                }},
            [](const json& args) -> json {
                auto types = optional_string_array_arg(args, "types");
                if (!types) {
                    return json{{"error", types.error()}};
                }
                return RuntimeEventJournal::instance().clear_payload(*types);
            });
    }

    void register_generic_gui_runtime_resources(mcp::ResourceRegistry& registry,
                                                vis::Visualizer* viewer) {
        RuntimeEventJournal::instance();

        registry.register_resource(
            mcp::McpResource{
                .uri = "lichtfeld://runtime/catalog",
                .name = "Runtime Catalog",
                .description = "Discoverable runtime jobs, event types, and resource URIs",
                .mime_type = "application/json"},
            [](const std::string& uri) -> std::expected<std::vector<McpResourceContent>, std::string> {
                return single_json_resource(uri, runtime_catalog_json());
            });

        registry.register_resource(
            mcp::McpResource{
                .uri = "lichtfeld://runtime/state",
                .name = "Runtime State",
                .description = "Normalized runtime job summary plus retained runtime event count",
                .mime_type = "application/json"},
            [viewer](const std::string& uri) -> std::expected<std::vector<McpResourceContent>, std::string> {
                return post_and_wait(viewer, [viewer, uri]() -> std::expected<std::vector<McpResourceContent>, std::string> {
                    auto payload = list_jobs_payload_on_gui(viewer, false, 0, true);
                    if (!payload) {
                        return std::unexpected(payload.error());
                    }
                    (*payload)["event_count"] = static_cast<int64_t>(RuntimeEventJournal::instance().size());
                    return single_json_resource(uri, *payload);
                });
            });

        registry.register_resource(
            mcp::McpResource{
                .uri = "lichtfeld://runtime/jobs",
                .name = "Runtime Jobs",
                .description = "Normalized runtime jobs without large output snapshots",
                .mime_type = "application/json"},
            [viewer](const std::string& uri) -> std::expected<std::vector<McpResourceContent>, std::string> {
                return post_and_wait(viewer, [viewer, uri]() -> std::expected<std::vector<McpResourceContent>, std::string> {
                    auto payload = list_jobs_payload_on_gui(viewer, false, 0, true);
                    if (!payload) {
                        return std::unexpected(payload.error());
                    }
                    return single_json_resource(uri, *payload);
                });
            });

        registry.register_resource_prefix(
            "lichtfeld://runtime/jobs/",
            [viewer](const std::string& uri) -> std::expected<std::vector<McpResourceContent>, std::string> {
                constexpr std::string_view prefix = "lichtfeld://runtime/jobs/";
                const auto job_id = uri.substr(prefix.size());
                if (job_id.empty()) {
                    return std::unexpected("Runtime job URI must include an id");
                }

                return post_and_wait(viewer, [viewer, uri, job_id]() -> std::expected<std::vector<McpResourceContent>, std::string> {
                    auto payload = describe_job_payload_on_gui(viewer, job_id, true, 20000, true);
                    if (!payload) {
                        return std::unexpected(payload.error());
                    }
                    return single_json_resource(uri, *payload);
                });
            });

        registry.register_resource(
            mcp::McpResource{
                .uri = "lichtfeld://runtime/events",
                .name = "Runtime Events",
                .description = "Recent retained runtime events",
                .mime_type = "application/json"},
            [](const std::string& uri) -> std::expected<std::vector<McpResourceContent>, std::string> {
                return single_json_resource(
                    uri,
                    RuntimeEventJournal::instance().tail_payload(std::vector<std::string>{}, 200));
            });

        registry.register_resource_prefix(
            "lichtfeld://runtime/events/",
            [](const std::string& uri) -> std::expected<std::vector<McpResourceContent>, std::string> {
                constexpr std::string_view prefix = "lichtfeld://runtime/events/";
                const auto event_type = uri.substr(prefix.size());
                if (event_type.empty()) {
                    return std::unexpected("Runtime event URI must include an event type");
                }
                if (!is_supported_runtime_event_type(event_type)) {
                    return std::unexpected("Unsupported runtime event type: " + event_type);
                }

                return single_json_resource(
                    uri,
                    RuntimeEventJournal::instance().tail_payload(
                        std::vector<std::string>{event_type},
                        200));
            });
    }

} // namespace lfs::app
