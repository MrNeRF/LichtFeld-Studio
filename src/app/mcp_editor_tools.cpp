/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "app/mcp_editor_tools.hpp"
#include "app/include/app/mcp_app_utils.hpp"
#include "core/error_envelope.hpp"
#include "core/events.hpp"
#include "gui/editor/python_editor.hpp"
#include "gui/panels/python_console_panel.hpp"
#include "mcp/mcp_tools.hpp"
#include "shaders/shader_api.hpp"
#include "visualizer/visualizer_impl.hpp"
#include <limits>

namespace lfs::app {
    namespace {
        using json = nlohmann::json;
        using vis::editor::DocumentKind;
        using vis::editor::EditorDocumentState;
        using vis::gui::panels::PythonConsoleState;

        Error invalid(const std::string& message) {
            return make_error(ErrorInit{.code = ErrorCode::InvalidArgument,
                                        .domain = ErrorDomain::MCP,
                                        .user_message = message,
                                        .detection = LFS_SOURCE_SITE_CURRENT()});
        }

        json textPayload(const std::string& text, int limit, const bool tail, const bool shader) {
            if (shader)
                limit = limit < 0 ? 20000 : std::min(limit, 1024 * 1024);
            const auto count = limit < 0 ? text.size() : std::min(text.size(), static_cast<std::size_t>(limit));
            auto begin = tail ? text.size() - count : 0;
            auto end = tail ? text.size() : count;
            const auto continuation = [&](std::size_t pos) { return (text[pos] & 0xc0) == 0x80; };
            while (begin < end && continuation(begin))
                ++begin;
            while (end > begin && end < text.size() && continuation(end))
                --end;
            return {{"text", text.substr(begin, end - begin)},
                    {"truncated", end - begin < text.size()},
                    {"total_chars", text.size()}};
        }

        json documentJson(const EditorDocumentState& state, const PythonConsoleState& console) {
            const auto& doc = state.document;
            json owned = json::array();
            for (const auto& target : state.owned_bindings)
                owned.push_back(vis::shaders::shaderTarget(target));
            const bool shader = doc.kind == DocumentKind::Shader;
            const bool running = shader ? state.shader_status &&
                                              (state.shader_status->compiling || !state.shader_status->pending.empty())
                                        : console.isScriptRunning();
            json result{{"success", true},
                        {"document_id", doc.id.to_string()},
                        {"document_kind", shader ? "shader" : "python"},
                        {"source_revision", doc.source_revision},
                        {"shader_id", doc.shader_asset ? json(doc.shader_asset->to_string()) : json(nullptr)},
                        {"asset_revision", doc.asset_revision},
                        {"target", vis::shaders::shaderTarget(doc.target)},
                        {"owns_binding", !owned.empty()},
                        {"owned_bindings", owned},
                        {"modified", state.modified},
                        {"running", running},
                        {"applied", state.applied}};
            if (state.shader_status)
                result["shader_status"] = vis::shaders::shaderStatus(*doc.shader_asset, *state.shader_status);
            return result;
        }

        struct Observation {
            vis::editor::EditorDocument document;
            std::uint64_t script_run = 0;
            json response;
        };

        Observation observe(const EditorDocumentState& state, PythonConsoleState& console, const json& args,
                            const Observation* origin = nullptr) {
            Observation result{state.document, console.scriptRun(), documentJson(state, console)};
            const auto& key = origin ? origin->document : state.document;
            const bool shader = key.kind == DocumentKind::Shader;
            std::string output, outcome;
            bool completed = !console.isScriptRunning();
            if (shader) {
                completed = true;
                if (state.document.kind != key.kind || state.document.source_revision != key.source_revision ||
                    state.document.shader_asset != key.shader_asset ||
                    state.document.asset_revision != key.asset_revision)
                    outcome = "superseded";
                else if (!state.shader_status)
                    outcome = "unbound";
                else if (state.shader_status->revision != key.asset_revision)
                    outcome = "superseded";
                else if (state.owned_bindings.empty())
                    outcome = "unbound";
                else {
                    outcome = vis::shaders::shaderOutcome(*state.shader_status, key.asset_revision, state.applied);
                    completed = !outcome.empty();
                }
                if (state.shader_status && outcome != "superseded" &&
                    state.shader_status->diagnostics_revision == key.asset_revision)
                    output = state.shader_status->diagnostics.raw;
            } else if (origin && origin->script_run != console.scriptRun()) {
                outcome = "superseded";
                completed = true;
            } else {
                output = console.getOutputText();
            }
            if (!outcome.empty())
                result.response["outcome"] = outcome;
            if (outcome == "superseded") {
                result.response["running"] = false;
                result.response["applied"] = false;
            } else if (!shader) {
                result.response["running"] = console.isScriptRunning();
            }
            result.response["completed"] = completed;
            result.response["saw_output"] = !output.empty();
            result.response["timed_out"] = false;
            result.response["output"] =
                textPayload(output, args.value("output_max_chars", 20000), args.value("output_tail", true), shader);
            return result;
        }

        // A project switch replaces editor tabs, including the one whose script asked for it. That script
        // runs on in the console, so its run stays observable by the closed document's id.
        bool runsClosedDocument(const PythonConsoleState& console, const core::Uuid& id) {
            return console.scriptDocument() == id;
        }
        EditorDocumentState closedScriptDocument(const core::Uuid& id) {
            return {.document = {.id = id, .kind = DocumentKind::Python}};
        }

        Result<Observation> execute(vis::VisualizerImpl& viewer, const std::string& operation, const json& args) {
            auto& console = PythonConsoleState::getInstance();
            auto* editor = console.getEditor();
            auto* service = viewer.getShaderService();
            if (!editor)
                return invalid("Editor is unavailable.");
            if (operation == "list_documents") {
                json documents = json::array();
                for (const auto& doc : editor->documents()) {
                    const auto state = editor->inspectDocument(doc.id, service);
                    if (!state)
                        return state.error();
                    documents.push_back(documentJson(*state, console));
                }
                return Observation{{},
                                   0,
                                   {{"success", true},
                                    {"active_document_id", editor->document().id.to_string()},
                                    {"documents", documents}}};
            }
            auto id = args.contains("document_id") ? *core::Uuid::from_string(args["document_id"].get<std::string>())
                                                   : editor->document().id;
            if (operation == "new_document") {
                const auto kind =
                    args.value("mode", "python") == "shader" ? DocumentKind::Shader : DocumentKind::Python;
                const auto text = args.value("code", "");
                if (kind == DocumentKind::Shader && text.size() > vis::shaders::kMaxShaderSourceBytes)
                    return invalid("Shader source exceeds 1 MiB.");
                id = editor->newDocument(text, kind);
            }
            auto state = editor->inspectDocument(id, service);
            if (!state) {
                if (operation != "wait")
                    return state.error();
                if (runsClosedDocument(console, id))
                    return observe(closedScriptDocument(id), console, args);
                vis::editor::EditorDocument closed;
                closed.id = id;
                return Observation{closed,
                                   0,
                                   {{"success", true},
                                    {"document_id", id.to_string()},
                                    {"outcome", "document_closed"},
                                    {"running", false},
                                    {"applied", false},
                                    {"completed", true},
                                    {"timed_out", false},
                                    {"saw_output", false},
                                    {"output", textPayload("", 0, false, false)}}};
            }
            const bool shader = state->document.kind == DocumentKind::Shader;
            if ((operation == "run" || operation == "wait") && shader &&
                (args.value("timeout_ms", 2000) < 0 || args.value("timeout_ms", 2000) > 30000))
                return invalid("Shader timeout_ms must be between 0 and 30000.");
            if (args.contains("asset_revision") && !shader)
                return invalid("asset_revision requires a shader document.");
            bool was_running = false;
            auto released = json::array();
            std::optional<std::size_t> chars;
            if (operation == "activate_document") {
                if (!editor->activateDocument(id))
                    return invalid("Editor document is closed or unknown.");
            } else if (operation == "set_mode") {
                std::optional<vis::shaders::ShaderAsset> asset;
                if (args.contains("shader_id")) {
                    if (args["mode"] != "shader" || !service)
                        return invalid("Shader association requires shader mode and a shader service.");
                    const auto asset_id = *core::Uuid::from_string(args["shader_id"].get<std::string>());
                    for (const auto& candidate : service->assets())
                        if (candidate.id == asset_id)
                            asset = candidate;
                    if (!asset)
                        return invalid("No such shader asset.");
                }
                const auto kind = args["mode"] == "shader" ? DocumentKind::Shader : DocumentKind::Python;
                if (auto changed = editor->setDocumentMode(id, kind, asset); !changed)
                    return changed.error();
            } else if (operation == "set_target") {
                const auto target = vis::shaders::parseShaderTarget(args["target"]);
                if (!target)
                    return target.error();
                if (!service)
                    return invalid("Shader service is unavailable.");
                if (auto changed = editor->setDocumentTarget(id, *target, *service); !changed)
                    return changed.error();
            } else if (operation == "run") {
                const std::optional<std::string> code =
                    args.contains("code") ? std::optional(args["code"].get<std::string>()) : std::nullopt;
                if (shader) {
                    if (!service)
                        return invalid("Shader service is unavailable.");
                    if (auto run = editor->runShader(*service, id, code); !run)
                        return run.error();
                    chars = code ? code->size() : state->text.size();
                } else {
                    if (console.isScriptRunning())
                        return invalid("A script is already running.");
                    const auto text = code.value_or(editor->getTextStripped(id));
                    if (text.empty())
                        return invalid("Editor is empty.");
                    if (code) {
                        if (auto edited = editor->setDocumentText(id, *code); !edited)
                            return edited.error();
                    }
                    console.runScriptAsync(text, id);
                    chars = text.size();
                }
            } else if (operation == "set_code") {
                const auto text = args["code"].get<std::string>();
                if (auto edited = editor->setDocumentText(id, text); !edited)
                    return edited.error();
                chars = text.size();
            } else if (operation == "interrupt") {
                was_running = documentJson(*state, console)["running"].get<bool>();
                if (shader && service) {
                    for (const auto& target : editor->stopShader(*service, id))
                        released.push_back(vis::shaders::shaderTarget(target));
                } else if (!shader && was_running) {
                    console.interruptScript();
                }
            }
            if ((operation == "run" || operation == "set_code") && args.value("show_console", true)) {
                core::events::cmd::ShowWindow{.window_name = "python_console", .show = true}.emit();
                if (!args.contains("document_id"))
                    console.focusEditor();
            }
            state = editor->inspectDocument(id, service);
            if (!state)
                return state.error();
            auto result = observe(*state, console, args);
            if (chars)
                result.response["chars"] = *chars;
            if (operation == "wait" && args.contains("asset_revision")) {
                result.document.asset_revision = args["asset_revision"].get<std::uint64_t>();
                result.response = observe(*state, console, args, &result).response;
            }
            if (operation == "get_code") {
                result.response.update(textPayload(state->text, args.value("max_chars", -1), false, false));
                if (!state->locator.empty() && !state->locator.contains("://"))
                    result.response["path"] = state->locator;
                auto markers = json::array();
                for (const auto& marker : state->markers)
                    markers.push_back({{"start_byte", marker.start},
                                       {"end_byte", marker.end},
                                       {"source_revision", state->document.source_revision}});
                result.response["markers"] = markers;
                result.response["marker_count"] = markers.size();
                result.response["diagnostics"] =
                    state->diagnostics.empty() ? json::array() : result.response["shader_status"]["diagnostics"];
            } else if (operation == "get_output") {
                json output_args{{"output_max_chars", args.value("max_chars", 20000)},
                                 {"output_tail", args.value("tail", true)}};
                result.response.update(observe(*state, console, output_args).response["output"]);
            } else if (operation == "interrupt") {
                result.response["was_running"] = was_running;
                result.response["released_bindings"] = released;
            }
            if (operation != "run" && operation != "wait") {
                for (const auto* field : {"output", "completed", "timed_out", "saw_output", "outcome"})
                    result.response.erase(field);
            }
            return result;
        }

        json waitFor(vis::VisualizerImpl* viewer, Observation origin, const json& args) {
            const bool completion = args.value("wait_for_completion", true);
            const bool output = args.value("wait_for_output", true);
            const int timeout = args.value("timeout_ms", 2000);
            const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(std::max(0, timeout));
            auto changes = viewer->getShaderService()->changes();
            auto generation = changes->generation.load(std::memory_order_acquire);
            const auto origin_generation = generation;
            auto result = origin.response;
            bool refresh = true;
            // A project switch cancels shader work. A Python run outlives it, as its own script may switch:
            // only the service closing ends that wait early.
            const auto cancellable_from = origin.document.kind == DocumentKind::Shader
                                              ? origin_generation
                                              : std::numeric_limits<std::uint64_t>::max();
            for (;;) {
                if (changes->cancelled(cancellable_from)) {
                    result.update({{"outcome", "cancelled"}, {"completed", true}, {"running", false}});
                    break;
                }
                const bool done = result.value("completed", false);
                if ((!completion || done) && (!output || result.value("saw_output", false) || done))
                    break;
                if (viewer->isOnViewerThread() || std::chrono::steady_clock::now() >= deadline) {
                    result["timed_out"] = true;
                    if (origin.document.kind == DocumentKind::Shader)
                        result["outcome"] = "timeout";
                    break;
                }
                if (!refresh)
                    changes->wait(generation, deadline);
                refresh = false;
                generation = changes->generation.load(std::memory_order_acquire);
                auto next = post_and_wait(viewer, [viewer, origin, args]() -> Result<json> {
                    auto& console = PythonConsoleState::getInstance();
                    auto* editor = console.getEditor();
                    const auto docs = editor ? editor->documents() : std::vector<vis::editor::EditorDocument>{};
                    if (!std::ranges::contains(docs, origin.document.id, &vis::editor::EditorDocument::id)) {
                        if (origin.document.kind == DocumentKind::Python && runsClosedDocument(console, origin.document.id))
                            return observe(closedScriptDocument(origin.document.id), console, args, &origin).response;
                        auto closed = origin.response;
                        closed.update({{"outcome", "document_closed"},
                                       {"completed", true},
                                       {"running", false},
                                       {"saw_output", false},
                                       {"output", textPayload("", 0, false, false)}});
                        return closed;
                    }
                    auto state = editor->inspectDocument(origin.document.id, viewer->getShaderService());
                    if (!state)
                        return state.error();
                    return observe(*state, console, args, &origin).response;
                });
                if (!next)
                    return {{"error", core::to_wire_envelope(next.error())}};
                result = *next;
            }
            if (origin.response.contains("chars"))
                result["chars"] = origin.response["chars"];
            result["wait_for_completion"] = completion;
            result["wait_for_output"] = output;
            result["timeout_ms"] = timeout;
            return result;
        }
    } // namespace

    void register_editor_tools(mcp::ToolRegistry& registry, vis::VisualizerImpl* viewer) {
        const json string{{"type", "string"}}, boolean{{"type", "boolean"}}, integer{{"type", "integer"}};
        const json mode{{"type", "string"}, {"enum", {"python", "shader"}}};
        const json target{
            {"type", "object"},
            {"properties",
             {{"kind", {{"type", "string"}, {"enum", {"global", "node", "reveal"}}}}, {"node_uuid", string}}},
            {"required", {"kind"}}};
        const json waits{{"wait_for_completion", boolean},
                         {"wait_for_output", boolean},
                         {"timeout_ms", integer},
                         {"output_max_chars", integer},
                         {"output_tail", boolean}};
        struct Tool {
            const char* name;
            const char* description;
            json properties;
            std::vector<std::string> required;
        };
        std::vector<Tool> tools{
            {"list_documents", "List editor documents without changing focus", json::object(), {}},
            {"new_document", "Create and activate an editor document", {{"mode", mode}, {"code", string}}, {}},
            {"activate_document", "Activate an existing editor document", {}, {"document_id"}},
            {"set_mode",
             "Change document language without executing or replacing text",
             {{"mode", mode}, {"shader_id", string}},
             {"mode"}},
            {"set_target", "Choose the target of the document's next Shader Run", {{"target", target}}, {"target"}},
            {"set_code",
             "Replace document text without executing it",
             {{"code", string}, {"show_console", boolean}},
             {"code"}},
            {"get_code", "Read document text and current diagnostic markers", {{"max_chars", integer}}, {}},
            {"get_output",
             "Read bounded Python console or shader compiler output",
             {{"max_chars", integer}, {"tail", boolean}},
             {}},
            {"is_running", "Observe document work separately from applied shader bindings", {}, {}},
            {"interrupt", "Interrupt Python or release this document's Shader Run bindings", {}, {}},
            {"run", "Run a document; Python wait defaults remain true with a 2000 ms timeout", waits, {}},
            {"wait", "Observe one document submission even if the active buffer changes", waits, {}}};
        for (auto& tool : tools) {
            const std::string name = tool.name;
            if (name == "run")
                tool.properties.update({{"code", string}, {"show_console", boolean}});
            if (name == "wait")
                tool.properties["asset_revision"] = {{"type", "integer"}, {"minimum", 0}};
            if (name != "list_documents" && name != "new_document")
                tool.properties["document_id"] = string;
            const bool query = name == "list_documents" || name == "get_code" || name == "get_output" ||
                               name == "is_running" || name == "wait";
            std::string kind = "mutation";
            if (query)
                kind = "query";
            else if (name == "run" || name == "interrupt")
                kind = "command";
            registry.register_tool(
                mcp::McpTool{
                    .name = "editor." + name,
                    .description = tool.description,
                    .input_schema = {.type = "object", .properties = tool.properties, .required = tool.required},
                    .metadata = {.category = "editor",
                                 .kind = kind,
                                 .runtime = "gui",
                                 .thread_affinity = "gui_thread",
                                 .long_running = name == "run" || name == "wait"}},
                [viewer, name, properties = tool.properties, required = tool.required](const json& args) -> json {
                    const auto validate = [&]() -> Status {
                        if (!args.is_object())
                            return Status::failure(invalid("Expected an argument object."));
                        for (const auto& field : required)
                            if (!args.contains(field))
                                return Status::failure(invalid("Missing field: " + field));
                        for (const auto& [field, value] : args.items()) {
                            if (!properties.contains(field))
                                return Status::failure(invalid("Unknown field: " + field));
                            const auto type = properties[field].value("type", "");
                            if ((type == "string" && !value.is_string()) ||
                                (type == "boolean" && !value.is_boolean()) ||
                                (type == "object" && !value.is_object()) ||
                                (type == "integer" && !value.is_number_integer()))
                                return Status::failure(invalid("Invalid field type: " + field));
                            if (field == "document_id" || field == "shader_id") {
                                const auto id = core::Uuid::from_string(value.get<std::string>());
                                if (!id || id->is_nil())
                                    return Status::failure(invalid("Invalid UUID: " + field));
                            }
                            if (type == "integer") {
                                const bool negative = !value.is_number_unsigned() && value.get<std::int64_t>() < 0;
                                const bool outside_int =
                                    value.is_number_unsigned()
                                        ? value.get<std::uint64_t>() > std::numeric_limits<int>::max()
                                        : value.get<std::int64_t>() < std::numeric_limits<int>::min() ||
                                              value.get<std::int64_t>() > std::numeric_limits<int>::max();
                                if (field == "asset_revision" ? negative : outside_int)
                                    return Status::failure(invalid("Integer out of range: " + field));
                            }
                        }
                        if (args.contains("mode") && args["mode"] != "python" && args["mode"] != "shader")
                            return Status::failure(invalid("Mode must be python or shader."));
                        return {};
                    };
                    if (auto valid = validate(); !valid)
                        return {{"error", core::to_wire_envelope(valid.error())}};
                    auto result = post_and_wait(viewer, [viewer, name, args] { return execute(*viewer, name, args); });
                    if (!result)
                        return {{"error", core::to_wire_envelope(result.error())}};
                    return name == "run" || name == "wait" ? waitFor(viewer, std::move(*result), args)
                                                           : result->response;
                });
        }
    }
} // namespace lfs::app
