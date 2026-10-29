/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "shaders/shader_api.hpp"
#include <algorithm>
#include <cmath>

namespace lfs::vis::shaders {
    namespace {
        using json = nlohmann::json;
        Error invalid(std::string message) {
            return make_error(ErrorInit{.code = ErrorCode::InvalidArgument,
                                        .domain = ErrorDomain::Rendering,
                                        .user_message = std::move(message),
                                        .detection = LFS_SOURCE_SITE_CURRENT()});
        }

        json bindingsJson(const std::vector<ShaderBinding>& bindings, std::optional<core::Uuid> asset = {}) {
            json result = json::array();
            for (const auto& binding : bindings) {
                if (asset && binding.asset != *asset)
                    continue;
                auto item = shaderTarget(binding.target);
                if (!asset)
                    item["shader_id"] = binding.asset.to_string();
                result.push_back(std::move(item));
            }
            return result;
        }

        json membersJson(const std::vector<ShaderMember>& members) {
            json result = json::array();
            for (const auto& member : members)
                result.push_back({{"shader_id", member.asset.to_string()}, {"source_revision", member.revision}});
            return result;
        }

        json diagnosticsJson(const CompileDiagnostics& diagnostics, std::optional<std::uint64_t> revision = {}) {
            json result = json::array();
            for (const auto& d : diagnostics.entries) {
                const char* severity = "note";
                if (d.severity == DiagnosticSeverity::Error)
                    severity = "error";
                else if (d.severity == DiagnosticSeverity::Warning)
                    severity = "warning";
                json item{{"severity", severity}, {"code", d.code}, {"message", d.message}, {"file", d.file}};
                if (d.line)
                    item["line"] = d.line;
                if (d.column)
                    item["column"] = d.column;
                if (revision)
                    item["source_revision"] = *revision;
                result.push_back(std::move(item));
            }
            return result;
        }

        json assetJson(const ShaderAssetView& view, const ShaderInspection& state, bool source) {
            const auto& asset = view.asset;
            json result{{"shader_id", asset.id.to_string()},
                        {"name", asset.name},
                        {"source_revision", asset.revision},
                        {"status", shaderStatus(asset.id, view.status)},
                        {"bindings", bindingsJson(state.bindings, asset.id)},
                        {"displayed_bindings", bindingsJson(state.displayed_bindings, asset.id)}};
            if (source)
                result["source"] = asset.source;
            return result;
        }

        Result<core::Uuid> uuid(const json& value) {
            if (!value.is_string())
                return invalid("Expected a UUID string.");
            const auto id = core::Uuid::from_string(value.get<std::string>());
            if (!id || id->is_nil())
                return invalid("Expected a non-nil UUID string.");
            return *id;
        }
    } // namespace

    json shaderTarget(const ShaderSlot& slot) {
        const char* kind = "reveal";
        if (slot.kind == ShaderSlot::Kind::Global)
            kind = "global";
        else if (slot.kind == ShaderSlot::Kind::Node)
            kind = "node";
        json result{{"kind", kind}};
        if (slot.kind == ShaderSlot::Kind::Node)
            result["node_uuid"] = slot.node.to_string();
        return result;
    }

    Result<ShaderSlot> parseShaderTarget(const json& t) {
        if (!t.is_object())
            return invalid("Target must be an object.");
        ShaderSlot target;

        if (!t.contains("kind") || !t["kind"].is_string())
            return invalid("Target requires kind.");
        for (const auto& [key, value] : t.items())
            if (key != "kind" && key != "node_uuid")
                return invalid("Unknown target field: " + key);
        const auto kind = t["kind"].get<std::string>();
        if (kind == "node") {
            if (!t.contains("node_uuid"))
                return invalid("Node target requires node_uuid.");
            auto node = uuid(t["node_uuid"]);
            if (!node)
                return node.error();
            target = {ShaderSlot::Kind::Node, *node};
        } else {
            if ((kind != "global" && kind != "reveal") || t.contains("node_uuid"))
                return invalid("Invalid target.");
            target.kind = kind == "global" ? ShaderSlot::Kind::Global : ShaderSlot::Kind::Reveal;
        }
        return target;
    }

    std::vector<ShaderTool> shaderTools() {
        const json string{{"type", "string"}}, boolean{{"type", "boolean"}},
            revision{{"type", "integer"}, {"minimum", 0}};
        const json target{
            {"type", "object"},
            {"properties",
             {{"kind", {{"type", "string"}, {"enum", {"global", "node", "reveal"}}}}, {"node_uuid", string}}},
            {"required", {"kind"}}};
        return {{"list", "List shader assets and bindings", json::object(), {}, true},

                {"get",
                 "Read a shader asset and its current status",
                 {{"shader_id", string}, {"include_source", boolean}},
                 {"shader_id"},
                 true},

                {"set_source",
                 "Create or update shader source; linking follows draw demand",
                 {
                     {"source", string},
                     {"shader_id", string},
                     {"name", string},
                     {"expected_revision", revision},
                 },
                 {"source"}},

                {"bind",
                 "Bind a shader to a target; acceptance does not imply display",
                 {{"shader_id", string}, {"target", target}},
                 {"shader_id", "target"}},

                {"unbind", "Remove a shader binding", {{"target", target}, {"shader_id", string}}, {"target"}},

                {"set_param",
                 "Update a reflected parameter without linking",
                 {{"shader_id", string},
                  {"name", string},
                  {"value",
                   {{"oneOf",
                     json::array(
                         {{{"type", "number"}},
                          {{"type", "array"}, {"minItems", 2}, {"maxItems", 4}, {"items", {{"type", "number"}}}}})}}},
                  {"expected_revision", revision}},
                 {"shader_id", "name", "value"}},

                {"remove", "Remove an unbound shader asset", {{"shader_id", string}}, {"shader_id"}}};
    }

    json shaderCatalog() {
        json tools = json::array();
        for (const auto& tool : shaderTools())
            tools.push_back("shader_" + tool.name);
        return {{"compiler_build", SlangCompiler::compiler_build()},
                {"language", "slang"},
                {"abi_hash", SlangCompiler::abi_hash()},
                {"primitive_kinds", {"splat"}},
                {"parameter_types", {"float", "float2", "float3", "float4"}},
                {"max_source_bytes", kMaxShaderSourceBytes},
                {"max_parameters", 16},
                {"max_bound_assets", kMaxBoundShaders},
                {"binding_order", {"global", "node", "reveal"}},
                {"supported_targets", {"global", "node", "reveal"}},
                {"tools", tools},
                {"job_id", "shader.compile"},
                {"resources",
                 {"lichtfeld://shaders/catalog", "lichtfeld://shaders/state", "lichtfeld://shaders/assets/<uuid>"}},
                {"linking", "draw_driven"},
                {"displayed_bindings", "Published snapshot slots; invisible objects may not draw"}};
    }

    json shaderStatus(const core::Uuid& id, const ShaderStatus& s) {
        static constexpr const char* states[] = {"unbound", "pending", "applied", "blocked", "failed"};
        json parameters = json::array(), values = json::object(), blocked = json::array();
        for (std::size_t i = 0; i < s.parameters.size(); ++i) {
            const auto& p = s.parameters[i];
            parameters.push_back({{"name", p.name},
                                  {"components", p.components},
                                  {"minimum", p.minimum},
                                  {"maximum", p.maximum},
                                  {"initial", p.initial}});
            if (p.components == 1)
                values[p.name] = s.values[i][0];
            else {
                values[p.name] = json::array();
                for (std::uint32_t c = 0; c < p.components; ++c)
                    values[p.name].push_back(s.values[i][c]);
            }
        }
        for (const auto& asset : s.blocked_by)
            blocked.push_back(asset.to_string());
        return {{"shader_id", id.to_string()},
                {"source_revision", s.revision},
                {"schema_revision", s.schema_revision},
                {"displayed_revision", s.displayed_revision},
                {"state", states[static_cast<int>(s.state)]},
                {"compiling", s.compiling},
                {"pending", s.pending},
                {"failed", s.failed},
                {"blocked_by", blocked},
                {"links", s.links},
                {"last_good", s.displayed_revision > 0 && s.displayed_revision != s.revision},
                {"diagnostics", diagnosticsJson(s.diagnostics, s.diagnostics_revision)},
                {"raw_diagnostics", s.diagnostics.raw},
                {"parameters", parameters},
                {"values", values}};
    }

    json shaderJob(const ShaderInspection& state) {
        json work = json::array();
        for (const auto& item : state.work)
            work.push_back(
                {{"members", membersJson(item.members)}, {"host", item.host}, {"compiling", item.compiling}});
        return {
            {"id", "shader.compile"},
            {"label", "Shaders"},
            {"kind", "shader"},
            {"active", !work.empty()},
            {"status", work.empty() ? "idle" : "running"},
            {"stage", ""},
            {"progress", nullptr},
            {"cancel_supported", false},
            {"dismiss_supported", false},
            {"actions", {{"start", false}, {"pause", false}, {"resume", false}, {"cancel", false}, {"dismiss", false}}},
            {"details", {{"work", work}}}};
    }

    json shaderState(const ShaderInspection& state) {
        json assets = json::array();
        for (const auto& view : state.assets)
            assets.push_back(assetJson(view, state, false));
        return {{"assets", assets},
                {"bindings", bindingsJson(state.bindings)},
                {"displayed_bindings", bindingsJson(state.displayed_bindings)},
                {"job", shaderJob(state)}};
    }

    std::string_view shaderEventName(ShaderEvent::Kind kind) {
        static constexpr std::string_view names[] = {"shader.compile.started", "shader.compile.completed",
                                                     "shader.compile.failed", "shader.bindings.changed",
                                                     "shader.applied"};
        return names[static_cast<int>(kind)];
    }

    json shaderEvent(const ShaderEvent& event) {
        if (event.kind == ShaderEvent::Kind::BindingsChanged)
            return {{"bindings", bindingsJson(event.bindings)}};
        if (event.kind == ShaderEvent::Kind::Applied)
            return {{"members", membersJson(event.members)}, {"bindings", bindingsJson(event.bindings)}};
        json result{{"members", membersJson(event.members)},
                    {"stage", event.host.empty() ? "publication" : "link"},
                    {"diagnostics", diagnosticsJson(event.diagnostics)},
                    {"raw_diagnostics", event.diagnostics.raw}};
        if (!event.host.empty())
            result["host"] = event.host;
        return result;
    }

    Result<json> shaderOperation(ShaderService& service, std::string_view operation, const json& args) {
        const auto tools = shaderTools();
        const auto tool = std::ranges::find(tools, operation, &ShaderTool::name);
        if (tool == tools.end() || !args.is_object())
            return invalid("Unknown shader operation or invalid arguments.");
        for (const auto& key : tool->required)
            if (!args.contains(key))
                return invalid("Missing field: " + key);
        for (const auto& [key, value] : args.items()) {
            if (!tool->properties.contains(key))
                return invalid("Unknown field: " + key);
            const auto type = tool->properties[key].value("type", "");
            if ((type == "string" && !value.is_string()) || (type == "boolean" && !value.is_boolean()) ||
                (type == "object" && !value.is_object()) ||
                (type == "integer" &&
                 (!value.is_number_integer() ||
                  (value.is_number_integer() && !value.is_number_unsigned() && value.get<std::int64_t>() < 0))))
                return invalid("Invalid field type: " + key);
        }
        std::optional<core::Uuid> id;
        if (args.contains("shader_id")) {
            auto parsed = uuid(args["shader_id"]);
            if (!parsed)
                return parsed.error();
            id = *parsed;
        }
        std::optional<std::uint64_t> revision;
        if (args.contains("expected_revision"))
            revision = args["expected_revision"].get<std::uint64_t>();
        if (revision && !id)
            return invalid("expected_revision requires shader_id.");
        ShaderSlot target;
        if (args.contains("target")) {
            auto parsed = parseShaderTarget(args["target"]);
            if (!parsed)
                return parsed.error();
            target = *parsed;
        }
        if (tool->query) {
            if (operation == "list")
                return json{{"assets", shaderState(service.inspect(false))["assets"]}, {"capabilities", shaderCatalog()}};
            const bool source = args.value("include_source", true);
            auto state = service.asset(*id, source);
            if (!state)
                return state.error();
            return assetJson(state->assets.front(), *state, source);
        }
        ShaderCommand command;
        if (operation == "set_source") {
            std::optional<std::string> name;
            if (args.contains("name"))
                name = args["name"].get<std::string>();
            command = SetShaderSource{id, args["source"].get<std::string>(), name, revision};
        } else if (operation == "bind")
            command = BindShader{target, *id};
        else if (operation == "unbind")
            command = UnbindShader{target, id};
        else if (operation == "remove")
            command = RemoveShader{*id};
        else {
            const auto& value = args["value"];
            std::vector<float> values;
            if (value.is_number())
                values.push_back(value.get<float>());
            else if (value.is_array() && value.size() >= 2 && value.size() <= 4) {
                for (const auto& item : value) {
                    if (!item.is_number())
                        return invalid("Parameter components must be numbers.");
                    values.push_back(item.get<float>());
                }
            } else
                return invalid("Parameter value must be a number or 2-4 numbers.");
            if (!std::ranges::all_of(values, [](float v) { return std::isfinite(v); }))
                return invalid("Parameter components must be finite.");
            command = SetShaderParameter{*id, args["name"].get<std::string>(), std::move(values), revision};
        }
        auto changed = service.command(command);
        if (!changed)
            return changed.error();
        json result{{"success", true}, {"shader_id", changed->to_string()}};
        if (operation == "remove" || operation == "unbind")
            result["removed"] = true;
        if (operation == "bind" || operation == "unbind")
            result["target"] = shaderTarget(target);
        if (operation != "remove") {
            const auto status = service.status(*changed);
            if (!status)
                return status.error();
            if (operation == "bind")
                result["status"] = shaderStatus(*changed, *status);
            if (operation == "set_source" || operation == "set_param")
                result["source_revision"] = status->revision;
            if (operation == "set_param") {
                result["name"] = args["name"];
                result["value"] = args["value"];
                result["schema_revision"] = status->schema_revision;
            }
        }
        return result;
    }
} // namespace lfs::vis::shaders
