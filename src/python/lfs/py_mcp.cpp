/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "py_mcp.hpp"

#include "core/base64.hpp"
#include "core/logger.hpp"
#include "mcp/mcp_protocol.hpp"
#include "mcp/mcp_tools.hpp"

#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>

namespace lfs::python {

    namespace {

        std::string python_type_to_json_type(const std::string& py_type) {
            if (py_type == "int")
                return "integer";
            if (py_type == "float")
                return "number";
            if (py_type == "bool")
                return "boolean";
            if (py_type == "str")
                return "string";
            if (py_type.starts_with("list[int]") || py_type.starts_with("List[int]"))
                return "array:integer";
            if (py_type.starts_with("list[float]") || py_type.starts_with("List[float]"))
                return "array:number";
            if (py_type.starts_with("list") || py_type.starts_with("List"))
                return "array";
            return "string";
        }

        mcp::json build_json_schema_property(const std::string& json_type, const std::string& description) {
            mcp::json prop;
            if (json_type.starts_with("array:")) {
                prop["type"] = "array";
                std::string item_type = json_type.substr(6);
                prop["items"] = mcp::json{{"type", item_type}};
            } else if (json_type == "array") {
                prop["type"] = "array";
            } else {
                prop["type"] = json_type;
            }
            if (!description.empty()) {
                prop["description"] = description;
            }
            return prop;
        }

        mcp::json python_value_to_json(nb::handle obj) {
            if (obj.is_none()) {
                return nullptr;
            }
            if (nb::isinstance<nb::bool_>(obj)) {
                return nb::cast<bool>(obj);
            }
            if (nb::isinstance<nb::int_>(obj)) {
                return nb::cast<int64_t>(obj);
            }
            if (nb::isinstance<nb::float_>(obj)) {
                return nb::cast<double>(obj);
            }
            if (nb::isinstance<nb::str>(obj)) {
                return nb::cast<std::string>(obj);
            }
            if (nb::isinstance<nb::list>(obj)) {
                mcp::json arr = mcp::json::array();
                nb::list lst = nb::cast<nb::list>(obj);
                for (size_t i = 0; i < lst.size(); ++i) {
                    arr.push_back(python_value_to_json(lst[i]));
                }
                return arr;
            }
            if (nb::isinstance<nb::dict>(obj)) {
                mcp::json dict = mcp::json::object();
                nb::dict d = nb::cast<nb::dict>(obj);
                nb::object items = d.attr("items")();
                for (auto item : items) {
                    nb::tuple tup = nb::cast<nb::tuple>(item);
                    std::string key = nb::cast<std::string>(tup[0]);
                    dict[key] = python_value_to_json(tup[1]);
                }
                return dict;
            }
            return nb::cast<std::string>(nb::str(obj));
        }

        nb::object json_to_python(const mcp::json& j) {
            if (j.is_null()) {
                return nb::none();
            }
            if (j.is_boolean()) {
                return nb::cast(j.get<bool>());
            }
            if (j.is_number_integer()) {
                return nb::cast(j.get<int64_t>());
            }
            if (j.is_number_float()) {
                return nb::cast(j.get<double>());
            }
            if (j.is_string()) {
                return nb::cast(j.get<std::string>());
            }
            if (j.is_array()) {
                nb::list lst;
                for (const auto& el : j) {
                    lst.append(json_to_python(el));
                }
                return lst;
            }
            if (j.is_object()) {
                nb::dict d;
                for (auto& [k, v] : j.items()) {
                    d[k.c_str()] = json_to_python(v);
                }
                return d;
            }
            return nb::none();
        }

        struct PyToolInfo {
            std::string name;
            std::string description;
            nb::object callback;
            mcp::json properties;
            std::vector<std::string> required;
        };

        std::vector<std::string> registered_tools_;
        std::mutex tools_mutex_;

        mcp::McpTool create_tool_from_info(const PyToolInfo& info) {
            return mcp::McpTool{
                .name = info.name,
                .description = info.description,
                .input_schema = {
                    .type = "object",
                    .properties = info.properties,
                    .required = info.required}};
        }

        mcp::ToolRegistry::ToolHandler create_python_handler(nb::object callback) {
            return [callback](const mcp::json& args) -> mcp::json {
                nb::gil_scoped_acquire gil;
                try {
                    nb::dict py_args = nb::cast<nb::dict>(json_to_python(args));
                    nb::object result = callback(**py_args);
                    return python_value_to_json(result);
                } catch (const std::exception& e) {
                    return mcp::json{{"error", std::string("Python tool error: ") + e.what()}};
                }
            };
        }

        PyToolInfo extract_tool_info(nb::callable fn, const std::string& name, const std::string& description) {
            PyToolInfo info;
            info.callback = fn;

            nb::object inspect = nb::module_::import_("inspect");
            nb::object sig = inspect.attr("signature")(fn);
            nb::object params = sig.attr("parameters");

            std::string tool_name = name;
            if (tool_name.empty()) {
                tool_name = nb::cast<std::string>(fn.attr("__name__"));
            }
            info.name = "plugin." + tool_name;

            std::string tool_desc = description;
            if (tool_desc.empty()) {
                nb::object doc = fn.attr("__doc__");
                if (!doc.is_none()) {
                    tool_desc = nb::cast<std::string>(doc);
                }
            }
            info.description = tool_desc;

            mcp::json properties = mcp::json::object();
            std::vector<std::string> required;

            nb::object items = params.attr("items")();
            for (auto item : items) {
                nb::tuple tup = nb::cast<nb::tuple>(item);
                std::string param_name = nb::cast<std::string>(tup[0]);
                nb::object param = tup[1];

                nb::object annotation = param.attr("annotation");
                nb::object default_val = param.attr("default");
                nb::object empty = inspect.attr("Parameter").attr("empty");

                std::string json_type = "string";
                if (!annotation.is(empty)) {
                    std::string type_name = nb::cast<std::string>(nb::str(annotation));
                    if (type_name.find("int") != std::string::npos && type_name.find("list") == std::string::npos) {
                        json_type = "integer";
                    } else if (type_name.find("float") != std::string::npos) {
                        json_type = "number";
                    } else if (type_name.find("bool") != std::string::npos) {
                        json_type = "boolean";
                    } else if (type_name.find("str") != std::string::npos) {
                        json_type = "string";
                    } else if (type_name.find("list") != std::string::npos || type_name.find("List") != std::string::npos) {
                        json_type = "array";
                    }
                }

                properties[param_name] = build_json_schema_property(json_type, "");

                bool has_default = !default_val.is(empty);
                if (!has_default) {
                    required.push_back(param_name);
                }
            }

            info.properties = properties;
            info.required = required;

            return info;
        }

        void register_python_tool(nb::callable fn, const std::string& name, const std::string& description) {
            PyToolInfo info = extract_tool_info(fn, name, description);
            mcp::McpTool tool = create_tool_from_info(info);
            auto handler = create_python_handler(info.callback);

            mcp::ToolRegistry::instance().register_tool(std::move(tool), std::move(handler));

            std::lock_guard lock(tools_mutex_);
            registered_tools_.push_back(info.name);

            LOG_INFO("Registered Python MCP tool: {}", info.name);
        }

        void unregister_python_tool(const std::string& name) {
            std::string full_name = name;
            if (!name.starts_with("plugin.")) {
                full_name = "plugin." + name;
            }
            mcp::ToolRegistry::instance().unregister_tool(full_name);

            std::lock_guard lock(tools_mutex_);
            registered_tools_.erase(
                std::remove(registered_tools_.begin(), registered_tools_.end(), full_name),
                registered_tools_.end());

            LOG_INFO("Unregistered Python MCP tool: {}", full_name);
        }

        std::vector<std::string> list_python_tools() {
            std::lock_guard lock(tools_mutex_);
            return registered_tools_;
        }

        std::vector<std::string> list_all_tools() {
            auto tools = mcp::ToolRegistry::instance().list_tools();
            std::vector<std::string> names;
            names.reserve(tools.size());
            for (const auto& tool : tools) {
                names.push_back(tool.name);
            }
            return names;
        }

        nb::list describe_all_tools() {
            nb::list result;
            for (const auto& tool : mcp::ToolRegistry::instance().list_tools()) {
                result.append(json_to_python(mcp::tool_to_json(tool)));
            }
            return result;
        }

        std::vector<std::string> list_all_resources() {
            auto resources = mcp::ResourceRegistry::instance().list_resources();
            std::vector<std::string> uris;
            uris.reserve(resources.size());
            for (const auto& resource : resources) {
                uris.push_back(resource.uri);
            }
            return uris;
        }

        nb::list read_resource_contents(const std::string& uri) {
            auto contents = mcp::ResourceRegistry::instance().read_resource(uri);
            if (!contents) {
                throw nb::value_error(contents.error().c_str());
            }

            nb::list result;
            for (const auto& content : *contents) {
                nb::dict item;
                item["uri"] = content.uri;
                if (content.mime_type) {
                    item["mime_type"] = *content.mime_type;
                }

                if (std::holds_alternative<std::string>(content.content)) {
                    const auto& string_content = std::get<std::string>(content.content);
                    const bool is_blob =
                        content.mime_type && content.mime_type->starts_with("image/");
                    if (is_blob) {
                        item["blob"] = string_content;
                    } else {
                        item["text"] = string_content;
                    }
                } else {
                    item["blob"] = core::base64_encode(std::get<std::vector<uint8_t>>(content.content));
                }

                result.append(std::move(item));
            }

            return result;
        }

    } // namespace

    void register_mcp(nb::module_& m) {
        auto mcp_module = m.def_submodule("mcp", "MCP (Model Context Protocol) tool registration");

        mcp_module.def(
            "register_tool",
            [](nb::callable fn, const std::string& name, const std::string& description) {
                register_python_tool(fn, name, description);
            },
            nb::arg("fn"), nb::arg("name") = "", nb::arg("description") = "",
            "Register a Python function as an MCP tool");

        mcp_module.def(
            "unregister_tool",
            [](const std::string& name) { unregister_python_tool(name); },
            nb::arg("name"),
            "Unregister an MCP tool");

        mcp_module.def(
            "list_tools", []() { return list_all_tools(); },
            "List all registered shared capabilities/tools");

        mcp_module.def(
            "list_python_tools", []() { return list_python_tools(); },
            "List Python-provided MCP tools registered through this module");

        mcp_module.def(
            "describe_tools", []() { return describe_all_tools(); },
            "Describe all registered shared capabilities/tools");

        mcp_module.def(
            "list_resources", []() { return list_all_resources(); },
            "List all registered MCP resources");

        mcp_module.def(
            "read_resource", [](const std::string& uri) { return read_resource_contents(uri); },
            nb::arg("uri"),
            "Read one registered MCP resource");

        mcp_module.def(
            "call_tool",
            [](const std::string& name, nb::handle args) {
                mcp::json json_args = mcp::json::object();
                if (!args.is_none()) {
                    json_args = python_value_to_json(args);
                    if (!json_args.is_object()) {
                        throw nb::value_error("call_tool args must be a dict/object");
                    }
                }
                return json_to_python(mcp::ToolRegistry::instance().call_tool(name, json_args));
            },
            nb::arg("name"), nb::arg("args") = nb::none(),
            "Invoke a registered shared capability/tool");

        mcp_module.def(
            "tool",
            [](const std::string& name, const std::string& description) {
                return nb::cpp_function([name, description](nb::callable fn) -> nb::callable {
                    register_python_tool(fn, name, description);
                    return fn;
                });
            },
            nb::arg("name") = "", nb::arg("description") = "",
            "Decorator to register a function as an MCP tool");
    }

} // namespace lfs::python
