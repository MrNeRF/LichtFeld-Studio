/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "py_shaders.hpp"
#include "py_error.hpp"
#include "python/python_runtime.hpp"
#include "shaders/shader_api.hpp"
#include "visualizer/post_work_utils.hpp"
#include "visualizer/visualizer_impl.hpp"
#include <chrono>
#include <nanobind/stl/string.h>
#include <thread>

namespace lfs::python {
    namespace {
        namespace nb = nanobind;
        using json = nlohmann::json;
        Error unavailable() {
            return make_error(ErrorInit{.code = ErrorCode::Unavailable,
                                        .domain = ErrorDomain::Python,
                                        .user_message = "Shaders require a running GUI viewer.",
                                        .detection = LFS_SOURCE_SITE_CURRENT()});
        }

        json observe(const std::string& operation, const json& args) {
            auto* viewer = get_visualizer();
            if (!viewer || !viewer->acceptsPostedWork())
                throw Exception(unavailable());
            const auto invoke = [operation, args]() -> Result<json> {
                auto* impl = dynamic_cast<vis::VisualizerImpl*>(get_visualizer());
                auto* service = impl ? impl->getShaderService() : nullptr;
                if (!service)
                    return unavailable();
                return vis::shaders::shaderOperation(*service, operation, args);
            };
            nb::gil_scoped_release release;
            return unwrap(viewer->isOnViewerThread() ? invoke()
                                                     : vis::post_work_and_wait(
                                                           [viewer](vis::Visualizer::WorkItem work) {
                                                               return viewer->postWork(std::move(work));
                                                           },
                                                           invoke, []() -> Result<json> { return unavailable(); }));
        }

        json fromPython(nb::handle value) {
            return json::parse(nb::cast<std::string>(nb::module_::import_("json").attr("dumps")(value)));
        }

        nb::object toPython(const json& value) { return nb::module_::import_("json").attr("loads")(value.dump()); }
        void optional(json& args, const char* key, nb::handle value) {
            if (!value.is_none())
                args[key] = fromPython(value);
        }
    } // namespace
    void register_shaders(nb::module_& module) {
        auto m = module.def_submodule("shaders", "Draw-driven shader assets and bindings");
        m.def(
            "list", [] { return toPython(observe("list", json::object())["assets"]); },
            nb::sig("def list() -> list[dict]"));
        m.def(
            "get",
            [](const std::string& shader_id, bool include_source) {
                return toPython(observe("get", {{"shader_id", shader_id}, {"include_source", include_source}}));
            },
            nb::arg("shader_id"), nb::arg("include_source") = true,
            nb::sig("def get(shader_id: str, include_source: bool = True) -> dict"));
        m.def(
            "set_source",
            [](const std::string& source, nb::object shader_id, nb::object name, nb::object expected_revision) {
                json args{{"source", source}};
                optional(args, "shader_id", shader_id);
                optional(args, "name", name);
                optional(args, "expected_revision", expected_revision);
                return toPython(observe("set_source", args));
            },
            nb::arg("source"), nb::arg("shader_id") = nb::none(), nb::arg("name") = nb::none(),
            nb::arg("expected_revision") = nb::none(),
            nb::sig("def set_source(source: str, shader_id: str | None = None, name: str | None = None, "
                    "expected_revision: int | None = None) -> dict"));
        m.def(
            "bind",
            [](const std::string& shader_id, nb::dict target) {
                return toPython(observe("bind", {{"shader_id", shader_id}, {"target", fromPython(target)}}));
            },
            nb::arg("shader_id"), nb::arg("target"), nb::sig("def bind(shader_id: str, target: dict) -> dict"));
        m.def(
            "unbind",
            [](nb::dict target, nb::object shader_id) {
                json args{{"target", fromPython(target)}};
                optional(args, "shader_id", shader_id);
                return toPython(observe("unbind", args));
            },
            nb::arg("target"), nb::arg("shader_id") = nb::none(),
            nb::sig("def unbind(target: dict, shader_id: str | None = None) -> dict"));
        m.def(
            "set_param",
            [](const std::string& shader_id, const std::string& name, nb::object value, nb::object expected_revision) {
                json args{{"shader_id", shader_id}, {"name", name}, {"value", fromPython(value)}};
                optional(args, "expected_revision", expected_revision);
                return toPython(observe("set_param", args));
            },
            nb::arg("shader_id"), nb::arg("name"), nb::arg("value"), nb::arg("expected_revision") = nb::none(),
            nb::sig("def set_param(shader_id: str, name: str, value: float | list[float], expected_revision: int | "
                    "None = None) -> dict"));
        m.def(
            "remove",
            [](const std::string& shader_id) { return toPython(observe("remove", {{"shader_id", shader_id}})); },
            nb::arg("shader_id"), nb::sig("def remove(shader_id: str) -> dict"));
        m.def(
            "status",
            [](const std::string& shader_id) {
                return toPython(observe("get", {{"shader_id", shader_id}, {"include_source", false}})["status"]);
            },
            nb::arg("shader_id"), nb::sig("def status(shader_id: str) -> dict"));
        m.def(
            "wait",
            [](const std::string& shader_id, nb::object source_revision, int timeout_ms) {
                if (timeout_ms < 0 || timeout_ms > 30000)
                    throw nb::value_error("timeout_ms must be between 0 and 30000");
                auto* viewer = get_visualizer();
                if (!viewer || !viewer->acceptsPostedWork())
                    throw Exception(unavailable());
                if (timeout_ms && viewer->isOnViewerThread())
                    throw nb::value_error("Shader waits cannot block the viewer thread");
                std::optional<std::uint64_t> revision;
                if (!source_revision.is_none()) {
                    const auto value = fromPython(source_revision);
                    if (!value.is_number_integer() || (!value.is_number_unsigned() && value.get<std::int64_t>() < 0))
                        throw nb::value_error("source_revision must be a nonnegative integer");
                    revision = value.get<std::uint64_t>();
                }
                const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
                auto* impl = dynamic_cast<vis::VisualizerImpl*>(viewer);
                if (!impl || !impl->getShaderService())
                    throw Exception(unavailable());
                auto changes = impl->getShaderService()->changes();
                const auto origin_generation = changes->generation.load(std::memory_order_acquire);
                for (;;) {
                    const auto generation = changes->generation.load(std::memory_order_acquire);
                    auto status = observe("get", {{"shader_id", shader_id}, {"include_source", false}})["status"];
                    if (!revision)
                        revision = status["source_revision"].get<std::uint64_t>();
                    vis::shaders::ShaderStatus observation;
                    observation.revision = status["source_revision"].get<std::uint64_t>();
                    const std::string state = status["state"];
                    observation.state = state == "failed" ? vis::shaders::ShaderState::Failed : state == "blocked" ? vis::shaders::ShaderState::Blocked
                                                                                            : state == "unbound"   ? vis::shaders::ShaderState::Unbound
                                                                                                                   : vis::shaders::ShaderState::Pending;
                    std::string outcome(vis::shaders::shaderOutcome(observation, *revision,
                                                                    state == "applied" && status["pending"].empty() && !status["compiling"].get<bool>()));
                    if (changes->cancelled(origin_generation))
                        outcome = "cancelled";
                    if (outcome.empty() && (viewer->isOnViewerThread() || std::chrono::steady_clock::now() >= deadline))
                        outcome = "timeout";
                    if (!outcome.empty())
                        return toPython(
                            {{"outcome", outcome}, {"status", status}, {"completed", outcome != "timeout"}});
                    nb::gil_scoped_release release;
                    changes->wait(generation, deadline);
                }
            },
            nb::arg("shader_id"), nb::arg("source_revision") = nb::none(), nb::arg("timeout_ms") = 2000,
            nb::sig("def wait(shader_id: str, source_revision: int | None = None, timeout_ms: int = 2000) -> dict"));
    }
} // namespace lfs::python
