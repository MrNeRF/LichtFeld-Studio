/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include "notification_bridge.hpp"
#include "py_animation.hpp"
#include "py_cameras.hpp"
#include "py_command.hpp"
#include "py_gizmo.hpp"
#include "py_io.hpp"
#include "py_mcp.hpp"
#include "py_operator.hpp"
#include "py_packages.hpp"
#include "py_params.hpp"
#include "py_pipeline.hpp"
#include "py_plugins.hpp"
#include "py_rendering.hpp"
#include "py_scene.hpp"
#include "py_scripts.hpp"
#include "py_selection.hpp"
#include "py_signals.hpp"
#include "py_splat_data.hpp"
#include "py_tensor.hpp"
#include "py_ui.hpp"
#include "py_uilist.hpp"
#include "py_viewport.hpp"

#include "control/command_api.hpp"
#include "control/control_boundary.hpp"
#include "core/event_bridge/command_center_bridge.hpp"
#include "core/event_bridge/scoped_handler.hpp"
#include "core/events.hpp"
#include "core/image_io.hpp"
#include "core/logger.hpp"
#include "core/parameters.hpp"
#include "internal/resource_paths.hpp"
#include "io/filesystem_utils.hpp"
#include "python/python_runtime.hpp"

#include "config.h"
#include "core/checkpoint_format.hpp"
#include "python/runner.hpp"
#include "rendering/rendering_manager.hpp"
#include "training/strategies/istrategy.hpp"
#include "training/trainer.hpp"
#include "visualizer/core/editor_context.hpp"
#include "visualizer/core/parameter_manager.hpp"
#include "visualizer/core/services.hpp"
#include "visualizer/gui/panel_registry.hpp"
#include "visualizer/operator/operator_registry.hpp"
#include "visualizer/scene/scene_manager.hpp"
#include "visualizer/training/training_manager.hpp"
#include "visualizer/window/window_manager.hpp"

#include <cstring>
#include <filesystem>
#include <format>
#include <fstream>
#include <functional>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#define GLM_ENABLE_EXPERIMENTAL
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtx/euler_angles.hpp>
#include <glm/gtx/matrix_decompose.hpp>

namespace nb = nanobind;

enum class RenderMode { Splats,
                        Points,
                        Rings,
                        Centers };
enum class OperatorResult { Finished,
                            Cancelled,
                            Running };

namespace {

    using lfs::training::Command;
    using lfs::training::CommandCenter;
    using lfs::training::CommandTarget;
    using lfs::training::ControlBoundary;
    using lfs::training::ControlHook;
    using lfs::training::HookContext;
    using lfs::training::SelectionKind;
    using lfs::training::TrainingPhase;
    using lfs::training::TrainingSnapshot;

    CommandCenter* get_command_center_opt() {
        return lfs::event::command_center();
    }

    CommandCenter& get_command_center() {
        auto* cc = get_command_center_opt();
        if (!cc) {
            throw std::runtime_error("Training system not initialized");
        }
        return *cc;
    }

    class PyControlSession {
    public:
        PyControlSession() = default;

        ~PyControlSession() { clear(); }

        void clear() {
            for (const auto& reg : registrations_) {
                ControlBoundary::instance().unregister_callback(reg.hook, reg.id);
            }
            registrations_.clear();
        }

        void on_training_start(nb::callable fn) { add(ControlHook::TrainingStart, std::move(fn)); }
        void on_iteration_start(nb::callable fn) { add(ControlHook::IterationStart, std::move(fn)); }
        void on_pre_optimizer_step(nb::callable fn) { add(ControlHook::PreOptimizerStep, std::move(fn)); }
        void on_post_step(nb::callable fn) { add(ControlHook::PostStep, std::move(fn)); }
        void on_training_end(nb::callable fn) { add(ControlHook::TrainingEnd, std::move(fn)); }

    private:
        struct RegistrationHandle {
            ControlHook hook;
            std::size_t id;
        };

        void add(ControlHook hook, nb::callable fn) {
            nb::object fn_obj = std::move(fn);
            auto cb = [fn_obj](const HookContext& ctx) {
                nb::gil_scoped_acquire gil;
                fn_obj(ctx.iteration, ctx.loss, ctx.num_gaussians, ctx.is_refining);
            };

            const auto id = ControlBoundary::instance().register_callback(hook, std::move(cb));
            registrations_.push_back({hook, id});
            owned_callbacks_.push_back(std::move(fn_obj));
        }

        std::vector<RegistrationHandle> registrations_;
        std::vector<nb::object> owned_callbacks_;
    };

    class PyScopedHandler {
    public:
        PyScopedHandler() = default;
        ~PyScopedHandler() = default;

        PyScopedHandler(const PyScopedHandler&) = delete;
        PyScopedHandler& operator=(const PyScopedHandler&) = delete;
        PyScopedHandler(PyScopedHandler&&) = default;
        PyScopedHandler& operator=(PyScopedHandler&&) = default;

        void on_training_start(nb::callable cb) { add_hook(ControlHook::TrainingStart, cb); }
        void on_iteration_start(nb::callable cb) { add_hook(ControlHook::IterationStart, cb); }
        void on_pre_optimizer_step(nb::callable cb) { add_hook(ControlHook::PreOptimizerStep, cb); }
        void on_post_step(nb::callable cb) { add_hook(ControlHook::PostStep, cb); }
        void on_training_end(nb::callable cb) { add_hook(ControlHook::TrainingEnd, cb); }

        void clear() {
            handler_ = lfs::event::ScopedHandler();
            owned_callbacks_.clear();
        }

    private:
        void add_hook(ControlHook hook, nb::callable cb) {
            nb::object fn = nb::cast<nb::object>(cb);
            owned_callbacks_.push_back(fn);

            handler_.subscribe_hook(hook, [fn](const HookContext& ctx) {
                nb::gil_scoped_acquire gil;
                try {
                    nb::dict d;
                    d["iter"] = ctx.iteration;
                    d["loss"] = ctx.loss;
                    d["num_splats"] = ctx.num_gaussians;
                    d["is_refining"] = ctx.is_refining;
                    fn(d);
                } catch (const std::exception& e) {
                    LOG_ERROR("Python hook error: {}", e.what());
                }
            });
        }

        lfs::event::ScopedHandler handler_;
        std::vector<nb::object> owned_callbacks_;
    };

    class PyContextView {
    public:
        PyContextView() {
            auto* cc = get_command_center_opt();
            if (cc) {
                snapshot_ = cc->snapshot();
                if (snapshot_.trainer) {
                    strategy_ = snapshot_.trainer->getParams().optimization.strategy;
                }
            }
        }

        int iteration() const { return snapshot_.iteration; }
        int max_iterations() const { return snapshot_.max_iterations; }
        float loss() const { return snapshot_.loss; }
        std::size_t num_gaussians() const { return snapshot_.num_gaussians; }
        bool is_refining() const { return snapshot_.is_refining; }
        bool is_training() const { return snapshot_.is_running; }
        bool is_paused() const { return snapshot_.is_paused; }

        std::string phase() const {
            switch (snapshot_.phase) {
            case TrainingPhase::Idle: return "idle";
            case TrainingPhase::IterationStart: return "iteration_start";
            case TrainingPhase::Forward: return "forward";
            case TrainingPhase::Backward: return "backward";
            case TrainingPhase::OptimizerStep: return "optimizer_step";
            case TrainingPhase::SafeControl: return "safe_control";
            default: return "unknown";
            }
        }

        std::string strategy() const { return strategy_; }

        void refresh() {
            auto* cc = get_command_center_opt();
            if (cc) {
                snapshot_ = cc->snapshot();
                if (snapshot_.trainer) {
                    strategy_ = snapshot_.trainer->getParams().optimization.strategy;
                } else {
                    strategy_ = "none";
                }
            }
        }

    private:
        TrainingSnapshot snapshot_{};
        std::string strategy_ = "none";
    };

    struct PyGaussiansView {
        std::size_t count() const {
            auto* cc = get_command_center_opt();
            if (!cc)
                return 0;
            const auto snap = cc->snapshot();
            if (!snap.trainer)
                return 0;
            return snap.trainer->get_strategy_mutable().get_model().size();
        }

        int sh_degree() const {
            auto* cc = get_command_center_opt();
            if (!cc)
                return 0;
            const auto snap = cc->snapshot();
            if (!snap.trainer)
                return 0;
            return snap.trainer->get_strategy_mutable().get_model().get_active_sh_degree();
        }

        int max_sh_degree() const {
            auto* cc = get_command_center_opt();
            if (!cc)
                return 0;
            const auto snap = cc->snapshot();
            if (!snap.trainer)
                return 0;
            return snap.trainer->get_strategy_mutable().get_model().get_max_sh_degree();
        }
    };

    // Optimizer view for Python - wraps CommandCenter operations
    struct PyOptimizerView {
        void scale_lr(float factor) {
            Command cmd;
            cmd.op = "scale_lr";
            cmd.target = CommandTarget::Optimizer;
            cmd.selection = {SelectionKind::All};
            cmd.args["factor"] = static_cast<double>(factor);
            const auto result = get_command_center().execute(cmd);
            if (!result) {
                throw std::runtime_error(std::format("scale_lr failed: {}", result.error()));
            }
        }

        void set_lr(float value) {
            Command cmd;
            cmd.op = "set_lr";
            cmd.target = CommandTarget::Optimizer;
            cmd.selection = {SelectionKind::All};
            cmd.args["value"] = static_cast<double>(value);
            const auto result = get_command_center().execute(cmd);
            if (!result) {
                throw std::runtime_error(std::format("set_lr failed: {}", result.error()));
            }
        }

        float get_lr() const {
            const auto snap = get_command_center().snapshot();
            if (!snap.trainer)
                return 0.0f;
            return snap.trainer->get_strategy_mutable().get_optimizer().get_lr();
        }
    };

    // Model view for Python - wraps CommandCenter operations
    struct PyModelView {
        void clamp(const std::string& attr, std::optional<float> min_val, std::optional<float> max_val) {
            Command cmd;
            cmd.op = "clamp_attribute";
            cmd.target = CommandTarget::Model;
            cmd.selection = {SelectionKind::All};
            cmd.args["attribute"] = attr;
            if (min_val)
                cmd.args["min"] = static_cast<double>(*min_val);
            if (max_val)
                cmd.args["max"] = static_cast<double>(*max_val);
            const auto result = get_command_center().execute(cmd);
            if (!result) {
                throw std::runtime_error(std::format("clamp failed: {}", result.error()));
            }
        }

        void scale(const std::string& attr, float factor) {
            Command cmd;
            cmd.op = "scale_attribute";
            cmd.target = CommandTarget::Model;
            cmd.selection = {SelectionKind::All};
            cmd.args["attribute"] = attr;
            cmd.args["factor"] = static_cast<double>(factor);
            const auto result = get_command_center().execute(cmd);
            if (!result) {
                throw std::runtime_error(std::format("scale failed: {}", result.error()));
            }
        }

        void set(const std::string& attr, float value) {
            Command cmd;
            cmd.op = "set_attribute";
            cmd.target = CommandTarget::Model;
            cmd.selection = {SelectionKind::All};
            cmd.args["attribute"] = attr;
            cmd.args["value"] = static_cast<double>(value);
            const auto result = get_command_center().execute(cmd);
            if (!result) {
                throw std::runtime_error(std::format("set failed: {}", result.error()));
            }
        }
    };

    // Session view for Python - provides access to optimizer and model operations
    struct PySession {
        PyOptimizerView optimizer() { return {}; }
        PyModelView model() { return {}; }

        void pause() {
            Command cmd;
            cmd.op = "pause";
            cmd.target = CommandTarget::Session;
            cmd.selection = {SelectionKind::All};
            const auto result = get_command_center().execute(cmd);
            if (!result) {
                throw std::runtime_error(std::format("pause failed: {}", result.error()));
            }
        }

        void resume() {
            Command cmd;
            cmd.op = "resume";
            cmd.target = CommandTarget::Session;
            cmd.selection = {SelectionKind::All};
            const auto result = get_command_center().execute(cmd);
            if (!result) {
                throw std::runtime_error(std::format("resume failed: {}", result.error()));
            }
        }

        void request_stop() {
            Command cmd;
            cmd.op = "request_stop";
            cmd.target = CommandTarget::Session;
            cmd.selection = {SelectionKind::All};
            const auto result = get_command_center().execute(cmd);
            if (!result) {
                throw std::runtime_error(std::format("request_stop failed: {}", result.error()));
            }
        }
    };

    // Thread-local Trainer pointer for get_scene() to access Scene during hooks
    thread_local lfs::training::Trainer* g_current_trainer = nullptr;

    // RAII guard to set/clear current trainer
    struct TrainerGuard {
        TrainerGuard(lfs::training::Trainer* t) { g_current_trainer = t; }
        ~TrainerGuard() { g_current_trainer = nullptr; }
    };

    // Hook registration helper
    std::size_t register_hook(ControlHook hook, nb::callable cb) {
        if (!cb)
            return 0;
        const nb::object ocb = nb::cast<nb::object>(cb);
        LOG_INFO("Python hook registered for hook {}", static_cast<int>(hook));
        return ControlBoundary::instance().register_callback(hook, [ocb, hook](const HookContext& ctx) {
            nb::gil_scoped_acquire guard;
            TrainerGuard trainer_guard(ctx.trainer);
            LOG_DEBUG("Python hook invoke hook={} iter={}", static_cast<int>(hook), ctx.iteration);
            try {
                nb::dict d;
                d["iter"] = ctx.iteration;
                d["loss"] = ctx.loss;
                d["num_splats"] = ctx.num_gaussians;
                d["is_refining"] = ctx.is_refining;
                ocb(d);
            } catch (const std::exception& e) {
                LOG_ERROR("Python hook threw: {}", e.what());
            }
        });
    }

    // Get Scene from application context, trainer, or operation context
    lfs::vis::Scene* get_scene_internal() {
        // Priority 1: Application scene (persistent, works from background threads)
        if (auto* app_scene = lfs::python::get_application_scene()) {
            return app_scene;
        }
        // Priority 2: Current trainer (headless mode during hooks)
        if (g_current_trainer) {
            return g_current_trainer->getScene();
        }
        // Priority 3: Operation context (short-lived, for capability invocations)
        return lfs::python::get_scene_for_python();
    }

} // namespace

NB_MODULE(lichtfeld, m) {
    m.doc() = "LichtFeld Python control module for Gaussian splatting";

    // Enums (Phase 4: typed APIs instead of magic strings)
    nb::enum_<RenderMode>(m, "RenderMode")
        .value("SPLATS", RenderMode::Splats)
        .value("POINTS", RenderMode::Points)
        .value("RINGS", RenderMode::Rings)
        .value("CENTERS", RenderMode::Centers);

    nb::enum_<OperatorResult>(m, "OperatorResult")
        .value("FINISHED", OperatorResult::Finished)
        .value("CANCELLED", OperatorResult::Cancelled)
        .value("RUNNING", OperatorResult::Running);

    // Add user site-packages to sys.path on module import
    {
        auto user_packages = lfs::python::get_user_packages_dir();
        nb::module_ sys = nb::module_::import_("sys");
        nb::list path = nb::cast<nb::list>(sys.attr("path"));
        std::string pkg_path = user_packages.string();
        bool found = false;
        for (size_t i = 0; i < path.size(); ++i) {
            if (nb::cast<std::string>(path[i]) == pkg_path) {
                found = true;
                break;
            }
        }
        if (!found) {
            path.insert(0, pkg_path.c_str());
        }
    }

    // Enum for hook types
    nb::enum_<ControlHook>(m, "Hook")
        .value("training_start", ControlHook::TrainingStart)
        .value("iteration_start", ControlHook::IterationStart)
        .value("pre_optimizer_step", ControlHook::PreOptimizerStep)
        .value("post_step", ControlHook::PostStep)
        .value("training_end", ControlHook::TrainingEnd);

    // Session class for managing callbacks
    nb::class_<PyControlSession>(m, "Session")
        .def(nb::init<>())
        .def("on_training_start", &PyControlSession::on_training_start, "Register training start callback")
        .def("on_iteration_start", &PyControlSession::on_iteration_start, "Register iteration start callback")
        .def("on_pre_optimizer_step", &PyControlSession::on_pre_optimizer_step, "Register pre-optimizer callback")
        .def("on_post_step", &PyControlSession::on_post_step, "Register post-step callback")
        .def("on_training_end", &PyControlSession::on_training_end, "Register training end callback")
        .def("clear", &PyControlSession::clear, "Unregister all callbacks");

    nb::class_<PyScopedHandler>(m, "ScopedHandler")
        .def(nb::init<>())
        .def("on_training_start", &PyScopedHandler::on_training_start, nb::arg("callback"),
             "Register training start callback")
        .def("on_iteration_start", &PyScopedHandler::on_iteration_start, nb::arg("callback"),
             "Register iteration start callback")
        .def("on_pre_optimizer_step", &PyScopedHandler::on_pre_optimizer_step, nb::arg("callback"),
             "Register pre-optimizer callback")
        .def("on_post_step", &PyScopedHandler::on_post_step, nb::arg("callback"),
             "Register post-step callback")
        .def("on_training_end", &PyScopedHandler::on_training_end, nb::arg("callback"),
             "Register training end callback")
        .def("clear", &PyScopedHandler::clear, "Unregister all callbacks");

    // Context view class (caches snapshot on creation, call refresh() to update)
    nb::class_<PyContextView>(m, "Context")
        .def(nb::init<>())
        .def_prop_ro("iteration", &PyContextView::iteration, "Current iteration count")
        .def_prop_ro("max_iterations", &PyContextView::max_iterations, "Maximum iteration count")
        .def_prop_ro("loss", &PyContextView::loss, "Current training loss value")
        .def_prop_ro("num_gaussians", &PyContextView::num_gaussians, "Number of Gaussians in the model")
        .def_prop_ro("is_refining", &PyContextView::is_refining, "Whether training is in refinement phase")
        .def_prop_ro("is_training", &PyContextView::is_training, "Whether training is currently running")
        .def_prop_ro("is_paused", &PyContextView::is_paused, "Whether training is paused")
        .def_prop_ro("phase", &PyContextView::phase, "Current training phase name")
        .def_prop_ro("strategy", &PyContextView::strategy, "Active training strategy name")
        .def("refresh", &PyContextView::refresh, "Update cached snapshot from current state");

    // Gaussians view class
    nb::class_<PyGaussiansView>(m, "Gaussians")
        .def(nb::init<>())
        .def_prop_ro("count", &PyGaussiansView::count, "Total number of Gaussians")
        .def_prop_ro("sh_degree", &PyGaussiansView::sh_degree, "Current active SH degree")
        .def_prop_ro("max_sh_degree", &PyGaussiansView::max_sh_degree, "Maximum SH degree");

    // Optimizer view class
    nb::class_<PyOptimizerView>(m, "Optimizer")
        .def(nb::init<>())
        .def("scale_lr", &PyOptimizerView::scale_lr, nb::arg("factor"), "Scale learning rate by factor")
        .def("set_lr", &PyOptimizerView::set_lr, nb::arg("value"), "Set learning rate")
        .def("get_lr", &PyOptimizerView::get_lr, "Get current learning rate");

    // Model view class
    nb::class_<PyModelView>(m, "Model")
        .def(nb::init<>())
        .def("clamp", &PyModelView::clamp, nb::arg("attr"), nb::arg("min") = nb::none(), nb::arg("max") = nb::none(),
             "Clamp attribute values")
        .def("scale", &PyModelView::scale, nb::arg("attr"), nb::arg("factor"), "Scale attribute by factor")
        .def("set", &PyModelView::set, nb::arg("attr"), nb::arg("value"), "Set attribute value");

    // Session class
    nb::class_<PySession>(m, "Session")
        .def(nb::init<>())
        .def("optimizer", &PySession::optimizer, "Get optimizer view")
        .def("model", &PySession::model, "Get model view")
        .def("pause", &PySession::pause, "Pause training")
        .def("resume", &PySession::resume, "Resume training")
        .def("request_stop", &PySession::request_stop, "Request training stop");

    // Convenience functions
    m.def(
        "context", []() { return PyContextView{}; }, "Get current training context");
    m.def(
        "gaussians", []() { return PyGaussiansView{}; }, "Get Gaussians info");
    m.def(
        "session", []() { return PySession{}; }, "Get training session");

    m.def(
        "trainer_state",
        []() -> const char* {
            const auto* const tm = lfs::python::get_trainer_manager();
            if (!tm)
                return "idle";
            switch (tm->getState()) {
            case lfs::vis::TrainingState::Idle: return "idle";
            case lfs::vis::TrainingState::Ready: return "ready";
            case lfs::vis::TrainingState::Running: return "running";
            case lfs::vis::TrainingState::Paused: return "paused";
            case lfs::vis::TrainingState::Stopping: return "stopping";
            case lfs::vis::TrainingState::Finished: return "finished";
            }
            return "idle";
        },
        "Get trainer state");

    m.def(
        "finish_reason",
        []() -> std::optional<std::string> {
            const auto* const tm = lfs::python::get_trainer_manager();
            if (!tm || !tm->isFinished())
                return std::nullopt;
            switch (tm->getStateMachine().getFinishReason()) {
            case lfs::vis::FinishReason::Completed: return "completed";
            case lfs::vis::FinishReason::UserStopped: return "stopped";
            case lfs::vis::FinishReason::Error: return "error";
            default: return std::nullopt;
            }
        },
        "Get finish reason if training finished");

    m.def(
        "trainer_error",
        []() -> std::optional<std::string> {
            const auto* const tm = lfs::python::get_trainer_manager();
            if (!tm)
                return std::nullopt;
            const auto& error = tm->getLastError();
            return error.empty() ? std::nullopt : std::make_optional(error);
        },
        "Get trainer error message");

    m.def(
        "start_training", []() {
            nb::gil_scoped_release release;
            lfs::core::events::cmd::StartTraining{}.emit();
        },
        "Start training with current parameters");
    m.def(
        "pause_training", []() {
            nb::gil_scoped_release release;
            lfs::core::events::cmd::PauseTraining{}.emit();
        },
        "Pause the current training run");
    m.def(
        "resume_training", []() {
            nb::gil_scoped_release release;
            lfs::core::events::cmd::ResumeTraining{}.emit();
        },
        "Resume a paused training run");
    m.def(
        "stop_training", []() {
            nb::gil_scoped_release release;
            lfs::core::events::cmd::StopTraining{}.emit();
        },
        "Stop the current training run");
    m.def(
        "reset_training", []() {
            nb::gil_scoped_release release;
            lfs::core::events::cmd::ResetTraining{}.emit();
        },
        "Reset training state to initial");
    m.def(
        "save_checkpoint", []() { lfs::core::events::cmd::SaveCheckpoint{}.emit(); },
        "Save a training checkpoint to disk");
    m.def(
        "clear_scene", []() {
            nb::gil_scoped_release release;
            lfs::core::events::cmd::ClearScene{}.emit();
        },
        "Remove all nodes from the scene");
    m.def(
        "switch_to_edit_mode", []() { lfs::core::events::cmd::SwitchToEditMode{}.emit(); },
        "Switch from training to edit mode");

    m.def(
        "load_file",
        [](const std::string& path, const bool is_dataset,
           const std::string& output_path, const std::string& init_path) {
            nb::gil_scoped_release release;
            lfs::core::events::cmd::LoadFile{
                .path = path,
                .is_dataset = is_dataset,
                .output_path = output_path,
                .init_path = init_path}
                .emit();
        },
        nb::arg("path"), nb::arg("is_dataset") = false,
        nb::arg("output_path") = "", nb::arg("init_path") = "",
        "Load a file (PLY, checkpoint) or dataset into the scene.");

    m.def(
        "load_config_file",
        [](const std::string& path) { lfs::core::events::cmd::LoadConfigFile{.path = path}.emit(); },
        nb::arg("path"), "Load a JSON configuration file.");

    m.def(
        "load_checkpoint_for_training",
        [](const std::string& checkpoint_path, const std::string& dataset_path, const std::string& output_path) {
            nb::gil_scoped_release release;
            lfs::core::events::cmd::LoadCheckpointForTraining{
                .checkpoint_path = checkpoint_path,
                .dataset_path = dataset_path,
                .output_path = output_path,
            }
                .emit();
        },
        nb::arg("checkpoint_path"), nb::arg("dataset_path"), nb::arg("output_path"),
        "Load a checkpoint for training with specified dataset and output paths.");

    m.def(
        "request_exit", []() { lfs::core::events::cmd::RequestExit{}.emit(); },
        "Request application exit (shows confirmation if needed).");

    m.def(
        "force_exit", []() { lfs::core::events::cmd::ForceExit{}.emit(); },
        "Force immediate application exit (bypasses confirmation).");

    m.def(
        "export_scene",
        [](int format, const std::string& path, const std::vector<std::string>& node_names, int sh_degree) {
            lfs::python::invoke_export(format, path, node_names, sh_degree);
        },
        nb::arg("format"), nb::arg("path"), nb::arg("node_names"), nb::arg("sh_degree"),
        "Export scene nodes to file. Format: 0=PLY, 1=SOG, 2=SPZ, 3=HTML.");

    m.def(
        "save_config_file",
        [](const std::string& path) {
            const auto* const param_manager = lfs::vis::services().paramsOrNull();
            if (!param_manager) {
                throw std::runtime_error("No parameter manager available");
            }
            lfs::core::param::TrainingParameters params;
            params.dataset = param_manager->getDatasetConfig();
            params.optimization = param_manager->copyActiveParams();
            if (const auto result = lfs::core::param::save_training_parameters_to_json(params, path); !result) {
                throw std::runtime_error("Failed to save config: " + result.error());
            }
        },
        nb::arg("path"), "Save current training configuration to a JSON file.");

    m.def(
        "has_trainer", []() {
            const auto* const tm = lfs::python::get_trainer_manager();
            return tm && tm->hasTrainer();
        },
        "Check if a trainer instance exists");

    m.def(
        "loss_buffer", []() -> std::vector<float> {
            const auto* const tm = lfs::python::get_trainer_manager();
            if (!tm)
                return {};
            auto loss_deque = tm->getLossBuffer();
            return std::vector<float>(loss_deque.begin(), loss_deque.end());
        },
        "Get the recent loss history as a list of floats");

    // Trainer status bar bindings
    m.def(
        "trainer_elapsed_seconds", []() -> float {
            const auto* tm = lfs::python::get_trainer_manager();
            return tm ? tm->getElapsedSeconds() : 0.0f;
        },
        "Get elapsed training time in seconds");

    m.def(
        "trainer_eta_seconds", []() -> float {
            const auto* tm = lfs::python::get_trainer_manager();
            return tm ? tm->getEstimatedRemainingSeconds() : -1.0f;
        },
        "Get estimated remaining time in seconds (-1 if unavailable)");

    m.def(
        "trainer_strategy_type", []() -> const char* {
            const auto* tm = lfs::python::get_trainer_manager();
            return (tm && tm->hasTrainer()) ? tm->getStrategyType() : "unknown";
        },
        "Get training strategy type (mcmc, default, etc.)");

    m.def(
        "trainer_is_gut_enabled", []() -> bool {
            const auto* tm = lfs::python::get_trainer_manager();
            return tm && tm->hasTrainer() && tm->isGutEnabled();
        },
        "Check if GUT is enabled");

    m.def(
        "trainer_max_gaussians", []() -> int {
            const auto* tm = lfs::python::get_trainer_manager();
            return tm ? tm->getMaxGaussians() : 0;
        },
        "Get maximum number of gaussians");

    m.def(
        "trainer_num_splats", []() -> int {
            const auto* tm = lfs::python::get_trainer_manager();
            return tm ? tm->getNumSplats() : 0;
        },
        "Get current number of splats");

    m.def(
        "trainer_current_iteration", []() -> int {
            const auto* tm = lfs::python::get_trainer_manager();
            return tm ? tm->getCurrentIteration() : 0;
        },
        "Get current iteration");

    m.def(
        "trainer_total_iterations", []() -> int {
            const auto* tm = lfs::python::get_trainer_manager();
            return tm ? tm->getTotalIterations() : 0;
        },
        "Get total iterations");

    m.def(
        "trainer_current_loss", []() -> float {
            const auto* tm = lfs::python::get_trainer_manager();
            return tm ? tm->getCurrentLoss() : 0.0f;
        },
        "Get current loss");

    // Scene manipulation
    m.def(
        "set_node_visibility", [](const std::string& name, bool visible) {
            lfs::core::events::cmd::SetPLYVisibility{.name = name, .visible = visible}.emit();
        },
        nb::arg("name"), nb::arg("visible"), "Set visibility of a scene node by name");

    m.def(
        "set_camera_training_enabled", [](const std::string& name, bool enabled) {
            auto* scene = get_scene_internal();
            if (!scene)
                return;
            scene->setCameraTrainingEnabled(name, enabled);
        },
        nb::arg("name"), nb::arg("enabled"), "Enable or disable a camera for training by name");

    m.def(
        "remove_node", [](const std::string& name, bool keep_children) {
            lfs::core::events::cmd::RemovePLY{.name = name, .keep_children = keep_children}.emit();
        },
        nb::arg("name"), nb::arg("keep_children") = false, "Remove a scene node by name");

    m.def(
        "select_node", [](const std::string& name) {
            lfs::core::events::ui::NodeSelected{
                .path = name,
                .type = "PLY",
                .metadata = {{"name", name}}}
                .emit();
        },
        nb::arg("name"), "Select a scene node by name");

    m.def(
        "add_to_selection", [](const std::string& name) {
            auto* sm = lfs::python::get_scene_manager();
            if (sm)
                sm->addToSelection(name);
        },
        nb::arg("name"), "Add a node to the current selection");

    m.def(
        "select_nodes", [](const std::vector<std::string>& names) {
            auto* sm = lfs::python::get_scene_manager();
            if (sm)
                sm->selectNodes(names);
        },
        nb::arg("names"), "Select multiple nodes at once");

    m.def(
        "deselect_all", []() {
            lfs::core::events::ui::NodeDeselected{}.emit();
        },
        "Deselect all scene nodes");

    m.def(
        "reparent_node", [](const std::string& name, const std::string& new_parent) {
            lfs::core::events::cmd::ReparentNode{.node_name = name, .new_parent_name = new_parent}.emit();
        },
        nb::arg("name"), nb::arg("new_parent"), "Move a node under a new parent node");

    m.def(
        "rename_node", [](const std::string& old_name, const std::string& new_name) {
            lfs::core::events::cmd::RenamePLY{.old_name = old_name, .new_name = new_name}.emit();
        },
        nb::arg("old_name"), nb::arg("new_name"), "Rename a scene node");

    m.def(
        "add_group", [](const std::string& name, const std::string& parent) {
            lfs::core::events::cmd::AddGroup{.name = name, .parent_name = parent}.emit();
        },
        nb::arg("name"), nb::arg("parent") = "", "Add a group node to the scene");

    // Transform APIs for operator system
    m.def(
        "get_selected_node_transform", []() -> std::optional<std::vector<float>> {
            auto* sm = lfs::python::get_scene_manager();
            if (!sm || !sm->hasSelectedNode())
                return std::nullopt;
            const auto m = sm->getSelectedNodeTransform();
            return std::vector<float>(&m[0][0], &m[0][0] + 16);
        },
        "Get transform matrix (16 floats, column-major) of selected node");

    m.def(
        "set_selected_node_transform", [](const std::vector<float>& mat) {
            auto* sm = lfs::python::get_scene_manager();
            if (!sm || !sm->hasSelectedNode() || mat.size() != 16)
                return;
            glm::mat4 m;
            std::memcpy(&m[0][0], mat.data(), 16 * sizeof(float));
            sm->setSelectedNodeTransform(m);
        },
        nb::arg("matrix"), "Set transform matrix (16 floats, column-major) of selected node");

    m.def(
        "get_selection_center", []() -> std::optional<std::vector<float>> {
            auto* sm = lfs::python::get_scene_manager();
            if (!sm || !sm->hasSelectedNode())
                return std::nullopt;
            const auto c = sm->getSelectionCenter();
            return std::vector<float>{c.x, c.y, c.z};
        },
        "Get center of current selection (local space)");

    m.def(
        "get_selection_world_center", []() -> std::optional<std::vector<float>> {
            auto* sm = lfs::python::get_scene_manager();
            if (!sm || !sm->hasSelectedNode())
                return std::nullopt;
            const auto c = sm->getSelectionWorldCenter();
            return std::vector<float>{c.x, c.y, c.z};
        },
        "Get center of current selection (world space)");

    m.def(
        "has_scene", []() -> bool {
            auto* sm = lfs::python::get_scene_manager();
            return sm && sm->getScene().getNodeCount() > 0;
        },
        "Check if a scene is loaded");

    m.def(
        "has_selection", []() -> bool {
            auto* sm = lfs::python::get_scene_manager();
            return sm && sm->hasSelectedNode();
        },
        "Check if a node is selected");

    m.def(
        "get_selected_node_names", []() -> std::vector<std::string> {
            auto* sm = lfs::python::get_scene_manager();
            if (!sm)
                return {};
            return sm->getSelectedNodeNames();
        },
        "Get names of all selected nodes");

    m.def(
        "get_selected_node_name", []() -> std::string {
            auto* sm = lfs::python::get_scene_manager();
            if (!sm || !sm->hasSelectedNode())
                return "";
            return sm->getSelectedNodeName();
        },
        "Get name of primary selected node");

    m.def(
        "can_transform_selection", []() -> bool {
            auto* ec = lfs::python::get_editor_context();
            return ec && ec->canTransformSelectedNode();
        },
        "Check if selected node can be transformed");

    m.def(
        "get_num_gaussians", []() -> size_t {
            auto* sm = lfs::python::get_scene_manager();
            if (!sm)
                return 0;
            const auto* model = sm->getScene().getCombinedModel();
            return model ? model->size() : 0;
        },
        "Get total number of gaussians in scene");

    m.def(
        "get_node_transform", [](const std::string& name) -> std::optional<std::vector<float>> {
            const auto* sm = lfs::python::get_scene_manager();
            if (!sm)
                return std::nullopt;
            const auto& mat = sm->getNodeTransform(name);
            return std::vector<float>(&mat[0][0], &mat[0][0] + 16);
        },
        nb::arg("name"), "Get node transform matrix (16 floats, column-major)");

    m.def(
        "set_node_transform", [](const std::string& name, const std::vector<float>& mat) {
            auto* sm = lfs::python::get_scene_manager();
            if (!sm || mat.size() != 16)
                return;
            glm::mat4 transform;
            std::memcpy(&transform[0][0], mat.data(), 16 * sizeof(float));
            sm->setNodeTransform(name, transform);
        },
        nb::arg("name"), nb::arg("matrix"), "Set node transform matrix (16 floats, column-major)");

    m.def(
        "capture_selection_transforms", []() -> nb::dict {
            const auto* sm = lfs::python::get_scene_manager();
            nb::dict result;
            if (!sm)
                return result;

            const auto& names = sm->getSelectedNodeNames();
            std::vector<std::vector<float>> transforms;
            std::vector<std::vector<float>> world_positions;
            transforms.reserve(names.size());
            world_positions.reserve(names.size());

            for (const auto& name : names) {
                const auto& mat = sm->getNodeTransform(name);
                transforms.emplace_back(&mat[0][0], &mat[0][0] + 16);
                world_positions.push_back({mat[3][0], mat[3][1], mat[3][2]});
            }

            result["names"] = names;
            result["transforms"] = transforms;
            result["world_positions"] = world_positions;
            return result;
        },
        "Capture transforms of all selected nodes");

    m.def(
        "decompose_transform", [](const std::vector<float>& mat) -> nb::dict {
            nb::dict result;
            if (mat.size() != 16)
                return result;

            glm::mat4 m;
            std::memcpy(&m[0][0], mat.data(), 16 * sizeof(float));

            const glm::vec3 translation(m[3]);

            glm::vec3 col0(m[0]), col1(m[1]), col2(m[2]);
            const glm::vec3 scale(glm::length(col0), glm::length(col1), glm::length(col2));

            if (scale.x > 0.0f)
                col0 /= scale.x;
            if (scale.y > 0.0f)
                col1 /= scale.y;
            if (scale.z > 0.0f)
                col2 /= scale.z;

            const glm::mat3 rot_mat(col0, col1, col2);
            const glm::quat quat = glm::quat_cast(rot_mat);

            glm::vec3 euler_rad;
            glm::extractEulerAngleXYZ(glm::mat4(rot_mat), euler_rad.x, euler_rad.y, euler_rad.z);
            const glm::vec3 euler_deg = glm::degrees(euler_rad);

            result["translation"] = std::vector<float>{translation.x, translation.y, translation.z};
            result["rotation_quat"] = std::vector<float>{quat.x, quat.y, quat.z, quat.w};
            result["rotation_euler"] = std::vector<float>{euler_rad.x, euler_rad.y, euler_rad.z};
            result["rotation_euler_deg"] = std::vector<float>{euler_deg.x, euler_deg.y, euler_deg.z};
            result["scale"] = std::vector<float>{scale.x, scale.y, scale.z};
            return result;
        },
        nb::arg("matrix"), "Decompose transform into translation, rotation, scale");

    m.def(
        "compose_transform", [](const std::vector<float>& translation, const std::vector<float>& euler_deg, const std::vector<float>& scale) -> std::vector<float> {
            if (translation.size() != 3 || euler_deg.size() != 3 || scale.size() != 3)
                return std::vector<float>(16, 0.0f);

            const glm::vec3 trans(translation[0], translation[1], translation[2]);
            const glm::vec3 euler_rad = glm::radians(glm::vec3(euler_deg[0], euler_deg[1], euler_deg[2]));
            const glm::vec3 s(scale[0], scale[1], scale[2]);

            const glm::mat4 t = glm::translate(glm::mat4(1.0f), trans);
            const glm::mat4 r = glm::eulerAngleXYZ(euler_rad.x, euler_rad.y, euler_rad.z);
            const glm::mat4 sc = glm::scale(glm::mat4(1.0f), s);
            const glm::mat4 m = t * r * sc;

            return std::vector<float>(&m[0][0], &m[0][0] + 16);
        },
        nb::arg("translation"), nb::arg("euler_deg"), nb::arg("scale"), "Compose transform matrix from translation, euler angles (degrees), and scale");

    // Icon loading for Python UI
    m.def(
        "load_icon", [](const std::string& name) -> uint64_t {
            try {
                const auto path = lfs::vis::getAssetPath("icon/" + name + ".png");
                const auto [data, width, height, channels] = lfs::core::load_image_with_alpha(path);

                const auto result = lfs::python::create_gl_texture(data, width, height, channels);
                lfs::core::free_image(data);

                return static_cast<uint64_t>(result.texture_id);
            } catch (const std::exception& e) {
                LOG_WARN("Failed to load icon {}: {}", name, e.what());
                return 0;
            }
        },
        nb::arg("name"), "Load an icon texture from assets/icon/{name}.png, returns OpenGL texture ID");

    m.def(
        "free_icon", [](const uint64_t texture_id) {
            if (texture_id > 0)
                lfs::python::delete_gl_texture(static_cast<uint32_t>(texture_id));
        },
        nb::arg("texture_id"), "Free an icon texture");

    // Camera commands
    m.def(
        "reset_camera", []() { lfs::core::events::cmd::ResetCamera{}.emit(); },
        "Reset camera to default position and orientation");
    m.def(
        "toggle_fullscreen", []() { lfs::core::events::ui::ToggleFullscreen{}.emit(); },
        "Toggle fullscreen mode");
    m.def(
        "is_fullscreen", []() -> bool {
            auto* wm = lfs::vis::services().windowOrNull();
            return wm ? wm->isFullscreen() : false;
        },
        "Check if the window is in fullscreen mode");
    m.def(
        "toggle_ui", []() { lfs::core::events::ui::ToggleUI{}.emit(); },
        "Toggle UI overlay visibility");

    m.def(
        "get_render_mode", []() -> RenderMode {
            const auto* rm = lfs::python::get_rendering_manager();
            if (!rm)
                return RenderMode::Splats;
            const auto& settings = rm->getSettings();
            if (settings.point_cloud_mode)
                return RenderMode::Points;
            if (settings.show_rings)
                return RenderMode::Rings;
            if (settings.show_center_markers)
                return RenderMode::Centers;
            return RenderMode::Splats;
        },
        "Get current render mode (Splats, Points, Rings, Centers)");

    m.def(
        "set_render_mode", [](RenderMode mode) {
            auto* rm = lfs::python::get_rendering_manager();
            if (!rm)
                return;
            auto settings = rm->getSettings();
            settings.point_cloud_mode = (mode == RenderMode::Points);
            settings.show_rings = (mode == RenderMode::Rings);
            settings.show_center_markers = (mode == RenderMode::Centers);
            rm->updateSettings(settings);
        },
        nb::arg("mode"), "Set the render mode (Splats, Points, Rings, Centers)");

    m.def(
        "is_orthographic", []() -> bool {
            const auto* rm = lfs::python::get_rendering_manager();
            return rm ? rm->getSettings().orthographic : false;
        },
        "Check if orthographic projection is active");

    m.def(
        "set_orthographic", [](bool ortho) {
            auto* rm = lfs::python::get_rendering_manager();
            if (!rm)
                return;
            auto settings = rm->getSettings();
            settings.orthographic = ortho;
            rm->updateSettings(settings);
        },
        nb::arg("ortho"), "Enable or disable orthographic projection");

    // Hook registration functions (decorator-style)
    m.def(
        "on_training_start", [](nb::callable cb) {
            register_hook(ControlHook::TrainingStart, cb);
            return cb;
        },
        nb::arg("callback"), "Decorator for training start handler");

    m.def(
        "on_iteration_start", [](nb::callable cb) {
            register_hook(ControlHook::IterationStart, cb);
            return cb;
        },
        nb::arg("callback"), "Decorator for iteration start handler");

    m.def(
        "on_post_step", [](nb::callable cb) {
            register_hook(ControlHook::PostStep, cb);
            return cb;
        },
        nb::arg("callback"), "Decorator for post-step handler");

    m.def(
        "on_pre_optimizer_step", [](nb::callable cb) {
            register_hook(ControlHook::PreOptimizerStep, cb);
            return cb;
        },
        nb::arg("callback"), "Decorator for pre-optimizer handler");

    m.def(
        "on_training_end", [](nb::callable cb) {
            register_hook(ControlHook::TrainingEnd, cb);
            return cb;
        },
        nb::arg("callback"), "Decorator for training end handler");

    // Register Tensor class
    lfs::python::register_tensor(m);

    // Scene submodule
    auto scene_module = m.def_submodule("scene", "Scene graph API");
    lfs::python::register_splat_data(scene_module);
    lfs::python::register_scene(scene_module);
    lfs::python::register_cameras(scene_module);

    // Rendering functions (render_view, compute_screen_positions, etc.)
    lfs::python::register_rendering(m);

    // Selection primitives for Python operators
    lfs::python::register_selection(m);

    // I/O submodule
    auto io_module = m.def_submodule("io", "File I/O operations");
    lfs::python::register_io(io_module);

    // Packages submodule (uses uv for package management)
    lfs::python::register_packages(m);

    // UI submodule (ImGui widgets and panel registration)
    auto ui_module = m.def_submodule("ui", "User interface API");
    lfs::python::register_ui(ui_module);

    // Signal bridge for reactive UI updates
    lfs::python::register_signals(ui_module);

    // Set up notification handlers (C++ events -> PyModalRegistry)
    lfs::python::setup_notification_handlers();

    // Keymap submodule (at root level for lf.keymap access)
    lfs::python::register_keymap(m);

    // Unified class registration API on root module
    lfs::python::register_class_api(m);

    // Build info submodule
    auto build_info = m.def_submodule("build_info", "Build configuration and version information");
    build_info.attr("version") = GIT_TAGGED_VERSION;
    build_info.attr("commit") = GIT_COMMIT_HASH_SHORT;
#ifdef DEBUG_BUILD
    build_info.attr("build_type") = "Debug";
#else
    build_info.attr("build_type") = "Release";
#endif
#ifdef PLATFORM_WINDOWS
    build_info.attr("platform") = "Windows";
#elif defined(PLATFORM_LINUX)
    build_info.attr("platform") = "Linux";
#else
    build_info.attr("platform") = "Unknown";
#endif
#ifdef CUDA_GL_INTEROP_ENABLED
    build_info.attr("cuda_gl_interop") = true;
#else
    build_info.attr("cuda_gl_interop") = false;
#endif
    build_info.attr("repo_url") = "https://github.com/MrNeRF/LichtFeld-Studio";
    build_info.attr("website_url") = "https://lichtfeld.io";

    lfs::python::register_commands(m);
    lfs::python::register_gizmos(m);
    lfs::python::register_operator_return_value(m);
    lfs::python::register_operators(m);
    lfs::python::register_pipeline(m);
    lfs::python::register_uilist(m);
    lfs::python::register_viewport(m);

    // Parameters (OptimizationParameters with RNA-style property access)
    lfs::python::register_params(m);

    // Plugin system (pure Python implementation, C++ bindings for convenience)
    lfs::python::register_plugins(m);

    // Script management (for Python scripts panel)
    lfs::python::register_scripts(m);

    // MCP (Model Context Protocol) tool registration
    lfs::python::register_mcp(m);

    // Animation submodule (multi-track property animation)
    auto anim_module = m.def_submodule("animation", "Animation system API");
    lfs::python::register_animation(anim_module);

    // Logging submodule
    auto log_module = m.def_submodule("log", "Logging utilities");
    log_module.def(
        "info",
        [](const std::string& msg) { LOG_INFO("[Python] {}", msg); },
        nb::arg("message"),
        "Log an info message");
    log_module.def(
        "debug",
        [](const std::string& msg) { LOG_DEBUG("[Python] {}", msg); },
        nb::arg("message"),
        "Log a debug message");
    log_module.def(
        "warn",
        [](const std::string& msg) { LOG_WARN("[Python] {}", msg); },
        nb::arg("message"),
        "Log a warning message");
    log_module.def(
        "error",
        [](const std::string& msg) { LOG_ERROR("[Python] {}", msg); },
        nb::arg("message"),
        "Log an error message");

    auto app_module = m.def_submodule("app", "Application-level operations");
    app_module.def(
        "open",
        [](const std::string& path_str) {
            namespace cmd = lfs::core::events::cmd;
            cmd::LoadFile{.path = path_str, .is_dataset = true}.emit();
        },
        nb::arg("path"),
        "Open a dataset or file in the application");

    // Get scene function - works in both headless (during hooks) and GUI mode
    m.def(
        "get_scene", []() -> std::optional<lfs::python::PyScene> {
            auto* scene = get_scene_internal();
            if (!scene) {
                return std::nullopt;
            }
            return lfs::python::PyScene(scene);
        },
        "Get the current scene (None if not available)");

    // Get scene generation counter for validity checking
    m.def(
        "get_scene_generation", []() -> uint64_t {
            return lfs::python::get_scene_generation();
        },
        "Get current scene generation counter (for validity checking)");

    // Run a Python script file
    m.def(
        "run", [](const std::string& path) {
            const std::filesystem::path script_path(path);
            if (!std::filesystem::exists(script_path)) {
                throw std::runtime_error("Script not found: " + path);
            }

            std::ifstream file(script_path);
            if (!file) {
                throw std::runtime_error("Cannot open script: " + path);
            }

            std::stringstream buffer;
            buffer << file.rdbuf();
            const std::string code = buffer.str();

            // Add script directory to sys.path and set __file__
            const auto parent = script_path.parent_path().string();
            const auto abs_path = std::filesystem::absolute(script_path).string();

            // Get Python's exec function
            nb::object builtins = nb::module_::import_("builtins");
            nb::object py_exec = builtins.attr("exec");

            std::string setup_code = std::format(
                "import sys\n"
                "if '{}' not in sys.path: sys.path.insert(0, '{}')\n"
                "__file__ = '{}'\n",
                parent, parent, abs_path);

            py_exec(setup_code);
            py_exec(code);

            LOG_INFO("Executed script: {}", path);
        },
        nb::arg("path"), "Execute a Python script file");

    // List scene contents
    m.def(
        "list_scene", []() {
            auto* scene = get_scene_internal();
            if (!scene) {
                nb::print("No scene available");
                return;
            }

            const auto nodes = scene->getNodes();
            nb::print(nb::str("Scene: {} nodes, {} gaussians\n").format(nodes.size(), scene->getTotalGaussianCount()));

            // Build id->node map
            std::unordered_map<int32_t, const lfs::vis::SceneNode*> node_map;
            for (const auto* n : nodes) {
                node_map[n->id] = n;
            }

            // Recursive print function
            std::function<void(const lfs::vis::SceneNode*, int)> print_node =
                [&](const lfs::vis::SceneNode* node, int depth) {
                    std::string indent(depth * 2, ' ');
                    char vis = node->visible ? '+' : '-';
                    char lock = node->locked ? 'L' : ' ';

                    std::string type_name;
                    switch (node->type) {
                    case lfs::vis::NodeType::SPLAT: type_name = "SPLAT"; break;
                    case lfs::vis::NodeType::POINTCLOUD: type_name = "POINTCLOUD"; break;
                    case lfs::vis::NodeType::GROUP: type_name = "GROUP"; break;
                    case lfs::vis::NodeType::CROPBOX: type_name = "CROPBOX"; break;
                    case lfs::vis::NodeType::ELLIPSOID: type_name = "ELLIPSOID"; break;
                    case lfs::vis::NodeType::DATASET: type_name = "DATASET"; break;
                    case lfs::vis::NodeType::CAMERA_GROUP: type_name = "CAMERA_GROUP"; break;
                    case lfs::vis::NodeType::CAMERA: type_name = "CAMERA"; break;
                    case lfs::vis::NodeType::IMAGE_GROUP: type_name = "IMAGE_GROUP"; break;
                    case lfs::vis::NodeType::IMAGE: type_name = "IMAGE"; break;
                    default: type_name = "UNKNOWN"; break;
                    }

                    std::string info = std::format("[{}{}] {} ({}, id={})",
                                                   vis, lock, node->name, type_name, node->id);

                    if (node->gaussian_count > 0) {
                        info += std::format(" [{} splats]", node->gaussian_count);
                    }

                    nb::print(nb::str("{}{}").format(indent, info));

                    for (int32_t child_id : node->children) {
                        auto it = node_map.find(child_id);
                        if (it != node_map.end()) {
                            print_node(it->second, depth + 1);
                        }
                    }
                };

            // Print root nodes
            for (const auto* node : nodes) {
                if (node->parent_id == lfs::vis::NULL_NODE) {
                    print_node(node, 0);
                }
            }
        },
        "Print the scene graph tree");

    // Frame callback for animations
    m.def(
        "on_frame", [](nb::callable cb) {
            nb::object ocb = nb::cast<nb::object>(cb);
            lfs::python::set_frame_callback([ocb](float dt) {
                try {
                    ocb(dt);
                } catch (const std::exception& e) {
                    LOG_ERROR("on_frame callback error: {}", e.what());
                    lfs::python::clear_frame_callback();
                }
            });
            LOG_INFO("Frame callback registered");
        },
        nb::arg("callback"), "Register a callback to be called each frame with delta time (seconds)");

    m.def(
        "stop_animation", []() {
            lfs::python::clear_frame_callback();
            LOG_INFO("Frame callback cleared");
        },
        "Stop any running animation (clears frame callback)");

    // GPU colormap: maps [N] normalized values in [0,1] to [N,3] RGB via jet colormap
    m.def(
        "colormap", [](const lfs::python::PyTensor& values, const std::string& name) -> lfs::python::PyTensor {
            if (name != "jet") {
                throw std::runtime_error("Only 'jet' colormap is currently supported");
            }
            const auto& t = values.tensor();
            assert(t.shape().rank() == 1);

            auto v = t.clamp(0.0f, 1.0f);

            using T = lfs::core::Tensor;

            // Jet: piecewise linear through Blue->Cyan->Green->Yellow->Red
            T r = ((v - 0.375f) * 4.0f).clamp(0.0f, 1.0f) - ((v - 0.875f) * 4.0f).clamp(0.0f, 1.0f);
            T g = ((v - 0.125f) * 4.0f).clamp(0.0f, 1.0f) - ((v - 0.625f) * 4.0f).clamp(0.0f, 1.0f);
            T one = T::ones_like(v);
            T b = (one - (v - 0.125f) * 4.0f).clamp(0.0f, 1.0f);

            std::vector<T> channels = {r, g, b};
            T result = T::stack(channels, 1);
            return lfs::python::PyTensor(std::move(result), true);
        },
        nb::arg("values"), nb::arg("name") = "jet", "Apply colormap to [N] values in [0,1], returns [N,3] RGB tensor on same device");

    // Create 4x4 matrix tensor from nested list (for transforms)
    m.def(
        "mat4", [](const std::vector<std::vector<float>>& rows) -> lfs::python::PyTensor {
            if (rows.size() != 4) {
                throw std::runtime_error("mat4 requires 4 rows");
            }
            for (const auto& row : rows) {
                if (row.size() != 4) {
                    throw std::runtime_error("mat4 requires 4 columns per row");
                }
            }
            auto tensor = lfs::core::Tensor::empty({4, 4}, lfs::core::Device::CPU, lfs::core::DataType::Float32);
            float* data = tensor.ptr<float>();
            for (int i = 0; i < 4; ++i) {
                for (int j = 0; j < 4; ++j) {
                    data[i * 4 + j] = rows[i][j];
                }
            }
            return lfs::python::PyTensor(std::move(tensor));
        },
        nb::arg("rows"), "Create a 4x4 matrix tensor from nested list [[r0], [r1], [r2], [r3]]");

    // Quick help
    m.def(
        "help", []() {
            nb::print(R"(LichtFeld Python API

Scene:
  lf.get_scene()      - Get the scene object (None if unavailable)
  lf.list_scene()     - Print scene graph tree
  scene.add_camera()  - Add camera node with R, T, focal, dimensions

Training:
  lf.context()        - Get training context (iteration, loss, etc.)
  lf.gaussians()      - Get gaussians info (count, sh_degree)
  lf.session()        - Get session for pause/resume/optimizer control

Hooks (decorators):
  @lf.on_training_start
  @lf.on_iteration_start
  @lf.on_post_step
  @lf.on_pre_optimizer_step
  @lf.on_training_end

Plugin Hooks (RAII):
  handler = lf.ScopedHandler()
  handler.on_iteration_start(callback)
  # Auto-unregisters when handler is destroyed

Utilities:
  lf.run("script.py") - Execute a Python script file
  lf.help()           - Show this help

Example:
  scene = lf.get_scene()
  for node in scene.get_nodes():
      print(node.name, node.type)
)");
        },
        "Show help for lichtfeld module");

    // Internal context API for GUI console (C++ calls these before/after executing user code)
    m.def(
        "_set_scene_context", [](nb::capsule scene_capsule) {
            auto* scene = static_cast<lfs::vis::Scene*>(scene_capsule.data());
            lfs::python::set_scene_for_python(scene);
        },
        nb::arg("scene_capsule"), "Internal: Set scene context for console execution");

    m.def(
        "_clear_scene_context", []() { lfs::python::set_scene_for_python(nullptr); },
        "Internal: Clear scene context after console execution");

    nb::class_<lfs::io::DatasetInfo>(m, "DatasetInfo", "Information about a dataset directory")
        .def_prop_ro(
            "base_path", [](const lfs::io::DatasetInfo& i) { return i.base_path.string(); },
            "Root directory of the dataset")
        .def_prop_ro(
            "images_path", [](const lfs::io::DatasetInfo& i) { return i.images_path.string(); },
            "Path to the images directory")
        .def_prop_ro(
            "sparse_path", [](const lfs::io::DatasetInfo& i) { return i.sparse_path.string(); },
            "Path to the COLMAP sparse reconstruction")
        .def_prop_ro(
            "masks_path", [](const lfs::io::DatasetInfo& i) { return i.masks_path.string(); },
            "Path to the masks directory")
        .def_prop_ro(
            "has_masks", [](const lfs::io::DatasetInfo& i) { return i.has_masks; },
            "Whether the dataset includes masks")
        .def_prop_ro(
            "image_count", [](const lfs::io::DatasetInfo& i) { return i.image_count; },
            "Number of images in the dataset")
        .def_prop_ro(
            "mask_count", [](const lfs::io::DatasetInfo& i) { return i.mask_count; },
            "Number of masks in the dataset")
        .def("__repr__", [](const lfs::io::DatasetInfo& i) {
            return std::format("DatasetInfo(base_path='{}', images={}, masks={})",
                               i.base_path.string(), i.image_count, i.mask_count);
        });

    m.def(
        "detect_dataset_info",
        [](const std::string& path) { return lfs::io::detect_dataset_info(path); },
        nb::arg("path"), "Detect dataset information from a directory path");

    nb::class_<lfs::core::CheckpointHeader>(m, "CheckpointHeader", "Information from a checkpoint file header")
        .def_ro("iteration", &lfs::core::CheckpointHeader::iteration)
        .def_ro("num_gaussians", &lfs::core::CheckpointHeader::num_gaussians)
        .def_ro("sh_degree", &lfs::core::CheckpointHeader::sh_degree)
        .def("__repr__", [](const lfs::core::CheckpointHeader& h) {
            return std::format("CheckpointHeader(iteration={}, num_gaussians={}, sh_degree={})",
                               h.iteration, h.num_gaussians, h.sh_degree);
        });

    m.def(
        "read_checkpoint_header",
        [](const std::string& path) -> std::optional<lfs::core::CheckpointHeader> {
            auto result = lfs::core::load_checkpoint_header(path);
            if (!result) {
                return std::nullopt;
            }
            return *result;
        },
        nb::arg("path"), "Read checkpoint header information (None if failed)");

    nb::class_<lfs::core::param::DatasetConfig>(m, "CheckpointParams", "Training parameters from a checkpoint")
        .def_prop_ro("dataset_path", [](const lfs::core::param::DatasetConfig& c) {
            return c.data_path.string();
        })
        .def_prop_ro("output_path", [](const lfs::core::param::DatasetConfig& c) {
            return c.output_path.string();
        })
        .def("__repr__", [](const lfs::core::param::DatasetConfig& c) {
            return std::format("CheckpointParams(dataset_path='{}', output_path='{}')",
                               c.data_path.string(), c.output_path.string());
        });

    m.def(
        "read_checkpoint_params",
        [](const std::string& path) -> std::optional<lfs::core::param::DatasetConfig> {
            auto result = lfs::core::load_checkpoint_params(path);
            if (!result) {
                return std::nullopt;
            }
            return result->dataset;
        },
        nb::arg("path"), "Read training parameters from a checkpoint (None if failed)");

    lfs::python::set_invalidate_poll_cache_callback([](uint8_t dependency) {
        const auto dep = static_cast<lfs::vis::op::PollDependency>(dependency);
        lfs::vis::op::OperatorRegistry::instance().invalidatePollCache(dep);
        lfs::vis::gui::PanelRegistry::instance().invalidate_poll_cache(dep);
    });

    // Module metadata
    m.attr("__version__") = "0.1.0";
    m.attr("__all__") = nb::make_tuple(
        // Core access
        "context", "gaussians", "session", "get_scene",
        // Types
        "Tensor", "Hook", "ScopedHandler",
        // Hook decorators
        "on_training_start", "on_iteration_start",
        "on_post_step", "on_pre_optimizer_step", "on_training_end",
        // Animation
        "on_frame", "stop_animation",
        // Utilities
        "run", "list_scene", "mat4", "colormap", "help",
        // Submodules
        "scene", "io", "packages", "mcp");
}
